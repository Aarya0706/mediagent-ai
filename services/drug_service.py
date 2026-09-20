"""
services/drug_service.py
---------------------------
Drug interaction checking, extracted from app.py where it was one
contiguous block (drug name normalization, OpenFDA evidence lookup,
and the LLM prompt that turns raw OpenFDA data into a plain-English
explanation) sitting between Tab 4 (Doctor Portal) and Tab 5 (Drug
Checker) purely because that's where it was first written.

No behavior changes - same OpenFDA query logic, same evidence
extraction, same prompt. Two renames only, both mechanical:
  _query_openfda   -> query_openfda    (public API of this module now)
  _parse_drug_field -> parse_drug_field
`build_drug_chain(llm)` replaces the old `_drug_chain = _drug_prompt |
_drug_llm | StrOutputParser()` module-level line - same chain, just
built from whichever LLM the caller passes in (services/llm_clients.py)
rather than a global captured at import time.
"""

import requests
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

DRUG_NAME_ALIASES = {
    "pacemol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "crocin": "acetaminophen",
    "calpol": "acetaminophen",
    "dolo": "acetaminophen",
    "dolo 650": "acetaminophen",

    "ecosprin": "aspirin",
    "disprin": "aspirin",

    "brufen": "ibuprofen",
    "advil": "ibuprofen",

    "augmentin": "amoxicillin clavulanate",
    "amoxyclav": "amoxicillin clavulanate",

    "zithromax": "azithromycin",
    "azee": "azithromycin",
}

DRUG_INTERACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist explaining drug interactions to a patient in plain English.
You will receive raw OpenFDA adverse event data about two drugs taken together.
Your job is to summarise what risks exist, how serious they are, and what the patient should do.

Respond in this EXACT format:
SEVERITY: <Major | Moderate | Minor | Unknown>
PLAIN_SUMMARY: <2-3 sentences explaining the interaction in simple language a patient can understand. No jargon.>
MECHANISM: <1 sentence explaining WHY this interaction happens, if known.>
PATIENT_ADVICE: <1-2 sentences on what the patient should do.>

If there is no relevant interaction data found, respond EXACTLY:

SEVERITY: Unknown
PLAIN_SUMMARY: Insufficient evidence was found in the OpenFDA drug label database to assess this drug combination. This does not mean the combination is safe or unsafe.
MECHANISM: Not available from the retrieved evidence.
PATIENT_ADVICE: Consult a doctor or pharmacist for guidance specific to this medication combination.

IMPORTANT:
- Never infer that a drug combination is safe because no interaction data was found.
- Never invent potential risks when the retrieved evidence is insufficient.
- Base the explanation only on the retrieved OpenFDA evidence."""),
    ("human", "Drug 1: {drug1}\nDrug 2: {drug2}\nOpenFDA Data Summary: {fda_data}")
])


def build_drug_chain(llm):
    """Same chain app.py used to build at module level as `_drug_chain`,
    just parameterized on the LLM client (see services/llm_clients.py)."""
    return DRUG_INTERACTION_PROMPT | llm | StrOutputParser()


def normalize_drug_name(drug_name: str) -> str:
    cleaned_name = drug_name.strip().lower()
    return DRUG_NAME_ALIASES.get(cleaned_name, cleaned_name)


def query_openfda(drug1: str, drug2: str) -> dict:
    drug1 = normalize_drug_name(drug1)
    drug2 = normalize_drug_name(drug2)

    base = "https://api.fda.gov/drug/label.json"

    # Terms allowed for evidence matching.
    # Keep these conservative to reduce false positives.
    EVIDENCE_TERMS = {
        "warfarin": [
            "warfarin",
            "coumarin anticoagulant",
            "coumarin anticoagulants",
        ],
        "aspirin": [
            "aspirin",
            "acetylsalicylic acid",
        ],
        "ibuprofen": [
            "ibuprofen",
        ],
        "acetaminophen": [
            "acetaminophen",
            "paracetamol",
        ],
        "sertraline": [
            "sertraline",
        ],
        "tramadol": [
            "tramadol",
        ],
    }

    def get_evidence_terms(drug_name: str) -> list:
        return EVIDENCE_TERMS.get(drug_name, [drug_name])

    def extract_evidence(results: list, target_drug: str) -> list:
        evidence_found = []
        target_terms = get_evidence_terms(target_drug)

        for result in results:
            for section in result.get("drug_interactions", []):
                sentences = section.replace("\n", " ").split(".")

                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()

                    if any(
                        term.lower() in sentence_lower
                        for term in target_terms
                    ):
                        start = max(0, i - 1)
                        end = min(len(sentences), i + 2)

                        evidence = ". ".join(
                            sentences[start:end]
                        ).strip()

                        if evidence:
                            evidence_found.append(evidence)

        return evidence_found

    def fetch_labels(drug_name: str) -> list:
        resp = requests.get(
            base,
            params={
                "search": (
                    f'(openfda.generic_name:"{drug_name}" OR '
                    f'openfda.brand_name:"{drug_name}" OR '
                    f'openfda.substance_name:"{drug_name}")'
                ),
                "limit": 100,
            },
            timeout=10,
        )

        if resp.status_code != 200:
            return []

        return resp.json().get("results", [])

    try:
        # Fetch FDA labels separately
        drug1_results = fetch_labels(drug1)
        drug2_results = fetch_labels(drug2)

        labels_checked = len(drug1_results) + len(drug2_results)

        relevant_interactions = []

        # In Drug 1 labels, search for evidence terms describing Drug 2
        relevant_interactions.extend(
            extract_evidence(drug1_results, drug2)
        )

        # In Drug 2 labels, search for evidence terms describing Drug 1
        relevant_interactions.extend(
            extract_evidence(drug2_results, drug1)
        )

        # Remove duplicate evidence
        relevant_interactions = list(
            dict.fromkeys(relevant_interactions)
        )

        return {
            "found": len(relevant_interactions) > 0,
            "count": len(relevant_interactions),
            "interactions": relevant_interactions[:5],
            "raw": {
                "drug1": drug1,
                "drug2": drug2,
                "labels_checked": labels_checked,
            },
        }

    except Exception as e:
        return {
            "found": False,
            "count": 0,
            "interactions": [],
            "raw": {},
            "error": str(e),
        }


def parse_drug_field(text: str, field: str) -> str:
    for line in text.split("\n"):
        if line.strip().upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""
