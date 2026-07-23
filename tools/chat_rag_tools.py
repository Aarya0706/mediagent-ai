import os
import sqlite3
import xml.etree.ElementTree as ET

import requests

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    ROOT,
    "data",
    "hospital.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# CHUNK BUILDING
# ============================================================
#
# No new "documents" table and no ingestion pipeline - chunks are built
# on the fly from data that already exists (lab_reports/lab_values from
# the Lab Report Explainer, cases from the Symptom Checker). Each chunk
# carries a human-readable "source" label so answers can cite exactly
# where a fact came from, the same way the PRD's RAG feature is supposed
# to ("Based on your March CBC report" rather than generic knowledge).

def get_patient_chunks(patient_name, lab_limit=20, case_limit=20):
    chunks = []

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, file_name, ai_summary, created_at
            FROM lab_reports
            WHERE LOWER(patient_name) = LOWER(?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (patient_name, lab_limit),
        )
        lab_reports = [dict(row) for row in cursor.fetchall()]

        for r in lab_reports:
            source = f"Lab report from {r['created_at']} ({r['file_name']})"

            if r["ai_summary"]:
                chunks.append({
                    "text": r["ai_summary"],
                    "source": source,
                })

            cursor.execute(
                """
                SELECT parameter, value, unit, ref_low, ref_high, flag
                FROM lab_values
                WHERE report_id = ?
                ORDER BY parameter
                """,
                (r["id"],),
            )
            values = [dict(row) for row in cursor.fetchall()]
            if values:
                value_lines = []
                for v in values:
                    ref = (
                        f"{v['ref_low']}-{v['ref_high']}"
                        if v["ref_low"] is not None and v["ref_high"] is not None
                        else "unknown"
                    )
                    value_lines.append(
                        f"{v['parameter']}: {v['value']} {v['unit']} "
                        f"(flag: {v['flag']}, reference range: {ref})"
                    )
                chunks.append({
                    "text": "Extracted lab values - " + "; ".join(value_lines),
                    "source": source,
                })

        cursor.execute(
            """
            SELECT symptoms, severity, department, summary, recommendation, created_at
            FROM cases
            WHERE LOWER(patient_name) = LOWER(?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (patient_name, case_limit),
        )
        cases = [dict(row) for row in cursor.fetchall()]

        for c in cases:
            source = f"Symptom checker case from {c['created_at']}"
            text = (
                f"Symptoms: {c['symptoms']}. Severity: {c['severity']}. "
                f"Department: {c['department']}. Summary: {c['summary']}. "
                f"Recommendation: {c['recommendation']}."
            )
            chunks.append({"text": text, "source": source})

    return chunks


def has_any_chunks(patient_name):
    return bool(get_patient_chunks(patient_name, lab_limit=1, case_limit=1))


# ============================================================
# RETRIEVAL (TF-IDF, not a trained embedding model)
# ============================================================
#
# Deliberately simple: TF-IDF + cosine similarity computed fresh at query
# time over just this patient's own chunks, instead of a persistent vector
# store built with sentence-transformers. Trade-off: this matches on word
# overlap, not deep semantic meaning (e.g. "kidney" won't match "renal"
# the way a real embedding model would) - acceptable here since the query
# is being matched against the patient's OWN report vocabulary, not a
# large external corpus. No new heavy dependency (no torch, no model
# download) - just scikit-learn.

# A small, hand-picked map from common lay/clinical terms to the vocabulary
# that actually shows up in lab report parameter names. TF-IDF only matches
# shared words, so "diabetes" and "Fasting Blood Sugar" score zero overlap
# even though they're the same topic. This directly patches that gap for
# the handful of conditions patients are most likely to ask about, without
# pulling in a real medical ontology or a trained embedding model.
CLINICAL_SYNONYMS = {
    "diabetes": ["blood sugar", "glucose", "fasting", "sugar", "diabetic"],
    "diabetic": ["blood sugar", "glucose", "fasting", "sugar"],
    "sugar": ["glucose", "fasting blood sugar", "diabetes"],
    "anemia": ["hemoglobin", "haemoglobin", "rbc", "red blood cell"],
    "anaemia": ["hemoglobin", "haemoglobin", "rbc", "red blood cell"],
    "kidney": ["creatinine", "renal"],
    "renal": ["creatinine", "kidney"],
    "heart": ["cholesterol", "cardiac", "lipid"],
    "cardiac": ["cholesterol", "heart", "lipid"],
    "cholesterol": ["lipid", "hdl", "ldl"],
    "infection": ["wbc", "white blood cell", "white blood cells"],
    "immune": ["wbc", "white blood cell", "white blood cells"],
    "blood pressure": ["hypertension"],
    "hypertension": ["blood pressure"],
    "thyroid": ["tsh", "t3", "t4"],
    "bleeding": ["platelet", "platelets"],
    "clotting": ["platelet", "platelets"],
}


def _expand_query(question):
    q_lower = question.lower()
    extra_terms = []
    for term, synonyms in CLINICAL_SYNONYMS.items():
        if term in q_lower:
            extra_terms.extend(synonyms)
    if extra_terms:
        return question + " " + " ".join(extra_terms)
    return question


def retrieve_relevant_chunks(question, chunks, top_k=4, min_similarity=0.05):
    """Returns (retrieved_chunks, confident).

    confident=True means at least one personal chunk cleared min_similarity -
    the patient's own records are a trustworthy enough basis to answer from
    alone. confident=False means nothing cleared the bar; the chunks
    returned (if any) are best-effort low-relevance matches, and callers
    should treat personal grounding as insufficient on its own - this is
    the signal used to decide whether to fall back to PubMed.
    """
    if not chunks:
        return [], False

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [c["text"] for c in chunks]
    expanded_question = _expand_query(question)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts + [expanded_question])
    except ValueError:
        # Happens if the vocabulary is empty (e.g. all-stopword chunks) - we
        # can't actually score relevance here, so treat this as "not
        # confident" too (rather than silently claiming a match), which lets
        # a PubMed fallback kick in upstream if one is configured.
        return chunks[:top_k], False

    query_vec = tfidf[-1]
    chunk_vecs = tfidf[:-1]
    similarities = cosine_similarity(query_vec, chunk_vecs)[0]

    ranked = sorted(
        zip(chunks, similarities),
        key=lambda pair: pair[1],
        reverse=True,
    )

    above_threshold = [chunk for chunk, score in ranked[:top_k] if score >= min_similarity]
    if above_threshold:
        return above_threshold, True

    # Nothing cleared the similarity bar - rather than returning nothing
    # (which forces a generic "couldn't find anything" answer even when the
    # real answer might be sitting right there under different wording),
    # hand the LLM the patient's top few chunks anyway and let its own
    # "say so if there isn't enough info" instruction be the actual judge.
    # Safe to do here because each patient's own chunk set is small - this
    # isn't searching a large external corpus. Reported as not-confident so
    # a PubMed fallback still gets triggered alongside this.
    return [chunk for chunk, _ in ranked[:min(3, len(ranked))]], False


# ============================================================
# PUBMED FALLBACK (NCBI Entrez E-utilities - free, no API key)
# ============================================================
#
# Two-tier RAG per the PRD: retrieve from the patient's own records first;
# only when that retrieval isn't confident (nothing cleared the TF-IDF
# similarity bar) do we also pull general medical literature from PubMed.
# This keeps PubMed as background/context, never as a replacement for what
# the patient's own data does or doesn't show.

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_TOOL_NAME = "MediAgentAI"
# NCBI asks (not requires) a contact email for programmatic use so they can
# reach out if a tool is misbehaving. Optional - set PUBMED_CONTACT_EMAIL in
# .env if you want it included.
PUBMED_CONTACT_EMAIL = os.getenv("PUBMED_CONTACT_EMAIL", "")


def _pubmed_esearch(query, max_results=3):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
        "tool": PUBMED_TOOL_NAME,
    }
    if PUBMED_CONTACT_EMAIL:
        params["email"] = PUBMED_CONTACT_EMAIL

    resp = requests.get(PUBMED_ESEARCH_URL, params=params, timeout=8)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_efetch(pmids):
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "tool": PUBMED_TOOL_NAME,
    }
    if PUBMED_CONTACT_EMAIL:
        params["email"] = PUBMED_CONTACT_EMAIL

    resp = requests.get(PUBMED_EFETCH_URL, params=params, timeout=10)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        abstract_parts = art.findall(".//AbstractText")
        year_el = art.find(".//PubDate/Year")
        journal_el = art.find(".//Journal/Title")

        pmid = pmid_el.text if pmid_el is not None else None
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"
        abstract = " ".join(
            "".join(p.itertext()).strip() for p in abstract_parts
        ).strip()
        year = year_el.text if year_el is not None else ""
        journal = journal_el.text if journal_el is not None else ""

        if pmid and abstract:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
            })

    return articles


def get_pubmed_chunks(question, max_results=3):
    """Search PubMed for abstracts relevant to `question` and return them in
    the same {"text", "source"} chunk shape used for personal-record chunks,
    so they drop straight into the same context-building code downstream.
    Returns [] on any network/parsing failure - PubMed being unreachable
    should degrade the chat feature, never crash it."""
    try:
        pmids = _pubmed_esearch(question, max_results=max_results)
        articles = _pubmed_efetch(pmids)
    except Exception:
        return []

    chunks = []
    for a in articles:
        source = f"PubMed: {a['title']} ({a['journal']}, {a['year']}) PMID:{a['pmid']}"
        chunks.append({
            "text": a["abstract"],
            "source": source,
            "is_pubmed": True,
        })
    return chunks


def get_relevant_context(question, patient_chunks, pubmed_max_results=3):
    """Personal-first, PubMed-fallback retrieval for the chat feature.

    Returns (context_chunks, used_pubmed). Only reaches out to PubMed when
    the patient's own records didn't confidently answer the question -
    PubMed is background literature, not a first resort.
    """
    personal_matches, confident = retrieve_relevant_chunks(question, patient_chunks)

    if confident:
        return personal_matches, False

    pubmed_matches = get_pubmed_chunks(question, max_results=pubmed_max_results)

    if not pubmed_matches:
        # Nothing confident in personal records and PubMed came back empty
        # or unreachable - fall back to the same best-effort personal
        # chunks as before, so the LLM still has something to reason over.
        return personal_matches, False

    return personal_matches + pubmed_matches, True


# ============================================================
# LLM ANSWER GENERATION
# ============================================================

CHAT_SYSTEM_PROMPT = """You are a health information assistant for a patient-facing app. You can be asked
two different kinds of questions, and you should handle each differently:

1. Questions ABOUT THIS PATIENT'S OWN RECORDS (e.g. "what did my last report show", "is my hemoglobin
   improving") - answer these using ONLY the retrieved excerpts below (their own lab reports/case
   history first; PubMed excerpts, if present, are background only). Never invent or guess a
   record-specific detail that isn't in the excerpts - if they don't cover it, say the records don't
   show that, rather than guessing. Cite the source naturally, e.g. "Based on your lab report from
   2026-07-21..." for personal records, or "Per PubMed (PMID:12345)..." for literature excerpts.

2. GENERAL health/medical questions that are not specific to this patient's own data (e.g. "what's a
   healthy daily sugar intake", "general tips for lowering cholesterol", "what does a CBC test for") -
   you may answer these from your own general medical knowledge, even when the excerpts below don't
   cover it or aren't relevant to the question at all. Clearly signal when you're giving general
   information rather than something drawn from this patient's own records (e.g. lead with "In
   general..." rather than "Based on your report..."), and add one brief closing reminder that general
   information isn't a substitute for personalized advice from their doctor or a dietitian.

If a question mixes both (e.g. "given my cholesterol, what should I eat?"), answer the personal part
strictly from the excerpts and the general part from your own knowledge, keeping the two clearly
separated in the answer.

Rules:
- Never state or imply a diagnosis for this patient. Frame patient-specific findings as what the
  records show, not medical conclusions.
- Never suggest starting, stopping, or changing any medication or dosage, for general questions or
  patient-specific ones.
- {mode_instruction}
"""

MODE_INSTRUCTIONS = {
    "patient": (
        "Write in simple, plain language a patient without a medical background would understand. "
        "Avoid jargon; briefly explain any clinical term you do use. Example register: "
        "'Your haemoglobin is a bit low - that's the part of your blood that carries oxygen around "
        "your body.'"
    ),
    "advanced": (
        "Write for a reader comfortable with clinical detail - assume the tone of one clinician "
        "briefing another. Actively use the standard clinical term for each finding instead of the "
        "plain-language version (e.g. say 'anaemia' not just 'low haemoglobin', 'hyperglycemia' not "
        "just 'high blood sugar', 'dyslipidemia' not just 'cholesterol levels are off') and do not "
        "stop to define these terms. Where relevant, note the general category of workup a clinician "
        "might consider next (e.g. 'iron studies', 'HbA1c') without recommending a specific action "
        "yourself. Example register: 'Haemoglobin of 10.2 g/dL is consistent with mild anaemia; "
        "iron studies or a CBC differential are the kind of next step your doctor might consider.'"
    ),
}


def generate_chat_answer(question, retrieved_chunks, mode, llm):
    """
    llm: a LangChain-compatible chat model, passed in from app.py.
    Returns {"answer": str, "sources": [str, ...]}.
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    if retrieved_chunks:
        context_lines = []
        for c in retrieved_chunks:
            context_lines.append(f"[Source: {c['source']}]\n{c['text']}")
        context_text = "\n\n".join(context_lines)
    else:
        # No personal or PubMed excerpts matched - don't hard-refuse here,
        # since the question may simply be a general one that doesn't need
        # this patient's records at all (e.g. "what's a healthy sugar
        # intake?"). The system prompt's own rules decide what to do with
        # an empty context: answer generally, or say the records don't
        # cover it, depending on which kind of question this is.
        context_text = "(No matching excerpts were retrieved from this patient's records or PubMed.)"

    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["patient"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT.format(mode_instruction=mode_instruction)),
        ("human", "Retrieved excerpts:\n\n{context}\n\nQuestion: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question}).strip()

    sources = sorted(set(c["source"] for c in retrieved_chunks))

    return {"answer": answer, "sources": sources}