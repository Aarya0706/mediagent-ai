# 🏥 MediAgent AI

### Agentic Hospital Triage & Decision Support System

AI-powered emergency assessment, department routing, doctor workflow management, and patient analytics.

**🌐 Live Demo:**  
https://mediagent-ai-pbgsa8rs7dvyyhbpyydtc7.streamlit.app

---

## 📌 Overview

MediAgent AI is an intelligent hospital triage system that helps healthcare staff quickly analyze patient symptoms, determine severity levels, recommend appropriate departments, and monitor hospital case trends through interactive dashboards.

The system combines Large Language Models (LLMs), rule-based routing, analytics dashboards, and patient history tracking to improve decision-making and patient management.

---

## ✨ Features

### 🩺 Patient Symptom Analysis
- AI-powered symptom assessment
- Severity classification (Critical / Moderate / Mild)
- Recommended hospital department routing
- Emergency alert generation

### 📋 Case History Management
- Stores analyzed patient cases
- Tracks symptom history
- Timestamped records

### 📊 Hospital Dashboard
- Cases by Department
- Cases by Severity
- Critical Case Percentage
- Real-time analytics visualization

### 👨‍⚕️ Doctor Portal
- Critical case monitoring
- Priority patient review
- Department-wise patient overview

---

## 🏗️ System Architecture

```text
Patient Symptoms
        │
        ▼
  Groq LLM Analysis
        │
        ├── Severity Detection
        ├── Department Routing
        └── Summary Generation
        │
        ▼
   SQLite Database
        │
        ▼
 Streamlit Dashboard
        │
        ├── Patient Triage
        ├── Case History
        ├── Analytics Dashboard
        └── Doctor Portal
```

## 🛠️ Tech Stack

- Python
- Streamlit
- SQLite
- LangChain
- Groq LLM
- Plotly
- Pandas

---

## 📂 Project Structure

```text
mediagent-ai/
│
├── agents/
├── database/
├── tools/
├── ui/
│   └── app.py
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Aarya0706/mediagent-ai.git
cd mediagent-ai
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run ui/app.py
```

---

## 🎯 Future Enhancements

- Appointment Scheduling
- Multi-hospital Integration
- Predictive Healthcare Analytics
- Doctor Recommendation Engine
- Patient Risk Scoring

---

## 👩‍💻 Author

**Aarya Shirsath**

Designed and developed as an AI-powered healthcare decision support project.
