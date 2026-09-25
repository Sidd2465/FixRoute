# AI Usage Disclosure Form
### Samsung PRISM GenAI Hackathon (3rd Edition 2026–27) — Theme 2

---

## 1. Team Details

| Field | Detail |
|---|---|
| **Team Name** | **Primauros** |
| **Project / Product Name** | **FixRoute — Smart Guided Troubleshooting Engine** |
| **Organization / Institution** | **Thapar Institute of Engineering & Technology (Thapar University)** |
| **Submission Date** | **24 September 2026** |
| **Theme ID** | **Theme 2 (Smart Guided Troubleshooting Engine)** |
| **Submission GitHub Link** | [https://github.com/Sidd2465/FixRoute](https://github.com/Sidd2465/FixRoute) |

---

## 2. AI Usage Declaration

* **Did your team use any Artificial Intelligence (AI) in developing this project?**  
  **`[ X ] YES`** &nbsp;&nbsp;&nbsp;&nbsp; `[   ] NO`

> **Summary:** AI was utilized in two distinct roles:  
> 1. **At Runtime (Inference Engine):** Google Gemini 3.6 Flash (`gemini-3.6-flash`) for structured, schema-constrained step extraction and planning parameter synthesis.  
> 2. **During Development (Engineering Assistance):** GitHub Copilot for code scaffolding, typing verification, and test harness construction.  
> 
> *All core algorithmic architectures—including the dual-field TF-IDF reference retriever, calibrated subsection gating, conservative deep link margin gating, and the sub-millisecond semantic cache—were designed, mathematically audited, and verified by the team.*

---

## 3. Purpose of AI Usage (Brief Details)

| Category | Role and Specific Usage in FixRoute |
|---|---|
| **Idea generation / brainstorming** | Formulating multi-stage pipeline stages and evaluating One UI settings routing edge cases. |
| **Code generation or assistance** | Scaffolding Pydantic v2 schemas, FastAPI test client fixtures, and benchmark logging scripts. |
| **UI / UX design** | Structuring clean JSON response envelopes adhering to the official `ContextDeeplinkResponse` schema. |
| **Content creation** | Synthesizing realistic held-out user query paraphrases to benchmark semantic cache hit rates. |
| **Data analysis** | Auditing TF-IDF cosine similarity distributions across 20 SIIS articles and 578 Settings deep links. |
| **Testing / debugging** | Constructing 15 pytest suites (169 unit & integration tests) and resolving cold-start latency percentiles. |
| **Other** | None |

---

## 4. Feature Origin Classification

### Feature 1: Dual-Field Reference & Subsection Routing
* **Classification:** **Both (Team Architecture & AI Assisted)**
* **AI Tools:** GitHub Copilot / Gemini.
* **Prompt & Architecture:** Team designed 0.8 original query + 0.2 title/content TF-IDF dual-field retriever with score ($\ge 0.50$) and margin ($\ge 0.10$) gates.
* **Output & Modifications:** Achieved 20/20 (100%) self-routing accuracy across official SIIS articles; added calibrated subsection gating (`min_score=0.25`, `min_margin=0.05`).

---

### Feature 2: Grounded Step & Prerequisite Extraction
* **Classification:** **Both (Team-Engineered Schema & AI Runtime)**
* **AI Tools:** Google Gemini 3.6 Flash (`gemini-3.6-flash`).
* **Prompt & Architecture:** Structured JSON extraction adhering to strict Pydantic `ExtractionResult` schema, categorizing steps, warnings, conditions, and missing details.
* **Output & Modifications:** Implemented custom local quote-validator that enforces exact substring matching against source text to mathematically guarantee zero hallucination.

---

### Feature 3: Gated Settings Deep Link Mapping
* **Classification:** **Both (Team-Designed Safety Rules & AI Integration)**
* **AI Tools:** scikit-learn TF-IDF / Copilot.
* **Prompt & Architecture:** Team implemented confidence gate (score $\ge 0.50$) and margin gate (margin $\ge 0.10$) over 578 One UI Settings deep links.
* **Output & Modifications:** Accurately maps clear intents to exact catalogue entries (e.g., `DL-0169` Navigation Bar), auto-wiring Bixby action/validation URIs while safely falling back to manual on ambiguous complaints.

---

### Feature 4: Sub-Millisecond Semantic Cache
* **Classification:** **Self-Generated / Both (Team Architecture & AI Optimization)**
* **AI Tools:** scikit-learn TF-IDF / Python in-memory store.
* **Prompt & Architecture:** Team architected dual-field word unigram + character n-gram similarity engine with disk persistence (`data/cache.json`).
* **Output & Modifications:** Pre-warmed with all 20 official Samsung SIIS topics. Achieved $2.60\text{ms}$ P50 latency and $100\%$ held-out paraphrase hit rate at $\$0$ runtime cost.

---

### Feature 5: Production FastAPI Microservice & Rule Validator
* **Classification:** **Both (Team Architecture & AI Test Scaffolding)**
* **AI Tools:** FastAPI, Uvicorn, Pydantic v2, Docker.
* **Prompt & Architecture:** Implemented `GET /health` and `POST /v1/troubleshoot` adhering to official `ContextDeeplinkResponse` envelope; authoritative 13-rule validator.
* **Output & Modifications:** 169 automated tests green, 100% zero-fatal rule compliance, fully containerized via Dockerfile.

---

## 5. Ethical & Compliance Confirmation

* **AI usage complies with hackathon guidelines and institutional policies:** &nbsp; **`[ X ] YES (Confirmed)`**
* **No proprietary, confidential, or copyrighted data was misused:** &nbsp; **`[ X ] I AGREE (Confirmed)`**
* *All reference text and deep links utilized remain authentic to the provided Samsung PRISM student kit assets. Private API credentials are strictly secured via environment variables and excluded from public version control.*

---

## 6. Declaration & Sign-Off

| Field | Detail | Field | Detail |
|---|---|---|---|
| **Name of Team Representative:** | Siddharth Singhal | **Role:** | Team Lead / Full-Stack Developer |
| **Secondary Team Member:** | Ridhi Sood | **Role:** | Developer / Machine Learning |
| **Signature:** | *Siddharth Singhal* (Electronically Signed) | **Date:** | 24 September 2026 |
