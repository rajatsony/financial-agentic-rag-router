# Financial Research & Analytics Agent

An autonomous, agentic RAG (Retrieval-Augmented Generation) system designed to route complex financial queries, retrieve semantic document context (e.g., SEC 10-K filings), and fetch real-time market data with deterministic failsafes.

## 🏗️ System Architecture

![Architecture Diagram](architecture.png)

### 1. Offline Data Pipeline (Ingestion)
* **Document Processing:** Ingests enterprise financial reports from the `/data` directory.
* **Chunking:** Semantic splitting with 500-token chunks and 50-token overlap to preserve mathematical and financial context.
* **Embedding:** Utilizes a dense Bi-Encoder (`BGE-small-en-v1.5`) for high-fidelity vector generation.
* **Storage:** Indexed in FAISS for low-latency similarity search.

### 2. Online Inference Pipeline (Agentic Routing)
* **The Brain:** Llama-3 (via Groq API) acts as the central reasoning engine, enforced by strict system prompts to output deterministic JSON tool calls.
* **Parallel Tool Paths:**
  * **Semantic Search:** FAISS retrieval followed by a Cross-Encoder for precise re-ranking (Top-3 injection).
  * **Live Market Data:** Integration with the `yfinance` API for real-time stock price fetching and analysis.

### 3. Resiliency & Telemetry
* **Deterministic Failsafes:** Engineered `try/except` loops around external API calls. If the market data API times out, the system routes the error back to the LLM for a graceful fallback response rather than crashing.
* **Evaluation Framework:** Asynchronous telemetry pipelines designed for automated performance evaluation using the RAGAS framework to monitor hallucination rates and context precision.

## 📁 Repository Structure

```text
financial-agentic-rag-router/
├── data/
│   └── AAPL_2025_10K_Report.pdf     # Enterprise financial corpus
├── rag_pipeline.py                  # Core agentic routing and execution logic
├── .env.example                     # Environment variable template
├── requirements.txt                 # Infrastructure dependencies
└── README.md                        # System documentation


## 🚀 Quick Start

### Prerequisites
* Python 3.9+
* Groq API Key

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajatsony/financial-agentic-rag-router.git
   cd financial-agentic-rag-router
```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   * Rename `.env.example` to `.env`.
   * Add your API key inside `.env`: `GROQ_API_KEY=your_key_here`

4. **Run the Agent:**
   ```bash
   python rag_pipeline.py
   ```
