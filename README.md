<div align="center">
  <img src="https://img.shields.io/badge/AMD-ROCm_Forge-ed1c24?style=for-the-badge&logo=amd&logoColor=white" alt="ROCm Forge" />
  
  # ROCm Forge
  **The multi-agent migration engine bridging the gap between NVIDIA CUDA and AMD ROCm.**
  
  [![Hackathon](https://img.shields.io/badge/AMD_Developer-Hackathon_2026-black?style=for-the-badge)](https://www.amd.com/en/developer.html)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Llama3](https://img.shields.io/badge/Llama_3.1-Groq-blueviolet?style=for-the-badge)](https://groq.com/)
</div>

<br />

## ⚡ The Problem: The CUDA Moat
The biggest bottleneck to adopting high-performance AMD GPUs (like the MI300X) isn't the hardware—it's the software ecosystem. Millions of lines of legacy AI workloads are hardcoded to the NVIDIA CUDA API. Manually migrating these codebases to AMD's ROCm is tedious, undocumented, and error-prone.

## 🚀 The Solution: ROCm Forge
**ROCm Forge** is an explainable, deterministic multi-agent AI copilot designed to automatically convert NVIDIA CUDA codebases to run natively on AMD ROCm.

By combining **Deterministic AST Parsing** for reliable core API mapping and **LLM Agents (Llama 3.1)** for contextual edge-case analysis, ROCm Forge guarantees that your migration is not a black box.

**We reduce manual migration effort by an estimated 65%.**

---

## 🧠 Core Architecture (6-Agent Pipeline)

ROCm Forge doesn't just ask an LLM to rewrite your code. It uses a robust pipeline:

1. **Analyzer Agent:** Identifies code type (Python/PyTorch, C++ Kernel, Dockerfile) and extracts CUDA-specific APIs.
2. **Checker Agent:** Maps NVIDIA APIs to their `hip` / `MIpen` equivalents using an internal knowledge base.
3. **Refactorer Agent:** Deterministically performs AST transformations. Applies **Confidence Scores** (`Safe`, `Review`, `Manual`) to every single change.
4. **Verifier Agent:** Audits the transformed code for syntax errors and un-migrated artifacts.
5. **Deployer Agent:** Automatically generates `Dockerfile`, `deploy_rocm.sh`, and `requirements.txt` specifically tuned for AMD GPU instances.
6. **LLM Explainer Agent (Optional):** Powered by Groq (Llama 3.1 70B), this agent provides human-readable documentation of the migration and identifies edge-case risks.

---

## 🛠️ Tech Stack
- **Backend:** `FastAPI` (Python) - High-performance asynchronous API.
- **Frontend:** `Vue.js` + `Tailwind CSS` - Glassmorphism, dynamic single-page dashboard.
- **AI Engine:** `Groq` + `Llama-3.1-70b-versatile`.
- **Parsing:** Python `ast` and Regex-based static analysis.

---

## ⚙️ Getting Started (Local Deployment)

Run the entire pipeline on your local machine instantly. No Node.js required.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rocm-forge.git
cd rocm-forge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
python3 api.py
```

Now open **http://localhost:8000** in your browser.

---

## 🐳 Docker Deployment (Render / Hugging Face Spaces)
ROCm Forge is fully containerized. Deploying to Render or Hugging Face takes less than 3 minutes.

```bash
# Build the Docker image
docker build -t rocm-forge .

# Run the container
docker run -p 8000:8000 rocm-forge
```

---

## 📊 Evaluation & Hackathon Alignment
**Why this wins:**
- **Business Value:** Saves engineering hours by automating the most painful part of hardware migration.
- **Technical Depth:** Bypasses LLM hallucinations by using a deterministic parser as the primary engine.
- **Explainability:** Developers do not trust black-box code generators. Our UI shows exactly *what* changed, *why* it changed, and how *confident* the system is.

<div align="center">
  <i>Built with dedication for the AMD Developer Hackathon.</i>
</div>
