# 🏆 ROCm Forge — Final Demo Script (30-60 Seconds)

**Goal:** Show the judges that you understand the pain of migrating CUDA to AMD, and that your multi-agent system provides a safe, transparent, and powerful solution.

---

### 1. The Hook (0:00 - 0:10)
*(Screen: Start with the ROCm Forge app open, clean state)*

**"Hi, we're Team Cipher. The biggest bottleneck to AMD GPU adoption isn't hardware—it's the massive ecosystem of legacy NVIDIA CUDA code. Manually migrating this to ROCm is tedious, error-prone, and undocumented."**

**"To solve this, we built ROCm Forge: an explainable, multi-agent AI copilot that automatically performs first-pass ROCm conversion and generates deployment artifacts."**

---

### 2. The WOW Moment (0:10 - 0:30)
*(Screen: Click "🤖 LLM Inference with vLLM (CUDA)" in the sidebar. Click "🔥 Migrate to ROCm")*

**"Watch this. We paste in standard NVIDIA vLLM inference code. Our 6-agent pipeline analyzes it, detects the CUDA APIs, and instantly transforms it."**

*(Screen: Scroll down to show the "⚡ Reduced effort by 65%" banner and the metrics row)*

**"It found 18 patterns and safely applied 7 transformations. But we don't just blindly rewrite code like a basic wrapper."**

*(Screen: Point to the "Transformations Applied" section with the Safe/Review/Manual labels)*

**"We built a deterministic refactoring engine that assigns confidence labels. It safely converts environment variables and packages, but flags inline CUDA C++ kernels for manual review. This transparency is what enterprise developers actually trust."**

---

### 3. The Diff & Deployment (0:30 - 0:45)
*(Screen: Click on the "📊 Diff View" tab)*

**"Here's the side-by-side diff showing exactly what changed—CUDA on the left, ROCm on the right."**

*(Screen: Click on the "🚀 Deployment" tab)*

**"And we don't stop at Python. The agent automatically generated a complete ROCm Dockerfile, a bash deployment script, and the exact requirements.txt needed to run this on the AMD Developer Cloud today."**

---

### 4. The Closer (0:45 - 1:00)
*(Screen: Click on the "🧠 AI Insights" tab if you have a Groq API key, otherwise just show the "📋 Agent Trace" tab)*

**"Under the hood, this is powered by a robust orchestrator routing tasks between Analyzer, Refactorer, and LLM reasoning agents."**

**"ROCm Forge isn't just a demo—it's a production-ready migration pipeline that reduces conversion effort by 65%, helping developers move to AMD hardware faster and safer. Thank you."**

---

## 💡 Pro-Tips for the Demo:
- **Pacing:** Speak clearly and match your clicks to your words. Don't rush.
- **The Banner:** When you mention "reduces effort by 65%", point to the new red banner we added.
- **Transparency:** Emphasize the "Safe/Review/Manual" confidence labels. Judges from engineering backgrounds LOVE tools that don't pretend to be 100% perfect. Transparency = Trust.
