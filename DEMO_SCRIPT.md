# 🚀 ROCm Forge: The Ultimate Pitch & Live Demo Script

*This script is designed to be conversational, confident, and highly impactful. It directly addresses the judges' technical backgrounds while keeping the narrative extremely simple and easy to follow.*

---

## 🎙️ PART 1: The Hook & The "Why Us?" (0:00 – 1:00)

**[Screen is showing the dark, sleek ROCm Forge UI]**

**You:** 
"Hi everyone, we are Team Cipher, and this is ROCm Forge. 

To migrate code from NVIDIA to AMD, you might think it's as simple as finding the word 'cuda' and replacing it with 'hip'. I guarantee you, several teams today built exactly that—a basic regex 'Find and Replace' tool. 

Here is why ROCm Forge is entirely different: **We built a compiler-level, 9-Agent AI Architecture.** 

Other tools just read text. ROCm Forge actually understands the physical hardware differences between an NVIDIA H100 and an AMD MI300X. It predicts build errors *before* you compile. It flags silent mathematical failures. We aren't just changing words; we are doing deep architectural translation.

Let me show you exactly what I mean."

---

## 💻 PART 2: The Live Demo (1:00 – 3:00)

### Click 1: PyTorch ResNet (The Standard Case)
**[Select "PyTorch ResNet Training" → Click MIGRATE]**

**You:**
"Let’s start simple. A standard PyTorch training script. We click migrate."

**[Point to the green 95% AMD Readiness Score]**
"Our 9 agents fire off instantly. Look at the dashboard: AMD Readiness is 95%. Everything is green. The AI knows that PyTorch's `.cuda()` API actually works transparently on ROCm, so it marks this as low-risk and gives us the green light to deploy. For simple scripts, it's an easy win."

---

### Click 2: Hugging Face & vLLM (The AI Ecosystem)
**[Select "Hugging Face Fine-Tuning" → Click MIGRATE]**
*(Wait 2 seconds)*
**[Select "LLM Inference with vLLM" → Click MIGRATE]**

**You:**
"What about modern generative AI? If we load a Hugging Face QLoRA script or a vLLM server... same result. The AI intelligently swaps NVIDIA wheels for ROCm wheels, handles `bitsandbytes`, and changes environment variables like `CUDA_VISIBLE_DEVICES` to `HIP_VISIBLE_DEVICES`. High readiness, ready to run."

---

### Click 3: Dockerfile (Infrastructure)
**[Select "Dockerfile (NVIDIA CUDA Base)" → Click MIGRATE]**

**You:**
"It even handles infrastructure. If I give it a production Dockerfile, the Deployer Agent automatically rips out the NVIDIA base images and injects the exact ROCm 6.2 Ubuntu images, removing obsolete NVIDIA driver flags along the way."

---

### Click 4: The Final Boss (The Differentiator)
**[Select "Tensor Core WMMA Kernel" → Click MIGRATE]**

**You:**
"But here is where we separate ourselves from everyone else. I’m going to feed it the 'Final Boss'—a low-level C++ Tensor Core kernel using NVIDIA WMMA intrinsics. A normal Find-and-Replace tool would swap the headers and call it a day... which would result in a broken, compiling nightmare."

**[Point to the dashboard turning RED and 0%]**
"Look at ROCm Forge. The Readiness Score plunges to ZERO. The Risk Heatmap is glowing red. The engine is literally screaming at us."

**[Click on the "Agent Trace" Tab]**
"And here is why: Our Hardware-Aware Scanner caught something critical. It found a hardcoded warp size of 32. If we just deployed this on an AMD GPU, which uses a 64-wide wavefront, it would silently compute the wrong math. 

Our Build Error Copilot goes even further, warning us to link the `-lrocblas` library to avoid a compiler crash. *This* is the power of a 9-Agent architecture."

---

## 🎯 PART 3: The Close (3:00 – 3:30)

**[Switch back to the main UI]**

**You:**
"ROCm Forge doesn’t just migrate code. It protects engineers from days of debugging hardware-level compiler errors. 

We’ve packaged an enterprise-grade migration consultant into a sleek, instant AI tool. Thank you, and we’d love to answer your questions."

---

## 🧠 Cheat Sheet: If Judges ask "What are the 9 Agents?"
If they ask, confidently list a few of the coolest ones to show off your backend engineering:
1. **Hardware-Aware Scanner:** Looks for implicit hardware assumptions (like Warp Size 32 vs 64).
2. **Build Error Copilot:** Cross-references code against a runbook of common ROCm compiler errors to fix them before compilation.
3. **Health Monitor:** Calculates the AMD Readiness score and drift detection.
4. **Deployer Agent:** Specifically handles Docker, dependencies, and environment setup.
5. **AST Refactorer:** Does semantic syntax-tree replacement, not just regex.
