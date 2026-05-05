# ROCm Forge — 3.5 Minute Presentation Script

> **Total time: 3 min 30 sec**
> Structure: Problem (30s) → Solution (30s) → Live Demo (1.5 min) → Architecture (30s) → Close (30s)

---

## 🎬 SLIDE 1: The Problem (0:00 – 0:30)

**[Show a stat slide: "$3 Trillion+ in CUDA code exists today"]**

> *"The biggest barrier to AMD GPU adoption isn't hardware — it's software.*
>
> *There are billions of lines of CUDA code in production right now. Every AI company, every research lab, every cloud provider has CUDA baked into their stack. Migrating this code to AMD's ROCm is manual, error-prone, and takes weeks.*
>
> *AMD's own hipify tool handles the easy stuff — string replacements. But what about the HARD stuff? Hardcoded warp sizes. Tensor Core intrinsics. Inline PTX assembly. That's where everything breaks.*
>
> *We built ROCm Forge to solve this."*

---

## 🎬 SLIDE 2: The Solution — One Line (0:30 – 1:00)

**[Switch to ROCm Forge UI in browser]**

> *"ROCm Forge is a 9-agent AI pipeline that doesn't just translate CUDA syntax — it understands GPU HARDWARE.*
>
> *It has three layers that no other tool has:*
>
> *First — a Hardware-Aware Scanner that catches warp-vs-wavefront bugs that will SILENTLY fail on AMD.*
>
> *Second — a Verification Pass with rescue branches that re-scans migrated code and auto-fixes anything it missed.*
>
> *Third — a Build Error Copilot that tells you exactly what will break BEFORE you even compile.*
>
> *Let me show you."*

---

## 🎬 LIVE DEMO (1:00 – 2:30) ⭐ THE MONEY SHOT

### Demo Move 1: Start Easy (1:00 – 1:20)

**[Click Sample 1: PyTorch Training]**

> *"Let's start with a standard PyTorch training script. CUDA env vars, cuDNN settings, mixed precision."*

**[Click MIGRATE → Wait for agent trace to animate]**

> *"Watch the 9 agents fire — Analyzer, Hardware Scanner, Exploration Scanner, Refactorer, Verification Pass, Safety Check, Health Monitor, Build Copilot, and LLM. All in under a second."*

> *"Migration score: 72 out of 100. Health: 95%. This is an easy migration — mostly string replacements. Every other tool can do this."*

### Demo Move 2: The HARD One — This Is Where We Win (1:20 – 2:30)

**[Click Sample 6: Tensor Core WMMA Kernel]**

> *"NOW let's do something that will actually break. This is an NVIDIA Tensor Core GEMM kernel using WMMA intrinsics. This is the HARDEST possible migration — standard hipify will completely fail on this."*

**[Click MIGRATE]**

> *"Look at what happens:*
>
> *Migration Score drops to ZERO. Migration Health: 0%. 14 critical lines.*
>
> *The Hardware-Aware Scanner caught wmma::mma_sync and flagged it for AMD Matrix Core MFMA intrinsic lowering.*
>
> *The Exploration Scanner found 7 instances of hardcoded warp size 32 — every single one will produce WRONG results on AMD's 64-wide wavefronts.*
>
> *The Build Error Copilot is already telling you: add rocwmma.hpp, replace __syncwarp, link with -lrocblas.*
>
> *No other migration tool does this. hipify won't catch it. ChatGPT won't catch it. We catch it because we understand the HARDWARE, not just the syntax."*

---

## 🎬 SLIDE 3: Architecture (2:30 – 3:00)

**[Show the Mermaid architecture diagram from README]**

> *"Under the hood — 9 specialized agents coordinated by an orchestrator:*
>
> *The key differentiators are the Hardware-Aware Scanner, the Verification Pass with rescue branches, and the Health Monitor that tracks per-line saliency — which lines are safe, which will silently fail.*
>
> *The whole system is backed by a custom LoRA fine-tuned CodeLlama model that we're training on AMD Instinct MI300X GPUs using ROCm 6.2 — eating our own dog food."*

---

## 🎬 SLIDE 4: Close (3:00 – 3:30)

> *"ROCm Forge reduces migration effort by 65%. It handles everything from Python scripts to Dockerfiles to the hardest Tensor Core kernels.*
>
> *We're not just translating syntax — we're doing hardware-aware, architecture-level code transformation. That's what makes this different.*
>
> *Thank you. We're Team Cipher."*

**[End on the ROCm Forge UI with the WMMA results still visible]**

---

## 📋 Presentation Tips

1. **Don't read the script** — memorize the flow, speak naturally
2. **The WMMA demo is your KILLER MOMENT** — spend the most time here
3. **Show the agent trace animating** — it looks impressive in the UI
4. **Keep your webcam off** if possible — focus on the screen recording
5. **Use Loom or OBS** — record screen + voiceover
6. **Practice twice** — aim for 3:15 on the second run (buffer for pauses)

## 🎯 What Judges Will Remember

| Moment | Why It Sticks |
|--------|--------------|
| "Migration Score: 0" on WMMA | Shows your tool handles the HARD problems |
| "14 critical lines" in saliency map | Proves you understand GPU architecture |
| "Warp 32 → Wavefront 64" detection | Technical depth no other team will have |
| "Trained on MI300X" | Shows you actually used AMD hardware |
