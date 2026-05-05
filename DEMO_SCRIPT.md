# ROCm Forge — 3.5 Minute Presentation Script

> **Total time: 3 min 30 sec**  
> Structure: Problem (30s) → Solution (30s) → Live Demo (1.5 min) → Architecture (30s) → Close (30s)

---

## 🎬 SLIDE 1: The Problem (0:00 – 0:30)

**[Show a stat slide or just the empty ROCm Forge UI]**

> *"The biggest bottleneck to AMD GPU adoption isn't hardware — it's software.*
>
> *There are billions of lines of CUDA code in production. Every AI company, every research lab, every cloud provider has CUDA baked into their stack. Migrating this to AMD's ROCm is manual, tedious, and error-prone.*
>
> *AMD's own hipify tool handles basic string replacements. But what about hardcoded warp sizes? Tensor Core intrinsics? Inline PTX assembly? That's where everything breaks silently.*
>
> *We built ROCm Forge to solve this."*

---

## 🎬 SLIDE 2: The Solution — One Line (0:30 – 1:00)

**[Switch to ROCm Forge UI — show the header: "9-Agent Migration Engine"]**

> *"ROCm Forge is a 9-agent AI pipeline that doesn't just translate syntax — it understands GPU hardware.*
>
> *Three things make it different from any other tool:*
>
> *First — a Hardware-Aware Scanner that catches warp-vs-wavefront bugs that SILENTLY produce wrong results on AMD.*
>
> *Second — a Verification Pass with rescue branches. If the first pass leaves any CUDA residue, the system self-heals and runs again.*
>
> *Third — a Build Error Copilot that tells you exactly what will break BEFORE you compile.*
>
> *Let me show you."*

---

## 🎬 LIVE DEMO (1:00 – 2:30) ⭐ THE MONEY SHOT

### Demo Move 1: Start Easy (1:00 – 1:20)

**[Select "PyTorch ResNet Training" from the dropdown → Click MIGRATE]**

> *"Let's start with a standard PyTorch training script."*

**[Point to the metrics that appear]**

> *"Watch the 9 agents fire. Migration Score: 72. AMD Readiness: 95%. The Risk Heatmap shows almost everything is Low Risk — green. This is an easy migration."*

**[Click "Agent Trace" tab]**

> *"In the Agent Trace tab, you can see exactly what each agent did — the Analyzer found 12 CUDA patterns, the Refactorer applied 8 safe transforms, and the Verification Pass confirmed zero leftover CUDA artifacts."*

### Demo Move 2: The HARD One (1:20 – 2:30)

**[Select "Tensor Core WMMA Kernel" from the dropdown → Click MIGRATE]**

> *"NOW let's do the hardest possible migration. This is an NVIDIA Tensor Core GEMM kernel using WMMA intrinsics. Standard hipify COMPLETELY fails on this."*

**[Point to the banner]**

> *"Look. Migration Score drops to ZERO. AMD Readiness: 0%. The Risk Heatmap is fully red."*

**[Click "Agent Trace" tab]**

> *"The Hardware-Aware Scanner caught wmma::mma_sync and flagged it for AMD Matrix Core MFMA intrinsic lowering.*
>
> *The Exploration Scanner found hardcoded warp size 32 — every instance will produce WRONG results on AMD's 64-wide wavefronts.*
>
> *The Build Error Copilot is already saying: add rocwmma.hpp, replace __syncwarp, link with -lrocblas."*

**[Click "Validation Report" tab]**

> *"And the Validation Report gives a complete checklist — what's safe, what needs manual review, and why this matters to AMD."*

> *"No other migration tool does this. We catch it because we understand the HARDWARE, not just the syntax."*

---

## 🎬 SLIDE 3: Architecture (2:30 – 3:00)

**[Show the Mermaid architecture diagram from README or a slide]**

> *"Under the hood — 9 specialized agents. The key differentiators are the Hardware-Aware Scanner, the Verification Pass with rescue branches, and the Health Monitor that generates per-line saliency maps.*
>
> *The system is backed by a custom LoRA fine-tuned CodeLlama model trained on AMD Instinct MI300X GPUs using ROCm 6.2."*

---

## 🎬 SLIDE 4: Close (3:00 – 3:30)

> *"ROCm Forge reduces CUDA-to-ROCm migration effort by an estimated 65%. It handles Python scripts, Dockerfiles, and the hardest Tensor Core kernels.*
>
> *We're not translating syntax. We're doing architecture-level, hardware-aware code transformation.*
>
> *ROCm Forge helps developers move from NVIDIA to AMD with confidence, speed, and transparency.*
>
> *Thank you. We're Team Cipher."*

**[End on the ROCm Forge UI with the WMMA results visible]**

---

## 📋 Recording Tips

| # | Tip |
|---|-----|
| 1 | **Don't read the script** — memorize the flow, speak naturally |
| 2 | **The WMMA demo is your KILLER MOMENT** — spend the most time here |
| 3 | **Show the Agent Trace tab animating** — it looks incredible |
| 4 | **Use Loom or OBS** — screen capture + voiceover |
| 5 | **Practice twice** — aim for 3:15 on second run (buffer for pauses) |
| 6 | **Keep webcam off** — focus entirely on the screen |

## 🎯 What Judges Will Remember

| Moment | Why It Sticks |
|--------|--------------|
| "Migration Score: 0" on WMMA | Shows your tool handles the HARDEST problems |
| "AMD Readiness: 0%" with red bar | Visual punch — judges SEE the danger |
| "14 critical lines" in saliency | Proves you understand GPU architecture |
| Agent Trace with 9 green steps | Shows this is a REAL multi-agent system |
| "Trained on MI300X" | Shows you actually used AMD hardware |
| Build Error Copilot suggestions | Pre-emptive — judges will be surprised |
