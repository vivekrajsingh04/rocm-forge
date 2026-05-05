# ROCm Forge — 3.5 Minute Presentation Script (The "War Story" Version)

> **Total time: 3 min 30 sec**  
> Structure: The Nightmare (30s) → The Hero (30s) → Live Demo (1.5 min) → The Engine Room (30s) → The Drop (30s)

---

## 🎬 SLIDE 1: The Nightmare (0:00 – 0:30)

**[Show a slide of a developer crying, or just the empty ROCm Forge UI]**

> *"Let me tell you a story about a developer who tried to migrate a massive CUDA codebase to AMD's ROCm."*
>
> *"He thought, 'Oh, I'll just use a Find-and-Replace tool! Just change `cuda` to `hip` everywhere. Easy!' ... Three weeks later, his code compiled perfectly. He ran it. And the math was completely, horribly wrong."*
>
> *"Why? Because standard string replacements don't know that NVIDIA warps are 32 threads wide, but AMD wavefronts are 64. They don't know how to map Tensor Core intrinsics to AMD Matrix Cores. They don't read the hardware manuals."*
>
> *"Migrating isn't a text problem. It's a hardware problem. That’s why we built ROCm Forge."*

---

## 🎬 SLIDE 2: The Hero (0:30 – 1:00)

**[Switch to ROCm Forge UI — show the header: "9-Agent Migration Engine"]**

> *"ROCm Forge is an autonomous, multi-agent AI copilot that acts like a senior engineer who has actually read both the NVIDIA and AMD hardware specs."*
>
> *"Instead of just regex, we use a 9-Agent pipeline.*
> *- First, an **AST-level Python Transformer** that actually understands code structure, not just strings.*
> *- Second, a **Hardware-Aware Scanner** that catches those silent mathematical bugs before they ruin your weekend.*
> *- And third, a **Build Error Copilot** that predicts exactly why your code will crash on compile, and fixes it."*
>
> *"Let me show you how it works."*

---

## 🎬 LIVE DEMO (1:00 – 2:30) ⭐ THE MONEY SHOT

### Demo Move 1: The Easy Win (1:00 – 1:20)

**[Select "PyTorch ResNet Training" from the dropdown → Click MIGRATE]**

> *"Let's start with a standard PyTorch script. We click migrate."*

**[Point to the metrics that appear]**

> *"Boom. Our 9 agents fire off. Look at this beautiful dashboard—Migration Score: 72. AMD Readiness: 95%. Everything is green. The Agent Trace tab shows exactly what our AI did at the syntax tree level. It’s an easy win."*
> 
> *"But wait. Hackathons are about doing the hard stuff, right?"*

### Demo Move 2: The Final Boss (1:20 – 2:30)

**[Select "Tensor Core WMMA Kernel" from the dropdown → Click MIGRATE]**

> *"Let's feed it the final boss: A low-level C++ Tensor Core GEMM kernel using WMMA intrinsics. A normal Find-and-Replace tool would look at this and just give up."*

**[Point to the banner]**

> *"Look at the dashboard now. Migration Score hits ZERO. The Risk Heatmap is glowing bright red. ROCm Forge is basically screaming at us."*

**[Click "Agent Trace" tab]**

> *"But look at why! The Hardware-Aware Scanner caught a hardcoded warp size of 32—that right there is the bug that would have silently ruined the math on AMD's 64-wide hardware."*
>
> *"The Build Error Copilot is already telling us: 'Hey, you need to add `rocwmma.hpp` and link `-lrocblas` or the compiler will throw a tantrum.'"*

**[Click "Validation Report" tab]**

> *"The Validation Report gives us a complete checklist. It didn't just blind-translate the code; it diagnosed the hardware mismatches. We catch this because we analyze the AST and the hardware, not just the text."*

---

## 🎬 SLIDE 3: The Engine Room (2:30 – 3:00)

**[Show the Mermaid architecture diagram from README or a slide]**

> *"Under the hood, this isn't just a simple ChatGPT wrapper. It’s a 9-agent orchestration pipeline."*
>
> *"We have exploration scanners looking for implicit assumptions, verification passes that double-check the work, and we even built a ready-to-fire benchmark suite to test memory bandwidth and TFLOPS directly on AMD MI300X GPUs."*

---

## 🎬 SLIDE 4: The Drop (3:00 – 3:30)

> *"ROCm Forge reduces the friction of moving to AMD by 65%. Whether it's a Python script, a Dockerfile, or a terrifying Tensor Core kernel, it handles it."*
>
> *"We are moving migration from 'Find and Replace and Pray' to 'Hardware-Aware and Autonomous.'"*
>
> *"Thank you. We're Team Cipher."*

---

## 📋 Recording Tips

| # | Tip |
|---|-----|
| 1 | **Smile at the start** — it sets the tone for the funny intro |
| 2 | **The WMMA demo is your KILLER MOMENT** — act genuinely excited when the score hits 0 |
| 3 | **Show the Agent Trace tab animating** — it looks incredible |
| 4 | **Use Loom or OBS** — screen capture + voiceover |
| 5 | **Keep webcam off** — let them focus entirely on the screen and your voice |
