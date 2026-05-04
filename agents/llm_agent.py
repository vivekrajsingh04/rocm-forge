"""
ROCm Forge — LLM Reasoning Agent
Uses Groq (free tier) for intelligent code analysis and migration advice.
Falls back gracefully if no API key is provided.
"""
import os
import json

# Try importing groq, handle missing package
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


SYSTEM_PROMPT = """You are ROCm Forge, an expert AI assistant specialized in migrating CUDA/NVIDIA code to AMD ROCm/HIP.

You have deep knowledge of:
- CUDA Runtime API → HIP API mappings
- cuBLAS → rocBLAS, cuDNN → MIOpen, cuFFT → rocFFT, NCCL → RCCL
- PyTorch CUDA → PyTorch ROCm (HIP backend)
- NVIDIA Docker → ROCm Docker configurations
- AMD Instinct MI300X GPU architecture
- ROCm 6.2 ecosystem and toolchain
- vLLM, Hugging Face, DeepSpeed on ROCm
- Performance differences: warp size 32 (NVIDIA) vs wavefront 64 (AMD)

When analyzing code, you must:
1. Identify ALL CUDA-specific patterns
2. Assess migration complexity honestly
3. Flag real compatibility risks (not theoretical ones)
4. Provide specific, actionable migration advice
5. Suggest ROCm-specific performance optimizations

Be concise, technical, and precise. Use bullet points."""


def get_llm_analysis(code: str, analysis_summary: dict, api_key: str = None) -> dict:
    """
    Get LLM-powered analysis of the code migration.
    Returns dict with: summary, risks, optimizations, advice
    Falls back to rule-based summary if no API key or if API fails.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    
    if not key or not HAS_GROQ:
        return _fallback_analysis(analysis_summary)
    
    try:
        client = Groq(api_key=key)
        
        # Truncate code if too long (Groq has context limits)
        code_snippet = code[:4000] if len(code) > 4000 else code
        
        user_prompt = f"""Analyze this CUDA/NVIDIA code for migration to AMD ROCm/HIP.

CODE:
```
{code_snippet}
```

AUTOMATED ANALYSIS FOUND:
- {analysis_summary.get('total_patterns', 0)} CUDA patterns
- {analysis_summary.get('cuda_apis', 0)} CUDA API calls
- {analysis_summary.get('libraries', 0)} CUDA library references
- {analysis_summary.get('env_vars', 0)} CUDA environment variables
- {analysis_summary.get('known_issues', 0)} known compatibility issues
- Migration Score: {analysis_summary.get('migration_score', 0)}/100

Provide your analysis in this exact JSON format:
{{
    "summary": "2-3 sentence summary of what this code does and migration outlook",
    "risks": ["risk 1", "risk 2", ...],
    "optimizations": ["optimization tip 1", "optimization tip 2", ...],
    "advice": "Key migration advice in 2-3 sentences",
    "difficulty": "Easy|Moderate|Complex|Advanced",
    "estimated_effort": "e.g. 30 minutes, 2 hours, etc."
}}

Return ONLY the JSON, no other text."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response (handle code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        result["source"] = "llm"
        return result
        
    except Exception as e:
        fallback = _fallback_analysis(analysis_summary)
        fallback["llm_error"] = str(e)
        return fallback


def get_llm_refactoring_review(original: str, refactored: str, changes: list, api_key: str = None) -> str:
    """Get LLM review of the refactored code. Returns markdown string."""
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    
    if not key or not HAS_GROQ:
        return _fallback_review(changes)
    
    try:
        client = Groq(api_key=key)
        
        changes_text = "\n".join([f"- Line {c['line']}: {c['note']}" for c in changes[:15]])
        
        prompt = f"""Review these CUDA→ROCm code migration changes and provide a brief expert assessment.

CHANGES MADE:
{changes_text}

Total changes: {len(changes)}

Write a 3-5 sentence expert review covering:
1. Whether the migrations are correct
2. Any missed opportunities
3. One specific ROCm performance tip

Be concise and technical."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_review(changes)


def _fallback_analysis(summary: dict) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    score = summary.get("migration_score", 100)
    total = summary.get("total_patterns", 0)
    apis = summary.get("cuda_apis", 0)
    issues = summary.get("known_issues", 0)
    
    if score >= 85:
        difficulty = "Easy"
        effort = "15-30 minutes"
        outlook = "straightforward"
    elif score >= 60:
        difficulty = "Moderate"
        effort = "1-2 hours"
        outlook = "manageable with some manual adjustments"
    elif score >= 35:
        difficulty = "Complex"
        effort = "3-6 hours"
        outlook = "requires careful attention to CUDA-specific patterns"
    else:
        difficulty = "Advanced"
        effort = "1-2 days"
        outlook = "significant refactoring needed for AMD compatibility"
    
    risks = []
    if apis > 0:
        risks.append(f"{apis} CUDA API calls need HIP equivalents")
    if summary.get("libraries", 0) > 0:
        risks.append("CUDA library dependencies require ROCm alternatives")
    if issues > 0:
        risks.append(f"{issues} known compatibility issues detected")
    if summary.get("env_vars", 0) > 0:
        risks.append("Environment variables need updating for ROCm")
    if not risks:
        risks.append("No significant migration risks detected")
    
    optimizations = [
        "Use PYTORCH_HIP_ALLOC_CONF=expandable_segments:True for better memory management",
        "Set MIOPEN_FIND_MODE=3 for auto-tuned convolution performance",
        "AMD GPUs use 64-wide wavefronts — batch sizes divisible by 64 perform best",
    ]
    
    return {
        "summary": f"This code contains {total} CUDA-specific patterns. Migration is {outlook}. "
                   f"Automated refactoring handles the core transformations.",
        "risks": risks,
        "optimizations": optimizations[:2],
        "advice": f"Focus on validating the {apis} API changes and test on AMD hardware. "
                  f"Most PyTorch code runs on ROCm with minimal changes.",
        "difficulty": difficulty,
        "estimated_effort": effort,
        "source": "rule-based",
    }


def _fallback_review(changes: list) -> str:
    """Rule-based fallback review."""
    n = len(changes)
    if n == 0:
        return "✅ No changes were needed — this code appears to be ROCm-compatible already."
    
    categories = set()
    for c in changes:
        note = c.get("note", "")
        if "Env var" in note:
            categories.add("environment variables")
        elif "API" in note:
            categories.add("API calls")
        elif "Header" in note:
            categories.add("header includes")
        elif "Path" in note:
            categories.add("file paths")
        elif "Message" in note:
            categories.add("error messages")
        elif "CLI" in note:
            categories.add("CLI tools")
    
    cat_str = ", ".join(categories) if categories else "various patterns"
    return (
        f"Applied {n} transformations covering {cat_str}. "
        f"All changes use verified CUDA→ROCm/HIP mappings. "
        f"Tip: After migration, run `python -c \"import torch; print(torch.cuda.is_available())\"` "
        f"on your AMD GPU to validate the setup."
    )
