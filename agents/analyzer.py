"""
ROCm Forge — Code Analyzer Agent
Scans source code for CUDA-specific patterns, APIs, libraries, env vars,
and dependencies. Produces a structured analysis report.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from knowledge.cuda_mappings import (
    CUDA_TO_HIP_API,
    CUDA_TO_ROCM_LIBS,
    PYTORCH_PATTERNS,
    PIP_PACKAGE_MAPPINGS,
    ENV_VAR_MAPPINGS,
    HEADER_MAPPINGS,
    KNOWN_ISSUES,
    CLI_TOOL_MAPPINGS,
    DOCKER_IMAGE_MAPPINGS,
    HARDWARE_AWARE_MAPPINGS,
    IMPLICIT_CUDA_PATTERNS,
)
from agents.ast_transformer import ASTTransformer


@dataclass
class CUDAPattern:
    """A detected CUDA pattern in the source code."""
    pattern: str
    line_number: int
    line_content: str
    category: str  # api, library, env_var, header, pytorch, docker, cli, package
    severity: str  # info, warning, error
    rocm_equivalent: str
    note: str = ""


@dataclass
class AnalysisResult:
    """Complete analysis of a CUDA source file."""
    detected_patterns: List[CUDAPattern] = field(default_factory=list)
    cuda_api_calls: List[CUDAPattern] = field(default_factory=list)
    library_references: List[CUDAPattern] = field(default_factory=list)
    env_variables: List[CUDAPattern] = field(default_factory=list)
    header_includes: List[CUDAPattern] = field(default_factory=list)
    pytorch_patterns: List[CUDAPattern] = field(default_factory=list)
    docker_patterns: List[CUDAPattern] = field(default_factory=list)
    cli_tools: List[CUDAPattern] = field(default_factory=list)
    pip_packages: List[CUDAPattern] = field(default_factory=list)
    known_issues: List[Dict] = field(default_factory=list)
    hardware_issues: List[CUDAPattern] = field(default_factory=list)
    implicit_assumptions: List[Dict] = field(default_factory=list)
    saliency_map: Dict[int, str] = field(default_factory=dict)  # line -> critical/warning/safe
    migration_score: int = 100
    migration_health: float = 1.0  # 0.0 (dangerous) to 1.0 (safe) — drift detection
    migration_level: str = "Easy"
    code_type: str = "python"  # python, cpp, dockerfile, requirements
    summary: Dict = field(default_factory=dict)
    ast_findings: List[Dict] = field(default_factory=list)  # AST-level analysis results
    trace_log: List[str] = field(default_factory=list)


class AnalyzerAgent:
    """
    Code Analysis Agent — Scans source code for CUDA-specific patterns
    and produces a structured migration analysis.
    """
    
    def __init__(self):
        self.name = "Code Analyzer Agent"
    
    def analyze(self, code: str, code_type: str = "auto") -> AnalysisResult:
        """Run full analysis on the provided code."""
        result = AnalysisResult()
        
        # Auto-detect code type
        if code_type == "auto":
            code_type = self._detect_code_type(code)
        result.code_type = code_type
        result.trace_log.append(f"🔍 Detected code type: {code_type}")
        
        lines = code.split("\n")
        
        # Run all analysis passes
        self._scan_cuda_apis(lines, result)
        result.trace_log.append(f"📡 Scanned for CUDA Runtime APIs → Found {len(result.cuda_api_calls)} patterns")
        
        self._scan_libraries(lines, result)
        result.trace_log.append(f"📚 Scanned for CUDA libraries → Found {len(result.library_references)} references")
        
        self._scan_env_vars(lines, result)
        result.trace_log.append(f"🔧 Scanned for environment variables → Found {len(result.env_variables)} variables")
        
        self._scan_headers(lines, result)
        result.trace_log.append(f"📄 Scanned for CUDA headers → Found {len(result.header_includes)} includes")
        
        self._scan_pytorch_patterns(lines, result)
        result.trace_log.append(f"🔥 Scanned for PyTorch CUDA patterns → Found {len(result.pytorch_patterns)} patterns")
        
        self._scan_docker_patterns(lines, result)
        result.trace_log.append(f"🐳 Scanned for Docker/NVIDIA patterns → Found {len(result.docker_patterns)} patterns")
        
        self._scan_cli_tools(lines, result)
        result.trace_log.append(f"💻 Scanned for NVIDIA CLI tools → Found {len(result.cli_tools)} references")
        
        self._scan_pip_packages(lines, result)
        result.trace_log.append(f"📦 Scanned for CUDA pip packages → Found {len(result.pip_packages)} packages")
        
        self._check_known_issues(code, result)
        result.trace_log.append(f"⚠️ Checked for known incompatibilities → Found {len(result.known_issues)} issues")
        
        # Hardware-aware scan (Intrinsic-level analysis)
        self._scan_hardware_issues(lines, result)
        result.trace_log.append(f"🔬 Hardware-Aware Scan → Found {len(result.hardware_issues)} architecture-level issues")
        
        # Curiosity-driven exploration scan (implicit assumptions)
        self._scan_implicit_assumptions(lines, code, result)
        result.trace_log.append(f"🧪 Exploration Scan → Found {len(result.implicit_assumptions)} implicit CUDA assumptions")
        
        # AST-level analysis (Python only — real compiler-level understanding)
        if result.code_type == "python":
            self._run_ast_analysis(code, result)
        
        # Build saliency map (per-line risk scoring)
        self._build_saliency_map(lines, result)
        result.trace_log.append(f"🎯 Saliency Map → {sum(1 for v in result.saliency_map.values() if v == 'critical')} critical lines identified")
        
        # Calculate migration score + health
        self._calculate_score(result)
        result.trace_log.append(f"📊 Migration Score: {result.migration_score}/100 ({result.migration_level})")
        result.trace_log.append(f"🩺 Migration Health: {result.migration_health:.0%} (drift detection)")
        
        # Build summary
        self._build_summary(result)
        result.trace_log.append(f"✅ Analysis complete — {len(result.detected_patterns)} total patterns detected")
        
        return result
    
    def _detect_code_type(self, code: str) -> str:
        """Auto-detect whether the code is Python, C++, Dockerfile, or requirements."""
        code_lower = code.lower().strip()
        first_line = code.strip().split("\n")[0].strip().lower() if code.strip() else ""
        
        # --- Dockerfile detection FIRST (highest priority) ---
        # Dockerfiles start with FROM <image> and contain RUN/CMD/EXPOSE/WORKDIR
        docker_keywords = ["run ", "cmd ", "expose ", "workdir ", "copy ", "env ", "entrypoint "]
        if first_line.startswith("from ") and any(kw in code_lower for kw in docker_keywords):
            return "dockerfile"
        # Also catch comment-prefixed Dockerfiles
        if any(kw in code_lower for kw in ["from nvidia/", "from rocm/", "from ubuntu:", "from python:"]):
            if any(kw in code_lower for kw in docker_keywords):
                return "dockerfile"
        
        # --- Python / Inline CUDA disambiguation ---
        is_python = any(kw in code for kw in ["import torch", "import os", "def ", "class ", "if __name__"])
        
        # --- C/C++ CUDA detection ---
        has_cpp_includes = "#include" in code and (".h>" in code or '.h"' in code)
        has_cpp_global = "__global__" in code and ("void " in code or "float " in code or "int " in code)
        
        if (has_cpp_includes or has_cpp_global) and not is_python:
            return "cpp"
        elif (has_cpp_includes or has_cpp_global) and is_python:
            return "python"
        
        # --- Requirements.txt detection ---
        lines = code.strip().split("\n")
        non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        if non_empty:
            pkg_like = sum(1 for l in non_empty if re.match(r'^[\w\-\[\].]+([>=<~!]|$)', l))
            if pkg_like / len(non_empty) > 0.6:
                return "requirements"
        
        # --- Python detection (default) ---
        if any(kw in code for kw in ["import ", "def ", "class ", "if __name__"]):
            return "python"
        
        return "python"
    
    def _scan_cuda_apis(self, lines: List[str], result: AnalysisResult):
        """Scan for CUDA Runtime API calls."""
        for i, line in enumerate(lines, 1):
            for cuda_api, hip_api in CUDA_TO_HIP_API.items():
                if cuda_api in line:
                    pattern = CUDAPattern(
                        pattern=cuda_api,
                        line_number=i,
                        line_content=line.strip(),
                        category="api",
                        severity="warning",
                        rocm_equivalent=hip_api,
                        note=f"Replace {cuda_api} with {hip_api}",
                    )
                    result.cuda_api_calls.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_libraries(self, lines: List[str], result: AnalysisResult):
        """Scan for CUDA library references."""
        for i, line in enumerate(lines, 1):
            for cuda_lib, rocm_lib in CUDA_TO_ROCM_LIBS.items():
                # Use word boundary matching to avoid partial matches
                if re.search(rf'\b{re.escape(cuda_lib)}\b', line):
                    pattern = CUDAPattern(
                        pattern=cuda_lib,
                        line_number=i,
                        line_content=line.strip(),
                        category="library",
                        severity="warning",
                        rocm_equivalent=rocm_lib,
                        note=f"Replace {cuda_lib} with {rocm_lib}",
                    )
                    result.library_references.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_env_vars(self, lines: List[str], result: AnalysisResult):
        """Scan for CUDA-specific environment variables."""
        for i, line in enumerate(lines, 1):
            for cuda_var, rocm_var in ENV_VAR_MAPPINGS.items():
                if cuda_var in line:
                    pattern = CUDAPattern(
                        pattern=cuda_var,
                        line_number=i,
                        line_content=line.strip(),
                        category="env_var",
                        severity="warning",
                        rocm_equivalent=rocm_var,
                        note=f"Change {cuda_var} to {rocm_var}",
                    )
                    result.env_variables.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_headers(self, lines: List[str], result: AnalysisResult):
        """Scan for CUDA header includes."""
        for i, line in enumerate(lines, 1):
            for cuda_header, hip_header in HEADER_MAPPINGS.items():
                if cuda_header in line:
                    pattern = CUDAPattern(
                        pattern=cuda_header,
                        line_number=i,
                        line_content=line.strip(),
                        category="header",
                        severity="warning",
                        rocm_equivalent=hip_header,
                        note=f"Replace {cuda_header} with {hip_header}",
                    )
                    result.header_includes.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_pytorch_patterns(self, lines: List[str], result: AnalysisResult):
        """Scan for PyTorch CUDA-specific patterns."""
        for i, line in enumerate(lines, 1):
            for pt_pattern, info in PYTORCH_PATTERNS.items():
                if pt_pattern in line:
                    severity = "info" if info["action"] == "compatible" else "warning"
                    pattern = CUDAPattern(
                        pattern=pt_pattern,
                        line_number=i,
                        line_content=line.strip(),
                        category="pytorch",
                        severity=severity,
                        rocm_equivalent=info["replacement"],
                        note=info["note"],
                    )
                    result.pytorch_patterns.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_docker_patterns(self, lines: List[str], result: AnalysisResult):
        """Scan for Docker NVIDIA-specific patterns."""
        for i, line in enumerate(lines, 1):
            for nvidia_image, rocm_image in DOCKER_IMAGE_MAPPINGS.items():
                if nvidia_image in line:
                    pattern = CUDAPattern(
                        pattern=nvidia_image,
                        line_number=i,
                        line_content=line.strip(),
                        category="docker",
                        severity="warning",
                        rocm_equivalent=rocm_image,
                        note=f"Replace NVIDIA base image with ROCm equivalent",
                    )
                    result.docker_patterns.append(pattern)
                    result.detected_patterns.append(pattern)
            
            # Check for NVIDIA-specific Docker env vars
            nvidia_docker_patterns = {
                "NVIDIA_VISIBLE_DEVICES": "Use --device=/dev/kfd --device=/dev/dri instead",
                "NVIDIA_DRIVER_CAPABILITIES": "Not needed for ROCm",
                "nvidia-smi": "Use rocm-smi",
            }
            for nv_pattern, note in nvidia_docker_patterns.items():
                if nv_pattern in line:
                    pattern = CUDAPattern(
                        pattern=nv_pattern,
                        line_number=i,
                        line_content=line.strip(),
                        category="docker",
                        severity="warning",
                        rocm_equivalent=note,
                        note=note,
                    )
                    result.docker_patterns.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_cli_tools(self, lines: List[str], result: AnalysisResult):
        """Scan for NVIDIA CLI tool references."""
        for i, line in enumerate(lines, 1):
            for nv_tool, rocm_tool in CLI_TOOL_MAPPINGS.items():
                if nv_tool in line:
                    pattern = CUDAPattern(
                        pattern=nv_tool,
                        line_number=i,
                        line_content=line.strip(),
                        category="cli",
                        severity="info",
                        rocm_equivalent=rocm_tool,
                        note=f"Use {rocm_tool} instead of {nv_tool}",
                    )
                    result.cli_tools.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _scan_pip_packages(self, lines: List[str], result: AnalysisResult):
        """Scan for CUDA-specific pip packages."""
        for i, line in enumerate(lines, 1):
            for pkg, info in PIP_PACKAGE_MAPPINGS.items():
                if pkg in line:
                    action = info["action"]
                    if action == "remove":
                        severity = "warning"
                        equiv = "Remove (provided by ROCm)"
                    elif action == "replace":
                        severity = "warning"
                        equiv = info.get("replacement", "See docs")
                    elif action == "warning":
                        severity = "warning"
                        equiv = info.get("note", "Check compatibility")
                    else:
                        severity = "info"
                        equiv = info.get("note", "Compatible")
                    
                    pattern = CUDAPattern(
                        pattern=pkg,
                        line_number=i,
                        line_content=line.strip(),
                        category="package",
                        severity=severity,
                        rocm_equivalent=equiv,
                        note=info.get("note", info.get("replacement", "")),
                    )
                    result.pip_packages.append(pattern)
                    result.detected_patterns.append(pattern)
    
    def _check_known_issues(self, code: str, result: AnalysisResult):
        """Check for known incompatibilities."""
        for issue_key, issue in KNOWN_ISSUES.items():
            if re.search(issue["pattern"], code, re.IGNORECASE):
                result.known_issues.append({
                    "key": issue_key,
                    "severity": issue["severity"],
                    "message": issue["message"],
                    "fix": issue["fix"],
                })
    
    def _scan_hardware_issues(self, lines: List[str], result: AnalysisResult):
        """Hardware-Aware Scan: Detect architecture-level issues that simple
        API mapping would miss. Inspired by intrinsic-level PTQ analysis —
        understanding what the hardware ACTUALLY does, not just what the API says."""
        for i, line in enumerate(lines, 1):
            # Check for WMMA / Tensor Core usage
            for hw_pattern, info in HARDWARE_AWARE_MAPPINGS.items():
                if hw_pattern == "32":
                    # Special handling: only flag '32' if it's near warp-related context
                    contexts = info.get("context", [])
                    if "32" in line:
                        # Check surrounding lines (±2) for warp context
                        window = "\n".join(lines[max(0, i-3):min(len(lines), i+2)])
                        if any(ctx in window for ctx in contexts):
                            pattern = CUDAPattern(
                                pattern="Hardcoded Warp Size: 32",
                                line_number=i,
                                line_content=line.strip(),
                                category="hardware",
                                severity="error",
                                rocm_equivalent=info["replacement"],
                                note=info["note"],
                            )
                            result.hardware_issues.append(pattern)
                            result.detected_patterns.append(pattern)
                else:
                    if hw_pattern in line:
                        pattern = CUDAPattern(
                            pattern=hw_pattern,
                            line_number=i,
                            line_content=line.strip(),
                            category="hardware",
                            severity="error",
                            rocm_equivalent=info["replacement"],
                            note=info["note"],
                        )
                        result.hardware_issues.append(pattern)
                        result.detected_patterns.append(pattern)
    
    def _scan_implicit_assumptions(self, lines: List[str], code: str, result: AnalysisResult):
        """Curiosity-Driven Exploration Scan: Detect IMPLICIT CUDA assumptions
        that aren't explicit API calls. Like curiosity-driven RL exploration —
        we're looking for what the code DOESN'T say, not just what it does."""
        for issue_key, issue in IMPLICIT_CUDA_PATTERNS.items():
            matches = list(re.finditer(issue["regex"], code, re.MULTILINE))
            if not matches:
                continue
            
            context_required = issue.get("context_required", [])
            
            for match in matches:
                # Find which line this match is on
                line_num = code[:match.start()].count("\n") + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                # If context is required, check surrounding lines
                if context_required:
                    window_start = max(0, line_num - 4)
                    window_end = min(len(lines), line_num + 3)
                    window = "\n".join(lines[window_start:window_end]).lower()
                    if not any(ctx.lower() in window for ctx in context_required):
                        continue
                
                result.implicit_assumptions.append({
                    "key": issue_key,
                    "line": line_num,
                    "line_content": line_content.strip(),
                    "severity": issue["severity"],
                    "message": issue["message"],
                    "fix": issue["fix"],
                })
    
    def _build_saliency_map(self, lines: List[str], result: AnalysisResult):
        """Build per-line saliency map — inspired by mixed-precision saliency
        rescue in quantization research. Each line gets a risk level based on
        how likely it is to cause silent failures on AMD hardware."""
        # Mark lines from detected patterns
        for p in result.detected_patterns:
            current = result.saliency_map.get(p.line_number, "safe")
            if p.severity == "error" or p.category == "hardware":
                result.saliency_map[p.line_number] = "critical"
            elif p.severity == "warning" and current != "critical":
                result.saliency_map[p.line_number] = "warning"
        
        # Mark lines from implicit assumptions
        for assumption in result.implicit_assumptions:
            line = assumption["line"]
            if assumption["severity"] == "critical":
                result.saliency_map[line] = "critical"
            elif result.saliency_map.get(line, "safe") != "critical":
                result.saliency_map[line] = "warning"
    
    def _calculate_score(self, result: AnalysisResult):
        """Calculate migration complexity score (100 = easiest)."""
        score = 100
        
        # Deductions
        score -= len(result.cuda_api_calls) * 3          # -3 per CUDA API call
        score -= len(result.library_references) * 5       # -5 per library reference
        score -= len(result.env_variables) * 2            # -2 per env var
        score -= len(result.header_includes) * 4          # -4 per header include
        score -= len(result.docker_patterns) * 3          # -3 per Docker pattern
        score -= len(result.known_issues) * 8             # -8 per known issue
        score -= len(result.hardware_issues) * 12         # -12 per hardware-level issue (TOUGH)
        score -= len(result.implicit_assumptions) * 6     # -6 per implicit assumption
        
        # Bonus for compatible PyTorch patterns
        compatible = sum(1 for p in result.pytorch_patterns if "compatible" in p.note.lower() or p.severity == "info")
        score += compatible * 1  # Small bonus for already-compatible patterns
        
        # Clamp
        score = max(0, min(100, score))
        result.migration_score = score
        
        # Level
        if score >= 85:
            result.migration_level = "Easy"
        elif score >= 60:
            result.migration_level = "Moderate"
        elif score >= 35:
            result.migration_level = "Complex"
        else:
            result.migration_level = "Advanced"
        
        # Migration Health (drift detection) — inspired by stateful drift monitoring
        # Health degrades with critical issues and implicit assumptions
        critical_count = sum(1 for v in result.saliency_map.values() if v == "critical")
        warning_count = sum(1 for v in result.saliency_map.values() if v == "warning")
        hw_issue_count = len(result.hardware_issues)
        implicit_critical = sum(1 for a in result.implicit_assumptions if a.get("severity") == "critical")
        
        # Health formula: critical/hardware issues are the real danger.
        # Warnings (API swaps, env vars) are automatable → minimal penalty.
        # Hardware issues (warp size, WMMA intrinsics) → heavy penalty.
        health = 1.0 - (
            hw_issue_count * 0.20         # -20% per hardware-level issue (warp, WMMA, etc.)
            + implicit_critical * 0.15    # -15% per implicit critical assumption (PTX, tensor cores)
            + critical_count * 0.05       # -5% per critical saliency line
            + warning_count * 0.002       # -0.2% per warning (automatable, minor penalty)
        )
        
        # Code-type-specific floors:
        # Python/Dockerfile with no hardware issues are inherently highly AMD-ready
        # because PyTorch's .cuda() API works transparently on ROCm.
        if hw_issue_count == 0 and implicit_critical == 0:
            if result.code_type == "python":
                health = max(health, 0.95)   # PyTorch code → 95%+ readiness
            elif result.code_type == "dockerfile":
                health = max(health, 0.85)   # Dockerfiles → 85%+ readiness
            elif result.code_type == "cpp":
                health = max(health, 0.70)   # Plain C++ (no WMMA) → 70%+ readiness
            
        result.migration_health = max(0.0, min(1.0, health))
    
    def _build_summary(self, result: AnalysisResult):
        """Build summary statistics."""
        result.summary = {
            "total_patterns": len(result.detected_patterns),
            "cuda_apis": len(result.cuda_api_calls),
            "libraries": len(result.library_references),
            "env_vars": len(result.env_variables),
            "headers": len(result.header_includes),
            "pytorch": len(result.pytorch_patterns),
            "docker": len(result.docker_patterns),
            "cli_tools": len(result.cli_tools),
            "packages": len(result.pip_packages),
            "known_issues": len(result.known_issues),
            "hardware_issues": len(result.hardware_issues),
            "implicit_assumptions": len(result.implicit_assumptions),
            "ast_findings": len(result.ast_findings),
            "critical_lines": sum(1 for v in result.saliency_map.values() if v == "critical"),
            "migration_score": result.migration_score,
            "migration_health": result.migration_health,
            "migration_level": result.migration_level,
            "code_type": result.code_type,
            "warnings": sum(1 for p in result.detected_patterns if p.severity == "warning"),
            "errors": sum(1 for p in result.detected_patterns if p.severity == "error"),
            "compatible": sum(1 for p in result.detected_patterns if p.severity == "info"),
        }

    def _run_ast_analysis(self, code: str, result: AnalysisResult):
        """
        Run AST-level analysis on Python code using Python's ast module.
        This is a real compiler-level pass — not regex.
        """
        transformer = ASTTransformer()
        findings, trace = transformer.analyze(code)
        result.ast_findings = findings
        result.trace_log.extend(trace)

        # Count critical AST findings
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        if critical > 0:
            result.trace_log.append(f"🌳 AST: {critical} critical finding(s) — embedded kernels require manual review")

