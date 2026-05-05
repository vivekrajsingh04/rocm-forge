"""
ROCm Forge — Code Refactorer Agent
Takes analysis results and transforms CUDA code into ROCm/HIP-compatible code.
Performs intelligent, context-aware code transformations.
"""
import re
from typing import List, Tuple
from knowledge.cuda_mappings import (
    CUDA_TO_HIP_API,
    CUDA_TO_ROCM_LIBS,
    ENV_VAR_MAPPINGS,
    HEADER_MAPPINGS,
    DOCKER_IMAGE_MAPPINGS,
    CLI_TOOL_MAPPINGS,
    HARDWARE_AWARE_MAPPINGS,
)


class RefactorerAgent:
    """
    Code Refactorer Agent — Transforms CUDA code into ROCm/HIP-compatible code.
    Uses deterministic, rule-based transformations for reliable migration.
    """
    
    def __init__(self):
        self.name = "Code Refactorer Agent"
        self.changes: List[dict] = []
        self.trace_log: List[str] = []
    
    @staticmethod
    def _confidence(note: str) -> str:
        """Assign confidence level to a transformation."""
        safe = ["Env var", "Path:", "CLI tool", "URL:", "Message:", "Header:"]
        review = ["Inline CUDA", "API:", "Library:", "Attention:", "Package:", "Variable:"]
        # If the note matches a safe pattern, it's safe
        for s in safe:
            if s in note:
                return "✅ Safe"
        for r in review:
            if r in note:
                return "⚠️ Review"
        return "🔧 Manual"
    
    def refactor(self, code: str, analysis_result) -> Tuple[str, List[dict], List[str]]:
        """
        Refactor the provided code from CUDA to ROCm.
        Returns: (refactored_code, list_of_changes, trace_log)
        """
        self.changes = []
        self.trace_log = []
        
        refactored = code
        code_type = analysis_result.code_type
        
        self.trace_log.append(f"🔄 Starting refactoring for {code_type} code...")
        
        # Apply transformations in order
        if code_type == "python":
            refactored = self._refactor_python(refactored, analysis_result)
        elif code_type == "cpp":
            refactored = self._refactor_cpp(refactored, analysis_result)
        elif code_type == "dockerfile":
            refactored = self._refactor_dockerfile(refactored, analysis_result)
        elif code_type == "requirements":
            refactored = self._refactor_requirements(refactored, analysis_result)
        
        # Add ROCm header comment
        refactored = self._add_migration_header(refactored, code_type)
        
        self.trace_log.append(f"✅ Refactoring complete — {len(self.changes)} transformations applied")
        
        return refactored, self.changes, self.trace_log
    
    def _refactor_python(self, code: str, analysis) -> str:
        """Refactor Python/PyTorch code."""
        lines = code.split("\n")
        result_lines = []
        
        for i, line in enumerate(lines):
            original = line
            modified = line
            change_notes = []
            
            # 1. Replace environment variables
            for cuda_var, rocm_var in ENV_VAR_MAPPINGS.items():
                if cuda_var in modified:
                    modified = modified.replace(cuda_var, rocm_var)
                    change_notes.append(f"Env var: {cuda_var} → {rocm_var}")
            
            # 2. Replace CUDA CLI tools in comments/strings
            for nv_tool, rocm_tool in CLI_TOOL_MAPPINGS.items():
                if nv_tool in modified:
                    modified = modified.replace(nv_tool, rocm_tool)
                    change_notes.append(f"CLI tool: {nv_tool} → {rocm_tool}")
            
            # 3. Replace CUDA_HOME paths
            if "/usr/local/cuda" in modified:
                modified = modified.replace("/usr/local/cuda-12.1", "/opt/rocm")
                modified = modified.replace("/usr/local/cuda-12.0", "/opt/rocm")
                modified = modified.replace("/usr/local/cuda-11.8", "/opt/rocm")
                modified = modified.replace("/usr/local/cuda", "/opt/rocm")
                change_notes.append("Path: /usr/local/cuda → /opt/rocm")
            
            # 4. Replace CUDA version references
            if "torch.version.cuda" in modified:
                modified = modified.replace("torch.version.cuda", "torch.version.hip")
                change_notes.append("API: torch.version.cuda → torch.version.hip")
            
            # 5. Replace cuDNN with MIOpen backend references
            if "torch.backends.cudnn" in modified:
                # cudnn.benchmark and cudnn.deterministic work on ROCm via MIOpen
                # but we add a helpful comment
                if not any("MIOpen" in l for l in result_lines[-3:] if result_lines):
                    change_notes.append("Info: torch.backends.cudnn works on ROCm via MIOpen backend")
            
            # 6. Handle flash_attention_2 → sdpa
            if 'attn_implementation="flash_attention_2"' in modified or "attn_implementation='flash_attention_2'" in modified:
                modified = modified.replace('"flash_attention_2"', '"sdpa"')
                modified = modified.replace("'flash_attention_2'", "'sdpa'")
                change_notes.append("Attention: flash_attention_2 → sdpa (PyTorch native, ROCm compatible)")
            
            # 7. Handle enforce_eager for CUDA graphs
            if "enforce_eager=False" in modified and "CUDA graphs" in line.lower() or "cuda graph" in line.lower():
                modified = modified.replace("enforce_eager=False", "enforce_eager=True")
                change_notes.append("Config: enforce_eager=True (CUDA graphs have limited ROCm support)")
            
            # 8. Replace cuda pip install URLs
            if "download.pytorch.org/whl/cu" in modified:
                modified = re.sub(
                    r'download\.pytorch\.org/whl/cu\d+',
                    'download.pytorch.org/whl/rocm6.2',
                    modified
                )
                change_notes.append("URL: PyTorch CUDA wheel → ROCm 6.2 wheel")
            
            # 9. Handle bitsandbytes
            if "bitsandbytes" in modified and "rocm" not in modified.lower():
                if "pip install bitsandbytes" in modified:
                    modified = modified.replace("pip install bitsandbytes", "pip install bitsandbytes-rocm")
                    change_notes.append("Package: bitsandbytes → bitsandbytes-rocm")
            
            # 10. Replace CUDA-specific error messages
            if "CUDA is not available" in modified:
                modified = modified.replace("CUDA is not available", "ROCm/HIP is not available")
                change_notes.append("Message: Updated error text for ROCm")
            if "No CUDA GPU" in modified:
                modified = modified.replace("No CUDA GPU", "No AMD GPU")
                change_notes.append("Message: Updated error text for AMD")
            if "CUDA not available" in modified:
                modified = modified.replace("CUDA not available", "ROCm/HIP not available")
                change_notes.append("Message: Updated error text for ROCm")
            if "Install CUDA" in modified:
                modified = modified.replace("Install CUDA drivers", "Install ROCm drivers")
                modified = modified.replace("Install CUDA toolkit", "Install ROCm toolkit")
                change_notes.append("Message: Updated install instructions for ROCm")
            if "CUDA GPU required" in modified:
                modified = modified.replace("CUDA GPU required", "AMD GPU with ROCm required")
                change_notes.append("Message: Updated GPU requirement text")
            
            # 11. Transform CUDA C code inside Python strings (inline kernels)
            cuda_c_inline = {
                "cuda_runtime.h": "hip/hip_runtime.h",
                "cuda.h": "hip/hip_runtime.h",
                "cudaMalloc": "hipMalloc",
                "cudaFree": "hipFree",
                "cudaMemcpy": "hipMemcpy",
                "cudaMemset": "hipMemset",
                "cudaDeviceSynchronize": "hipDeviceSynchronize",
                "cudaGetLastError": "hipGetLastError",
                "cudaGetErrorString": "hipGetErrorString",
                "cudaError_t": "hipError_t",
                "cudaSuccess": "hipSuccess",
                "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
                "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
            }
            for cuda_tok, hip_tok in cuda_c_inline.items():
                if cuda_tok in modified:
                    modified = modified.replace(cuda_tok, hip_tok)
                    change_notes.append(f"Inline CUDA: {cuda_tok} → {hip_tok}")
            
            # 12. Rename cuda_sources variable to hip_sources
            if "cuda_sources" in modified and "load_inline" not in modified:
                modified = modified.replace("cuda_sources", "hip_sources")
                change_notes.append("Variable: cuda_sources → hip_sources")
            
            # Record changes
            if modified != original:
                for note in change_notes:
                    self.changes.append({
                        "line": i + 1,
                        "original": original.strip(),
                        "modified": modified.strip(),
                        "note": note,
                        "confidence": self._confidence(note),
                    })
            
            result_lines.append(modified)
        
        self.trace_log.append(f"🐍 Python refactoring: {len(self.changes)} changes applied")
        return "\n".join(result_lines)
    
    def _refactor_cpp(self, code: str, analysis) -> str:
        """Refactor C/C++ CUDA code to HIP."""
        lines = code.split("\n")
        result_lines = []
        
        for i, line in enumerate(lines):
            original = line
            modified = line
            change_notes = []
            
            # 1. Replace CUDA headers
            for cuda_header, hip_header in HEADER_MAPPINGS.items():
                if cuda_header in modified:
                    modified = modified.replace(cuda_header, hip_header)
                    change_notes.append(f"Header: {cuda_header} → {hip_header}")
            
            # 2. Replace CUDA API calls
            for cuda_api, hip_api in CUDA_TO_HIP_API.items():
                if cuda_api in modified:
                    modified = modified.replace(cuda_api, hip_api)
                    change_notes.append(f"API: {cuda_api} → {hip_api}")
            
            # 3. Replace library calls
            for cuda_lib, rocm_lib in CUDA_TO_ROCM_LIBS.items():
                if cuda_lib in modified and cuda_lib not in ["nvcc", "NVCC"]:
                    modified = modified.replace(cuda_lib, rocm_lib)
                    change_notes.append(f"Library: {cuda_lib} → {rocm_lib}")
            
            # Record changes
            if modified != original:
                for note in change_notes:
                    self.changes.append({
                        "line": i + 1,
                        "original": original.strip(),
                        "modified": modified.strip(),
                        "note": note,
                        "confidence": self._confidence(note),
                    })
            
            result_lines.append(modified)
        
        # === Hardware-Aware Refactoring Pass ===
        # Second pass: detect and rewrite Tensor Core / Warp-level patterns
        final_lines = []
        all_code = "\n".join(result_lines)
        for i, line in enumerate(result_lines):
            modified = line
            # WMMA/Tensor Core → rocWMMA/MFMA
            for hw_key, hw_info in HARDWARE_AWARE_MAPPINGS.items():
                if hw_key == "32":
                    # Context-dependent: only replace '32' when near warp operations
                    contexts = hw_info.get("context", [])
                    window = "\n".join(result_lines[max(0, i-2):min(len(result_lines), i+3)])
                    if any(ctx in window for ctx in contexts):
                        # Replace hardcoded 32 with wavefront-safe value
                        if re.search(r'\b32\b', modified):
                            original = modified
                            modified = re.sub(r'\b32\b', hw_info["replacement"], modified, count=1)
                            if modified != original:
                                self.changes.append({
                                    "line": i + 1,
                                    "original": original.strip(),
                                    "modified": modified.strip(),
                                    "note": hw_info["note"],
                                    "confidence": "⚠️ Review",
                                })
                else:
                    if hw_key in modified:
                        original = modified
                        modified = modified.replace(hw_key, hw_info["replacement"])
                        self.changes.append({
                            "line": i + 1,
                            "original": original.strip(),
                            "modified": modified.strip(),
                            "note": hw_info["note"],
                            "confidence": "⚠️ Review",
                        })
            final_lines.append(modified)
        
        self.trace_log.append(f"⚙️ C++/CUDA refactoring: {len(self.changes)} changes applied")
        self.trace_log.append(f"🔬 Hardware-aware pass: Tensor Core → MFMA, Warp → Wavefront checks applied")
        return "\n".join(final_lines)
    
    def _refactor_dockerfile(self, code: str, analysis) -> str:
        """Refactor Dockerfile from NVIDIA to ROCm."""
        lines = code.split("\n")
        result_lines = []
        skip_nvidia_env = False
        
        for i, line in enumerate(lines):
            original = line
            modified = line
            change_notes = []
            
            # 1. Replace base images
            for nv_image, rocm_image in DOCKER_IMAGE_MAPPINGS.items():
                if nv_image in modified:
                    modified = modified.replace(nv_image, rocm_image)
                    change_notes.append(f"Base image: NVIDIA → ROCm")
            
            # 2. Replace NVIDIA env vars
            if "NVIDIA_VISIBLE_DEVICES" in modified:
                modified = "# NVIDIA_VISIBLE_DEVICES not needed — use --device flags in docker run"
                change_notes.append("Docker: Removed NVIDIA_VISIBLE_DEVICES")
            
            if "NVIDIA_DRIVER_CAPABILITIES" in modified:
                modified = "# NVIDIA_DRIVER_CAPABILITIES not needed on ROCm"
                change_notes.append("Docker: Removed NVIDIA_DRIVER_CAPABILITIES")
            
            # 3. Replace CUDA paths
            if "CUDA_HOME" in modified:
                modified = modified.replace("CUDA_HOME", "ROCM_HOME")
                modified = modified.replace("/usr/local/cuda", "/opt/rocm")
                change_notes.append("Path: CUDA_HOME → ROCM_HOME, /usr/local/cuda → /opt/rocm")
            
            # 4. Replace nvidia-smi
            if "nvidia-smi" in modified:
                modified = modified.replace("nvidia-smi", "rocm-smi")
                change_notes.append("CLI: nvidia-smi → rocm-smi")
            
            # 5. Replace PyTorch install commands
            if "download.pytorch.org/whl/cu" in modified:
                modified = re.sub(
                    r'download\.pytorch\.org/whl/cu\d+',
                    'download.pytorch.org/whl/rocm6.2',
                    modified
                )
                change_notes.append("URL: CUDA PyTorch wheel → ROCm wheel")
            
            # 6. Replace CUDA verification
            if 'assert torch.cuda.is_available()' in modified:
                modified = modified.replace(
                    'assert torch.cuda.is_available(), \'CUDA not available!\'',
                    'assert torch.cuda.is_available(), \'ROCm/HIP not available!\''
                )
                change_notes.append("Message: CUDA → ROCm/HIP")
            
            # Record changes
            if modified != original:
                for note in change_notes:
                    self.changes.append({
                        "line": i + 1,
                        "original": original.strip(),
                        "modified": modified.strip(),
                        "note": note,
                        "confidence": self._confidence(note),
                    })
            
            result_lines.append(modified)
        
        # Add ROCm-specific Docker instructions at the end
        result_lines.append("")
        result_lines.append("# === ROCm Docker Run Command ===")
        result_lines.append("# docker run --device=/dev/kfd --device=/dev/dri \\")
        result_lines.append("#   --group-add video --group-add render \\")
        result_lines.append("#   --ipc=host --shm-size=16g \\")
        result_lines.append("#   -e HIP_VISIBLE_DEVICES=0 \\")
        result_lines.append("#   your-image-name")
        
        self.trace_log.append(f"🐳 Dockerfile refactoring: {len(self.changes)} changes applied")
        return "\n".join(result_lines)
    
    def _refactor_requirements(self, code: str, analysis) -> str:
        """Refactor requirements.txt for ROCm compatibility."""
        lines = code.split("\n")
        result_lines = [
            "# ============================================================",
            "# ROCm Forge — Migrated Requirements for AMD GPU",
            "# IMPORTANT: Install PyTorch first:",
            "#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2",
            "# Then: pip install -r requirements.txt",
            "# ============================================================",
            "",
        ]
        
        from knowledge.cuda_mappings import PIP_PACKAGE_MAPPINGS
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            
            pkg_name = re.split(r'[>=<~!\[]', stripped)[0].strip()
            
            if pkg_name in PIP_PACKAGE_MAPPINGS:
                info = PIP_PACKAGE_MAPPINGS[pkg_name]
                action = info["action"]
                
                if action == "remove":
                    result_lines.append(f"# REMOVED: {stripped}  — {info['note']}")
                    self.changes.append({
                        "line": i + 1,
                        "original": stripped,
                        "modified": f"# REMOVED: {stripped}",
                        "note": info["note"],
                        "confidence": "✅ Safe",
                    })
                elif action == "replace":
                    install_cmd = info.get("install", "")
                    result_lines.append(f"# REPLACED: {stripped}")
                    result_lines.append(f"# Install with: {install_cmd}")
                    self.changes.append({
                        "line": i + 1,
                        "original": stripped,
                        "modified": f"# Install: {install_cmd}",
                        "note": f"Replace with {info.get('replacement', 'ROCm build')}",
                        "confidence": "⚠️ Review",
                    })
                elif action == "warning":
                    result_lines.append(f"{stripped}  # ⚠️ {info.get('note', 'Check ROCm compatibility')}")
                    self.changes.append({
                        "line": i + 1,
                        "original": stripped,
                        "modified": f"{stripped} (warning added)",
                        "note": info.get("note", ""),
                        "confidence": "⚠️ Review",
                    })
                else:
                    result_lines.append(f"{stripped}  # ℹ️ {info.get('note', '')}")
            else:
                result_lines.append(stripped)
        
        self.trace_log.append(f"📦 Requirements refactoring: {len(self.changes)} changes applied")
        return "\n".join(result_lines)
    
    def _add_migration_header(self, code: str, code_type: str) -> str:
        """Add ROCm Forge migration header to the code."""
        if code_type == "python":
            header = (
                "# ============================================================\n"
                "# 🔥 Migrated by ROCm Forge — CUDA → AMD ROCm/HIP\n"
                "# This code has been automatically migrated for AMD GPU compatibility.\n"
                "# Target: AMD Instinct MI300X / ROCm 6.2\n"
                "# ============================================================\n\n"
            )
        elif code_type == "cpp":
            header = (
                "// ============================================================\n"
                "// 🔥 Migrated by ROCm Forge — CUDA → AMD HIP\n"
                "// This code has been automatically migrated for AMD GPU.\n"
                "// Compile with: hipcc instead of nvcc\n"
                "// Target: AMD Instinct MI300X / ROCm 6.2\n"
                "// ============================================================\n\n"
            )
        elif code_type == "dockerfile":
            header = (
                "# ============================================================\n"
                "# 🔥 Migrated by ROCm Forge — NVIDIA → AMD ROCm\n"
                "# This Dockerfile has been migrated for AMD GPU deployment.\n"
                "# ============================================================\n\n"
            )
        else:
            header = (
                "# ============================================================\n"
                "# 🔥 Migrated by ROCm Forge — CUDA → AMD ROCm\n"
                "# ============================================================\n\n"
            )
        
        return header + code
