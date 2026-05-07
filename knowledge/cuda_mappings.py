"""
ROCm Forge — CUDA to ROCm/HIP Comprehensive Mapping Database
Contains all API, library, package, environment variable, and header mappings
needed for autonomous CUDA-to-AMD migration.
"""

# =============================================================================
# CUDA Runtime API → HIP Runtime API Mappings
# =============================================================================
CUDA_TO_HIP_API = {
    # Memory Management
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyAsync": "hipMemcpyAsync",
    "cudaMemset": "hipMemset",
    "cudaMemsetAsync": "hipMemsetAsync",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMallocManaged": "hipMallocManaged",
    "cudaMallocHost": "hipHostMalloc",
    "cudaFreeHost": "hipHostFree",
    "cudaHostAlloc": "hipHostMalloc",
    "cudaMemGetInfo": "hipMemGetInfo",
    "cudaMemcpy2D": "hipMemcpy2D",
    "cudaMallocPitch": "hipMallocPitch",
    "cudaPointerGetAttributes": "hipPointerGetAttributes",
    
    # Device Management
    "cudaGetDevice": "hipGetDevice",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaDeviceReset": "hipDeviceReset",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaDeviceGetAttribute": "hipDeviceGetAttribute",
    "cudaChooseDevice": "hipChooseDevice",
    
    # Stream Management
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamCreateWithFlags": "hipStreamCreateWithFlags",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaStreamWaitEvent": "hipStreamWaitEvent",
    "cudaStreamQuery": "hipStreamQuery",
    
    # Event Management
    "cudaEventCreate": "hipEventCreate",
    "cudaEventCreateWithFlags": "hipEventCreateWithFlags",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaEventQuery": "hipEventQuery",
    
    # Kernel Launch
    "cudaLaunchKernel": "hipLaunchKernel",
    "cudaFuncSetCacheConfig": "hipFuncSetCacheConfig",
    "cudaFuncGetAttributes": "hipFuncGetAttributes",
    
    # Error Handling
    "cudaGetLastError": "hipGetLastError",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    
    # Type Mappings
    "cudaError_t": "hipError_t",
    "cudaSuccess": "hipSuccess",
    "cudaStream_t": "hipStream_t",
    "cudaEvent_t": "hipEvent_t",
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaMemcpyKind": "hipMemcpyKind",
    
    # Texture & Surface (limited)
    "cudaCreateTextureObject": "hipCreateTextureObject",
    "cudaDestroyTextureObject": "hipDestroyTextureObject",
}


# =============================================================================
# CUDA Library → ROCm Library Mappings
# =============================================================================
CUDA_TO_ROCM_LIBS = {
    "cuBLAS": "rocBLAS",
    "cublas": "rocblas",
    "cublasCreate": "rocblas_create_handle",
    "cublasDestroy": "rocblas_destroy_handle",
    "cublasSgemm": "rocblas_sgemm",
    "cublasDgemm": "rocblas_dgemm",
    "cuDNN": "MIOpen",
    "cudnn": "miopen",
    "cudnnCreate": "miopenCreate",
    "cudnnDestroy": "miopenDestroy",
    "cuFFT": "rocFFT",
    "cufft": "rocfft",
    "cufftPlan1d": "rocfft_plan_create",
    "cufftExecC2C": "rocfft_execute",
    "cuSPARSE": "rocSPARSE",
    "cusparse": "rocsparse",
    "cuRAND": "rocRAND",
    "curand": "rocrand",
    "curandCreateGenerator": "rocrand_create_generator",
    "cuSOLVER": "rocSOLVER",
    "cusolver": "rocsolver",
    "NCCL": "RCCL",
    "nccl": "rccl",
    "ncclCommInitRank": "ncclCommInitRank",  # Same API in RCCL
    "Thrust": "rocThrust",
    "thrust": "rocthrust",
    "CUB": "hipCUB",
    "cub": "hipcub",
    "nvcc": "hipcc",
    "NVCC": "HIPCC",
}


# =============================================================================
# PyTorch-Specific CUDA → ROCm Patterns
# =============================================================================
PYTORCH_PATTERNS = {
    # These work on ROCm but may need attention
    "torch.backends.cudnn.benchmark": {
        "replacement": "torch.backends.cudnn.benchmark",
        "note": "Works on ROCm via MIOpen backend. Consider setting to True for performance.",
        "action": "info",
    },
    "torch.backends.cudnn.deterministic": {
        "replacement": "torch.backends.cudnn.deterministic",
        "note": "Works on ROCm via MIOpen backend.",
        "action": "info",
    },
    "torch.backends.cudnn.enabled": {
        "replacement": "torch.backends.cudnn.enabled",
        "note": "Controls MIOpen on ROCm. Works transparently.",
        "action": "info",
    },
    "torch.cuda.amp": {
        "replacement": "torch.cuda.amp",
        "note": "AMP works on ROCm. GradScaler and autocast are supported.",
        "action": "compatible",
    },
    "torch.cuda.is_available()": {
        "replacement": "torch.cuda.is_available()",
        "note": "Returns True on ROCm when HIP is available. Works transparently.",
        "action": "compatible",
    },
    ".cuda()": {
        "replacement": ".cuda()",
        "note": "Works on ROCm. Moves tensors to HIP device.",
        "action": "compatible",
    },
    "torch.cuda.device_count()": {
        "replacement": "torch.cuda.device_count()",
        "note": "Returns number of AMD GPUs on ROCm.",
        "action": "compatible",
    },
    "torch.cuda.set_device": {
        "replacement": "torch.cuda.set_device",
        "note": "Works on ROCm.",
        "action": "compatible",
    },
    "torch.cuda.current_device()": {
        "replacement": "torch.cuda.current_device()",
        "note": "Works on ROCm.",
        "action": "compatible",
    },
    "torch.cuda.empty_cache()": {
        "replacement": "torch.cuda.empty_cache()",
        "note": "Works on ROCm.",
        "action": "compatible",
    },
    "torch.cuda.memory_allocated": {
        "replacement": "torch.cuda.memory_allocated",
        "note": "Works on ROCm for memory tracking.",
        "action": "compatible",
    },
}


# =============================================================================
# pip Package Migration Mappings
# =============================================================================
PIP_PACKAGE_MAPPINGS = {
    # Nvidia runtime packages (remove)
    "nvidia-cuda-runtime-cu11": {"action": "remove", "note": "Not needed on ROCm. GPU runtime is provided by ROCm stack."},
    "nvidia-cuda-runtime-cu12": {"action": "remove", "note": "Not needed on ROCm. GPU runtime is provided by ROCm stack."},
    "nvidia-cuda-nvrtc-cu11": {"action": "remove", "note": "Not needed on ROCm."},
    "nvidia-cuda-nvrtc-cu12": {"action": "remove", "note": "Not needed on ROCm."},
    "nvidia-cublas-cu11": {"action": "remove", "note": "Replaced by rocBLAS (system package)."},
    "nvidia-cublas-cu12": {"action": "remove", "note": "Replaced by rocBLAS (system package)."},
    "nvidia-cudnn-cu11": {"action": "remove", "note": "Replaced by MIOpen (system package)."},
    "nvidia-cudnn-cu12": {"action": "remove", "note": "Replaced by MIOpen (system package)."},
    "nvidia-cufft-cu11": {"action": "remove", "note": "Replaced by rocFFT (system package)."},
    "nvidia-cufft-cu12": {"action": "remove", "note": "Replaced by rocFFT (system package)."},
    "nvidia-cusparse-cu11": {"action": "remove", "note": "Replaced by rocSPARSE (system package)."},
    "nvidia-cusparse-cu12": {"action": "remove", "note": "Replaced by rocSPARSE (system package)."},
    "nvidia-cusolver-cu11": {"action": "remove", "note": "Replaced by rocSOLVER (system package)."},
    "nvidia-cusolver-cu12": {"action": "remove", "note": "Replaced by rocSOLVER (system package)."},
    "nvidia-nccl-cu11": {"action": "remove", "note": "Replaced by RCCL (system package)."},
    "nvidia-nccl-cu12": {"action": "remove", "note": "Replaced by RCCL (system package)."},
    "nvidia-nvtx-cu11": {"action": "remove", "note": "ROCm has rocTracer for profiling."},
    "nvidia-nvtx-cu12": {"action": "remove", "note": "ROCm has rocTracer for profiling."},
    
    # PyTorch ecosystem (replace with ROCm builds)
    "torch": {
        "action": "replace",
        "replacement": "torch (ROCm 6.2 build)",
        "install": "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2",
    },
    "torchvision": {
        "action": "replace",
        "replacement": "torchvision (ROCm 6.2 build)",
        "install": "pip install torchvision --index-url https://download.pytorch.org/whl/rocm6.2",
    },
    "torchaudio": {
        "action": "replace",
        "replacement": "torchaudio (ROCm 6.2 build)",
        "install": "pip install torchaudio --index-url https://download.pytorch.org/whl/rocm6.2",
    },
    
    # Quantization & optimization libraries
    "bitsandbytes": {
        "action": "replace",
        "replacement": "bitsandbytes-rocm",
        "install": "pip install bitsandbytes-rocm",
    },
    "auto-gptq": {
        "action": "warning",
        "note": "ROCm support varies. Check https://github.com/AutoGPTQ/AutoGPTQ for ROCm-compatible releases.",
    },
    "awq": {
        "action": "warning",
        "note": "Limited ROCm support. Consider using GPTQ or native quantization instead.",
    },
    
    # Attention & kernel libraries
    "xformers": {
        "action": "warning",
        "note": "Limited ROCm support. Use PyTorch native SDPA (torch.nn.functional.scaled_dot_product_attention) instead.",
    },
    "flash-attn": {
        "action": "warning",
        "note": "Flash Attention has experimental ROCm support. Use PyTorch SDPA as a reliable alternative.",
    },
    
    # Compiler
    "triton": {
        "action": "replace",
        "replacement": "triton (ROCm build)",
        "install": "pip install triton --index-url https://download.pytorch.org/whl/rocm6.2",
    },
    
    # Serving
    "vllm": {
        "action": "replace",
        "replacement": "vllm (ROCm build)",
        "install": "pip install vllm",
        "note": "vLLM supports ROCm natively. Use ROCm-compatible PyTorch.",
    },
    
    # Distributed training
    "deepspeed": {
        "action": "info",
        "note": "DeepSpeed supports ROCm. Install with ROCm-compatible PyTorch and set DS_BUILD_OPS=1.",
    },
    "horovod": {
        "action": "info",
        "note": "Horovod has ROCm support. Build from source with HOROVOD_GPU=ROCM.",
    },
}


# =============================================================================
# Environment Variable Mappings
# =============================================================================
ENV_VAR_MAPPINGS = {
    "CUDA_VISIBLE_DEVICES": "HIP_VISIBLE_DEVICES",
    "CUDA_HOME": "ROCM_HOME",
    "CUDA_PATH": "ROCM_PATH",
    "CUDA_ROOT": "ROCM_PATH",
    "CUDA_LAUNCH_BLOCKING": "HIP_LAUNCH_BLOCKING",
    "CUDA_DEVICE_ORDER": "HIP_DEVICE_ORDER",
    "NCCL_DEBUG": "NCCL_DEBUG",  # Same in RCCL
    "NCCL_SOCKET_IFNAME": "NCCL_SOCKET_IFNAME",  # Same in RCCL
    "CUDA_DEVICE_MAX_CONNECTIONS": "GPU_MAX_HW_QUEUES",
}


# =============================================================================
# C/C++ Header Mappings
# =============================================================================
HEADER_MAPPINGS = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda_fp16.h": "hip/hip_fp16.h",
    "cuda_bf16.h": "hip/hip_bf16.h",
    "cublas_v2.h": "rocblas/rocblas.h",
    "cublas.h": "rocblas/rocblas.h",
    "cudnn.h": "miopen/miopen.h",
    "cufft.h": "rocfft/rocfft.h",
    "cusparse.h": "rocsparse/rocsparse.h",
    "curand.h": "rocrand/rocrand.h",
    "curand_kernel.h": "rocrand/rocrand_kernel.h",
    "cusolver_common.h": "rocsolver/rocsolver.h",
    "cusolverDn.h": "rocsolver/rocsolver.h",
    "nccl.h": "rccl/rccl.h",
    "nvToolsExt.h": "roctracer/roctx.h",
    "cooperative_groups.h": "hip/hip_cooperative_groups.h",
    "thrust/device_vector.h": "thrust/device_vector.h",  # rocThrust compatible
    "thrust/host_vector.h": "thrust/host_vector.h",
}


# =============================================================================
# CUDA Kernel Syntax → HIP Kernel Syntax
# =============================================================================
KERNEL_SYNTAX_MAPPINGS = {
    "__global__": "__global__",  # Same in HIP
    "__device__": "__device__",  # Same in HIP
    "__host__": "__host__",      # Same in HIP
    "__shared__": "__shared__",  # Same in HIP
    "__constant__": "__constant__",  # Same in HIP
    "__syncthreads()": "__syncthreads()",  # Same in HIP
    "threadIdx": "threadIdx",  # Same in HIP
    "blockIdx": "blockIdx",    # Same in HIP
    "blockDim": "blockDim",    # Same in HIP
    "gridDim": "gridDim",      # Same in HIP
    "warpSize": "warpSize",    # Same in HIP (but check: 64 on AMD vs 32 on NVIDIA)
}


# =============================================================================
# Hardware-Aware Mappings (Tough Engineering)
# =============================================================================
HARDWARE_AWARE_MAPPINGS = {
    # Warp vs Wavefront rewriting
    "32": {
        "context": ["__syncwarp", "warp_size", "warpSize"],
        "replacement": "__AMDGCN_WAVEFRONT_SIZE",
        "note": "Hardware-Aware Refactoring: Hardcoded Warp Size (32) replaced with dynamic AMD Wavefront Size (64)."
    },
    # NVIDIA Tensor Core -> AMD Matrix Core
    "wmma::mma_sync": {
        "replacement": "__builtin_amdgcn_mfma_f32_32x32x1f32",
        "note": "Intrinsic Lowering: Translated NVIDIA Tensor Core mma.sync to AMD Matrix Core (MFMA) intrinsic."
    },
    "nvcuda::wmma": {
        "replacement": "rocwmma",
        "note": "Library Abstraction: Replaced nvcuda::wmma namespace with rocwmma."
    },
    "wmma::": {
        "replacement": "rocwmma::",
        "note": "Library Abstraction: Replaced NVIDIA wmma namespace with AMD rocwmma. Verification required."
    }
}


# =============================================================================
# Known Incompatibilities & Warnings
# =============================================================================
KNOWN_ISSUES = {
    "warp_size": {
        "pattern": "warpSize",
        "severity": "warning",
        "message": "AMD GPUs use warp size 64 (wavefront) vs NVIDIA's 32. Code assuming warpSize==32 will break.",
        "fix": "Use __builtin_amdgcn_wavefrontsize() or check warpSize at runtime.",
    },
    "cooperative_groups": {
        "pattern": "cooperative_groups",
        "severity": "warning",
        "message": "Cooperative groups have limited support in HIP. Test thoroughly.",
        "fix": "Use basic __syncthreads() where possible, or check HIP cooperative groups API.",
    },
    "dynamic_parallelism": {
        "pattern": "cudaLaunchKernel.*<<<",
        "severity": "error",
        "message": "Dynamic parallelism (launching kernels from kernels) is not supported on most AMD GPUs.",
        "fix": "Restructure code to launch all kernels from the host.",
    },
    "tensor_cores": {
        "pattern": "wmma|mma\\.sync|tensor_core",
        "severity": "warning",
        "message": "NVIDIA Tensor Core operations need to be replaced with AMD Matrix Core (MFMA) operations.",
        "fix": "Use rocWMMA library or PyTorch's native mixed precision.",
    },
    "cudnn_specific": {
        "pattern": "cudnnSetConvolution|cudnnFindConvolution",
        "severity": "info",
        "message": "cuDNN convolution APIs map to MIOpen but with different tuning behavior.",
        "fix": "MIOpen auto-tunes by default. Set MIOPEN_FIND_MODE for control.",
    },
}


# =============================================================================
# CLI Tool Mappings
# =============================================================================
CLI_TOOL_MAPPINGS = {
    "nvidia-smi": "rocm-smi",
    "nvcc": "hipcc",
    "cuda-memcheck": "rocm-debug-agent",
    "nvprof": "rocprof",
    "nsight-compute": "omniperf",
    "nsight-systems": "omnitrace",
    "cuda-gdb": "rocgdb",
    "deviceQuery": "rocminfo",
}


# =============================================================================
# Docker Base Image Mappings
# =============================================================================
DOCKER_IMAGE_MAPPINGS = {
    "nvidia/cuda:11.8.0-devel-ubuntu22.04": "rocm/dev-ubuntu-22.04:6.2-complete",
    "nvidia/cuda:12.1.0-devel-ubuntu22.04": "rocm/dev-ubuntu-22.04:6.2-complete",
    "nvidia/cuda:12.2.0-devel-ubuntu22.04": "rocm/dev-ubuntu-22.04:6.2-complete",
    "nvidia/cuda:11.8.0-runtime-ubuntu22.04": "rocm/dev-ubuntu-22.04:6.2",
    "nvidia/cuda:12.1.0-runtime-ubuntu22.04": "rocm/dev-ubuntu-22.04:6.2",
    "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel": "rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0",
    "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel": "rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0",
}


# =============================================================================
# Implicit CUDA Assumptions — Curiosity-Driven Exploration Scan
# (Inspired by curiosity-driven RL: detect patterns that AREN'T explicitly
# CUDA API calls but silently assume NVIDIA hardware behavior)
# =============================================================================
IMPLICIT_CUDA_PATTERNS = {
    "hardcoded_warp_32": {
        "regex": r"\b32\b",
        "context_required": ["thread", "warp", "lane", "mask", "shuffle", "ballot", "shfl", "__syncwarp", "blockDim"],
        "severity": "critical",
        "message": "Hardcoded value 32 in thread/warp context. AMD wavefronts are 64-wide — this WILL produce silent wrong results.",
        "fix": "Replace with warpSize or __builtin_amdgcn_wavefrontsize() for portability.",
    },
    "hardcoded_sm_count": {
        "regex": r"(?:num_sm|sm_count|multiprocessor|SM_COUNT)\s*=\s*\d+",
        "context_required": [],
        "severity": "warning",
        "message": "Hardcoded Streaming Multiprocessor count. AMD uses Compute Units (CUs) with different counts per GPU.",
        "fix": "Query device properties at runtime: hipDeviceGetAttribute(&val, hipDeviceAttributeMultiprocessorCount, dev).",
    },
    "shared_mem_bank_32": {
        "regex": r"(?:bank|BANK).*\b32\b|__shfl(?:_sync)?",
        "context_required": [],
        "severity": "warning",
        "message": "Shared memory bank conflict assumptions. AMD GCN/RDNA shared memory has 32 banks (same) but different conflict resolution.",
        "fix": "Test for bank conflicts using rocprof. AMD LDS has different padding requirements.",
    },
    "l2_cache_residency": {
        "regex": r"cudaAccessPolicyWindow|cudaStreamAttrValue|accessPolicyWindow|l2_cache|L2_CACHE",
        "context_required": [],
        "severity": "warning",
        "message": "CUDA L2 cache residency controls have no direct AMD equivalent.",
        "fix": "AMD uses different L2 cache hierarchy. Remove L2 residency hints; use rocprof to tune.",
    },
    "ptx_inline_asm": {
        "regex": r"asm\s*\(\s*\".*(?:mov|ld|st|add|mul|setp|bar|shfl|atom).*\"",
        "context_required": [],
        "severity": "critical",
        "message": "Inline PTX assembly detected. PTX is NVIDIA-specific ISA — completely incompatible with AMD.",
        "fix": "Replace with HIP C++ intrinsics or AMD GCN inline assembly (__builtin_amdgcn_*).",
    },
    "cuda_graph_capture": {
        "regex": r"cudaStreamBeginCapture|cudaGraphLaunch|cuda\.CUDAGraph|with\s+torch\.cuda\.graph",
        "context_required": [],
        "severity": "warning",
        "message": "CUDA Graphs have limited and experimental support on ROCm. May cause hangs or errors.",
        "fix": "Use enforce_eager=True in vLLM. For custom code, test hipGraphLaunch carefully or remove graph capture.",
    },
    "tensor_core_mma": {
        "regex": r"mma\.sync|wmma::|nvcuda::wmma|__hmma|mma_sync",
        "context_required": [],
        "severity": "critical",
        "message": "NVIDIA Tensor Core (WMMA/MMA) intrinsics detected. These require complete rewrite for AMD Matrix Cores (MFMA).",
        "fix": "Use rocWMMA library or replace with __builtin_amdgcn_mfma_* intrinsics.",
    },
    "occupancy_calculator": {
        "regex": r"cudaOccupancyMaxPotentialBlockSize|cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "context_required": [],
        "severity": "warning",
        "message": "CUDA occupancy API. AMD has different occupancy characteristics due to 64-wide wavefronts and different register files.",
        "fix": "Use hipOccupancyMaxPotentialBlockSize. Note: optimal block sizes differ on AMD (prefer multiples of 64).",
    },
}


# =============================================================================
# ROCm Build Error Runbook — Incident Copilot Database
# (Inspired by incident-response copilots: maps common build/runtime errors
# to root causes and automatic fixes)
# =============================================================================
ROCM_BUILD_ERROR_RUNBOOK = {
    "hip_not_found": {
        "error_pattern": r"hip/hip_runtime\.h.*No such file|cannot find -lhip",
        "root_cause": "ROCm SDK not installed or not in PATH",
        "fix_steps": [
            "Install ROCm: sudo apt install rocm-dev",
            "Set environment: export ROCM_HOME=/opt/rocm",
            "Add to PATH: export PATH=$ROCM_HOME/bin:$PATH",
        ],
        "severity": "critical",
    },
    "hipcc_not_found": {
        "error_pattern": r"hipcc.*not found|hipcc.*No such file",
        "root_cause": "HIP compiler not installed or not in PATH",
        "fix_steps": [
            "Install HIP: sudo apt install hip-dev",
            "Verify: which hipcc",
            "Add to PATH: export PATH=/opt/rocm/bin:$PATH",
        ],
        "severity": "critical",
    },
    "unsupported_gpu_arch": {
        "error_pattern": r"unsupported gpu architecture|Cannot determine AMD GPU",
        "root_cause": "GPU architecture not specified or not supported by this ROCm version",
        "fix_steps": [
            "Check GPU: rocminfo | grep 'Name:'",
            "Set target: export HIP_VISIBLE_DEVICES=0",
            "Compile with arch: hipcc --offload-arch=gfx942 (MI300X) or gfx90a (MI250)",
        ],
        "severity": "critical",
    },
    "rocblas_not_found": {
        "error_pattern": r"cannot find -lrocblas|rocblas\.h.*No such file",
        "root_cause": "rocBLAS library not installed",
        "fix_steps": [
            "Install: sudo apt install rocblas-dev",
            "Link path: export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH",
        ],
        "severity": "error",
    },
    "miopen_error": {
        "error_pattern": r"miopenStatus.*Error|MIOpen.*failed|miopen.*not found",
        "root_cause": "MIOpen (cuDNN equivalent) configuration issue",
        "fix_steps": [
            "Install: sudo apt install miopen-hip-dev",
            "Set tuning: export MIOPEN_FIND_MODE=3",
            "Clear cache: rm -rf ~/.config/miopen/",
        ],
        "severity": "error",
    },
    "hip_out_of_memory": {
        "error_pattern": r"hipErrorOutOfMemory|HIP out of memory|RuntimeError.*out of memory",
        "root_cause": "GPU memory exhausted — may need different allocation strategy for AMD",
        "fix_steps": [
            "Set: export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True",
            "Reduce batch size or model precision",
            "Monitor: rocm-smi --showmeminfo vram",
        ],
        "severity": "error",
    },
    "warp_size_mismatch": {
        "error_pattern": r"warpSize.*32|lane.*out of range|invalid shuffle",
        "root_cause": "Code assumes NVIDIA warp size (32) but AMD wavefronts are 64-wide",
        "fix_steps": [
            "Replace hardcoded 32 with warpSize or __builtin_amdgcn_wavefrontsize()",
            "Update __shfl_sync masks to cover 64 lanes",
            "Check all bitwise operations on lane masks",
        ],
        "severity": "critical",
    },
    "rccl_timeout": {
        "error_pattern": r"RCCL.*timeout|NCCL.*timeout.*RCCL|collective.*timeout",
        "root_cause": "Multi-GPU communication timeout — RCCL (NCCL equivalent) needs tuning",
        "fix_steps": [
            "Set: export NCCL_SOCKET_IFNAME=eth0",
            "Increase timeout: export RCCL_TIMEOUT=600",
            "Check GPU topology: rocm-smi --showtoponuma",
        ],
        "severity": "error",
    },
}
