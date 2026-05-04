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
