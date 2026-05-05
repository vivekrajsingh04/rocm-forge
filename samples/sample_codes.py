"""
ROCm Forge — Sample CUDA/Nvidia Code for Demo
Realistic code samples that users can load to test the migration agent.
"""

SAMPLES = {
    # =================================================================
    # Sample 1: PyTorch ResNet Training on CUDA
    # =================================================================
    "pytorch_training": {
        "title": "🧠 PyTorch ResNet Training (CUDA)",
        "description": "A typical PyTorch training script using CUDA with mixed precision, cuDNN tuning, and distributed setup.",
        "code": '''import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import os

# CUDA configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# cuDNN optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def setup_distributed():
    """Initialize distributed training with NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def train():
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install CUDA toolkit.")
    
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Model
    model = torchvision.models.resnet50(pretrained=True)
    model = model.cuda()
    
    # Mixed precision training
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = torchvision.datasets.FakeData(
        size=10000, image_size=(3, 224, 224), num_classes=1000, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, GPU Mem: {gpu_mem:.2f} GB")
        
        scheduler.step()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(dataloader):.4f}")
    
    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "checkpoint.pth")
    print("Training complete!")

if __name__ == "__main__":
    train()
''',
    },

    # =================================================================
    # Sample 2: LLM Inference with vLLM on CUDA
    # =================================================================
    "llm_inference": {
        "title": "🤖 LLM Inference with vLLM (CUDA)",
        "description": "A vLLM-based LLM serving script configured for NVIDIA GPUs with tensor parallelism.",
        "code": '''"""LLM Inference Server using vLLM on NVIDIA GPU"""
import os
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# NVIDIA GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
os.environ["NCCL_DEBUG"] = "INFO"

def check_gpu():
    """Verify CUDA GPU availability."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Install CUDA drivers.")
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"GPU Count: {torch.cuda.device_count()}")

def run_inference():
    """Run LLM inference using vLLM."""
    check_gpu()
    
    # Initialize vLLM with NVIDIA optimizations
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=False,  # Use CUDA graphs
        enable_prefix_caching=True,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.1,
    )
    
    # Run batch inference
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate Fibonacci numbers.",
        "What are the benefits of AMD GPUs for AI workloads?",
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\\nPrompt: {prompt}")
        print(f"Response: {generated_text[:200]}...")
    
    # GPU memory stats
    print(f"\\nGPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

if __name__ == "__main__":
    run_inference()
''',
    },

    # =================================================================
    # Sample 3: Custom CUDA Kernel via PyTorch Extension
    # =================================================================
    "custom_kernel": {
        "title": "⚡ Custom CUDA Kernel (C++ Extension)",
        "description": "A custom CUDA kernel integrated via PyTorch C++ extensions. Demonstrates low-level GPU programming.",
        "code": '''"""
Custom CUDA Kernel Example — Vector Addition
Demonstrates writing a custom CUDA kernel and integrating it with PyTorch.
This code must be converted to HIP for AMD GPU compatibility.
"""
import torch
import torch.utils.cpp_extension

# Inline CUDA kernel compiled at runtime
cuda_source = """
#include <cuda_runtime.h>
#include <cuda.h>

// CUDA kernel: vector addition
__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel: fused multiply-add
__global__ void fused_mul_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = alpha * a[idx] * b[idx] + c[idx];
    }
}

// Launcher function
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto n = a.size(0);
    auto c = torch::empty_like(a);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel error: ") + cudaGetErrorString(err)
        );
    }
    
    cudaDeviceSynchronize();
    return c;
}

torch::Tensor fused_mul_add_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor c, float alpha
) {
    auto n = a.size(0);
    auto out = torch::empty_like(a);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    fused_mul_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n
    );
    
    cudaDeviceSynchronize();
    return out;
}
"""

cpp_source = """
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor fused_mul_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, float alpha);
"""

def main():
    """Demonstrate custom CUDA kernel usage."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Compile the CUDA extension inline
    custom_ops = torch.utils.cpp_extension.load_inline(
        name="custom_cuda_ops",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["vector_add_cuda", "fused_mul_add_cuda"],
        verbose=True,
    )
    
    # Test the custom kernel
    n = 1_000_000
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.randn(n, device="cuda")
    
    # Vector addition
    result = custom_ops.vector_add_cuda(a, b)
    expected = a + b
    assert torch.allclose(result, expected, atol=1e-5), "Vector add failed!"
    print(f"✅ Vector addition: PASSED ({n} elements)")
    
    # Fused multiply-add
    alpha = 2.0
    result_fma = custom_ops.fused_mul_add_cuda(a, b, c, alpha)
    expected_fma = alpha * a * b + c
    assert torch.allclose(result_fma, expected_fma, atol=1e-5), "FMA failed!"
    print(f"✅ Fused multiply-add: PASSED ({n} elements)")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = custom_ops.vector_add_cuda(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"⚡ Benchmark: {1000/elapsed:.0f} kernel calls/sec")

if __name__ == "__main__":
    main()
''',
    },

    # =================================================================
    # Sample 4: Hugging Face Fine-Tuning on CUDA
    # =================================================================
    "hf_finetuning": {
        "title": "🔬 Hugging Face Fine-Tuning (CUDA)",
        "description": "Fine-tuning a Hugging Face model with LoRA on NVIDIA GPUs using bitsandbytes quantization.",
        "code": '''"""
Fine-tuning a Hugging Face model with QLoRA on NVIDIA GPU
Uses bitsandbytes for 4-bit quantization and PEFT for LoRA adapters.
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# CUDA configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def main():
    # Verify CUDA
    assert torch.cuda.is_available(), "CUDA GPU required for fine-tuning!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    # 4-bit quantization config (bitsandbytes + CUDA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Uses flash-attn (CUDA)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]")
    
    def format_prompt(example):
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['response']}"
        return tokenizer(text, truncation=True, max_length=512, padding="max_length")
    
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,  # NVIDIA mixed precision
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",  # bitsandbytes optimizer
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save
    model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")
    
    # Final GPU stats
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
''',
    },

    # =================================================================
    # Sample 5: Dockerfile with NVIDIA Base Image
    # =================================================================
    "nvidia_dockerfile": {
        "title": "🐳 Dockerfile (NVIDIA CUDA Base)",
        "description": "A production Dockerfile using NVIDIA CUDA base images for AI model serving.",
        "code": '''# Production Dockerfile for AI Model Serving on NVIDIA GPU
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# NVIDIA environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# System dependencies
RUN apt-get update && apt-get install -y \\
    python3 python3-pip python3-dev \\
    git wget curl \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app/

# Verify CUDA setup
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
RUN nvidia-smi

# Expose port for serving
EXPOSE 8000

# Run the model server
CMD ["python3", "server.py"]
''',
    },

    # =================================================================
    # Sample 6: WMMA Tensor Core Kernel (HARDEST MIGRATION)
    # This is the "tough engineering" sample — demonstrates ROCm Forge
    # handling intrinsic-level hardware translation that standard
    # hipify tools completely fail on.
    # =================================================================
    "tensor_core_wmma": {
        "title": "🔬 Tensor Core WMMA Kernel (Advanced)",
        "description": "NVIDIA Tensor Core matrix multiply using WMMA intrinsics. This is the HARDEST migration case — requires intrinsic lowering from NVIDIA mma.sync to AMD Matrix Core (MFMA).",
        "code": '''/**
 * NVIDIA Tensor Core Matrix Multiply using WMMA API
 * This kernel uses warp-level matrix operations that are
 * fundamentally tied to NVIDIA hardware architecture.
 * 
 * Migration difficulty: ADVANCED
 * - WMMA intrinsics have no direct hipify mapping
 * - Warp size 32 is hardcoded throughout
 * - Tensor Core tile sizes are NVIDIA-specific
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // NVIDIA WMMA header

using namespace nvcuda;

// WMMA tile dimensions (NVIDIA Tensor Core specific)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Matrix dimensions
const int M = 4096;
const int N = 4096;
const int K = 4096;

__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Warp-level indices (NVIDIA warp = 32 threads)
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare WMMA fragments (NVIDIA Tensor Core data structures)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrix tiles into WMMA fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Tensor Core matrix multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

int main() {
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Launch configuration (32 threads per warp)
    dim3 threads(32, 4);
    dim3 blocks((M + 32 - 1) / 32, (N + WMMA_N - 1) / WMMA_N);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    wmma_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        return 1;
    }

    // Calculate TFLOPS
    double tflops = (2.0 * M * N * K) / (milliseconds * 1e9);
    printf("WMMA GEMM: %.2f ms, %.2f TFLOPS\\n", milliseconds, tflops);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',
    },
}
