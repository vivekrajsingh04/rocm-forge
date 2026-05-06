"""
ROCm Forge — MI300X Benchmark Suite
Ready to run on AMD Developer Cloud when credits arrive.

Usage:
    python benchmark/rocm_benchmark.py --all
    python benchmark/rocm_benchmark.py --device-info
    python benchmark/rocm_benchmark.py --memory
    python benchmark/rocm_benchmark.py --compute
"""
import argparse
import time
import json
import os
import sys
from datetime import datetime


def check_rocm_available():
    """Check if ROCm/HIP runtime is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ No GPU detected. ROCm/HIP runtime not available.")
            print("   Make sure you're running on an AMD GPU instance with ROCm installed.")
            return False
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {device_name}")
        print(f"   PyTorch version: {torch.__version__}")
        hip_version = getattr(torch.version, 'hip', None)
        cuda_version = getattr(torch.version, 'cuda', None)
        if hip_version:
            print(f"   HIP version: {hip_version}")
        elif cuda_version:
            print(f"   CUDA version: {cuda_version}")
        return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def device_info_benchmark():
    """Collect detailed device information for the hackathon submission."""
    import torch

    results = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "hip_version": getattr(torch.version, 'hip', 'N/A'),
        "gpu_count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multi_processor_count": props.multi_processor_count,
            "major": props.major,
            "minor": props.minor,
        }
        results["devices"].append(device_info)

    print("\n📊 Device Information")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    return results


def memory_benchmark():
    """Test GPU memory allocation and bandwidth."""
    import torch

    device = torch.device("cuda:0")
    results = {"tests": []}

    sizes_mb = [256, 512, 1024, 2048, 4096]
    print("\n💾 Memory Benchmark")
    print("=" * 50)

    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // 4  # float32
        torch.cuda.synchronize()

        # Allocation
        start = time.perf_counter()
        tensor = torch.zeros(n_elements, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        alloc_time = (time.perf_counter() - start) * 1000

        # Fill (bandwidth test)
        start = time.perf_counter()
        tensor.fill_(1.0)
        torch.cuda.synchronize()
        fill_time = (time.perf_counter() - start) * 1000
        bandwidth_gbps = (size_mb / 1024) / (fill_time / 1000) if fill_time > 0 else 0

        # Copy D2D
        start = time.perf_counter()
        tensor2 = tensor.clone()
        torch.cuda.synchronize()
        copy_time = (time.perf_counter() - start) * 1000

        result = {
            "size_mb": size_mb,
            "alloc_ms": round(alloc_time, 2),
            "fill_ms": round(fill_time, 2),
            "bandwidth_gbps": round(bandwidth_gbps, 2),
            "copy_d2d_ms": round(copy_time, 2),
        }
        results["tests"].append(result)
        print(f"  {size_mb:>5} MB → alloc: {alloc_time:6.2f}ms  fill: {fill_time:6.2f}ms  bandwidth: {bandwidth_gbps:7.2f} GB/s  D2D: {copy_time:6.2f}ms")

        del tensor, tensor2
        torch.cuda.empty_cache()

    return results


def compute_benchmark():
    """Test raw compute performance with matrix operations."""
    import torch

    device = torch.device("cuda:0")
    results = {"tests": []}

    sizes = [1024, 2048, 4096, 8192]
    print("\n⚡ Compute Benchmark (GEMM)")
    print("=" * 50)

    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=torch.float32)
        b = torch.randn(n, n, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        num_runs = 10
        start = time.perf_counter()
        for _ in range(num_runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000

        # TFLOPS = 2 * N^3 / time
        tflops = (2 * n**3) / (elapsed / 1000) / 1e12

        result = {
            "matrix_size": n,
            "avg_ms": round(elapsed, 2),
            "tflops": round(tflops, 2),
        }
        results["tests"].append(result)
        print(f"  {n:>5}x{n} → {elapsed:8.2f} ms  ({tflops:6.2f} TFLOPS)")

        del a, b
        torch.cuda.empty_cache()

    # FP16 test
    print("\n⚡ Compute Benchmark (GEMM FP16)")
    print("=" * 50)
    for n in [4096, 8192]:
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)

        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        num_runs = 10
        start = time.perf_counter()
        for _ in range(num_runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        tflops = (2 * n**3) / (elapsed / 1000) / 1e12

        result = {
            "matrix_size": f"{n}_fp16",
            "avg_ms": round(elapsed, 2),
            "tflops": round(tflops, 2),
        }
        results["tests"].append(result)
        print(f"  {n:>5}x{n} FP16 → {elapsed:8.2f} ms  ({tflops:6.2f} TFLOPS)")

        del a, b
        torch.cuda.empty_cache()

    return results


def run_all_benchmarks():
    """Run all benchmarks and save results."""
    print("🔥 ROCm Forge — MI300X Benchmark Suite")
    print("=" * 50)

    if not check_rocm_available():
        print("\n⚠️  Run this on AMD Developer Cloud with MI300X GPU.")
        print("   python benchmark/rocm_benchmark.py --all")
        return

    results = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info_benchmark(),
        "memory": memory_benchmark(),
        "compute": compute_benchmark(),
    }

    # Save results
    os.makedirs("benchmark/results", exist_ok=True)
    output_file = f"benchmark/results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print("📸 Use these results as proof of AMD GPU usage in your hackathon submission!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROCm Forge MI300X Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--device-info", action="store_true", help="Show device information")
    parser.add_argument("--memory", action="store_true", help="Run memory benchmarks")
    parser.add_argument("--compute", action="store_true", help="Run compute benchmarks")
    args = parser.parse_args()

    if args.all:
        run_all_benchmarks()
    elif args.device_info:
        if check_rocm_available():
            device_info_benchmark()
    elif args.memory:
        if check_rocm_available():
            memory_benchmark()
    elif args.compute:
        if check_rocm_available():
            compute_benchmark()
    else:
        parser.print_help()
        print("\n💡 Quick start: python benchmark/rocm_benchmark.py --all")
