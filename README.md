# Efficient Multi-GPU Gradient Aggregation and Optimization with CUDA Kernels and NCCL
## Introduction
Training machine learning models efficiently across multiple GPUs is a fundamental challenge in high-performance computing and parallel programming. In this project, we 
implement an optimized multi-GPU logistic regression training
system in CUDA. We evaluated and compared different strategies, including single-GPU (baseline and optimized), naive
multi-GPU, and fused multi-GPU setups. Some optimizations
include using column-major memory layouts, multi-buffer
pipelining, and fused compute-communication kernels using
NCCL, inspired by the paper [Optimizing Distributed ML
Communication with Fused Computation-Collective Opera
tions](https://arxiv.org/abs/2305.06942). Our main goal was to reduce communication-
computation bottlenecks and scale performance with increas
ing GPUs and feature sizes. To achieve efficient multi-
GPU training, we need to optimize both single and multi-
GPU inter-device communications. Therefore this project also
demonstrates with low-level kernel code to show kernel-level
optimizations. At the kernel level, we maximize per-GPU
throughput with memory coalescing by ensuring threads in
a warp access contiguous memory addresses. We explore
column-major memory layout by transposing our matrix so
features are contiguous, we also explore shared memory and
register tiling for coalesced reads, and use loop unrolling
to reduce loop overhead. For multi-GPU optimizations, we
combine communication, computation, and data-transfer in
a pipeline, using grouped NCCL AllReduce to help fuse
AllReduce to a collective communication plan across devices.
We also explore 3-stream pipeline per GPU to help parallel
execution of copy, compute, and communication for different
mini-batches on each GPU. With 3 buffers
on host and device, we also achieve 3-way ping-pong to
prevent pipeline stalls. We also use fused gradient scaling for
the stochastic gradient descent kernel to reduce kernel launch
overhead and minimize latency between communication.

## Kernels
`forward_rm(...)` kernel: Baseline forward pass using row-major layout.
Each row is a sample, columns are features.
Computes:

$$\text{logit}[s] = \sigma\left(\sum_{f=0}^{nf} X[s \cdot nf + f] \cdot W[f]\right)$$


`backward_rm(...)` kernel: Each thread handles a different weight index f, this computes
the gradient w.r.t weights for row-major layout

$$g[f] = \frac{1}{\text{batch}} \sum_s (p[s] - y[s]) \cdot X[s \cdot nf + f]$$


`forward_cm(...)` kernel: Optimizes forward pass for column-major layout (where features are stored contiguously). Shared memory `ws` loads a tile of weights into shared memory.

`backward_cm(...)` kernel: Computes one weight gradient using stride-based access.

#### Utility Kernels:

`scale_kernel(...)` scales gradient by scalar.
`sgd_kernel(...)` stochastic gradient descent 
`scale_and_sgd_kernel(...)` combines scale + SGD in a single kernel to reduce launch overhead and memory bandwidth


## Hardware:
For this project, we use 5 Tesla V100 GPUs

## To run project:
```bash
nvcc -o logregv4check logregv4check.cu -lnccl -lcuda -lcudart -Xcompiler "-fopenmp"
```
CLI options:

Naive single-GPU (baseline)
```bash
./logregv4check 
```

Optimized single-GPU
```bash
./logregv4check --optimized
```

Optimized single-GPU (column-major layout)
```bash
./logregv4check --cm
```

Naive multi-GPU
```bash
./logregv4check --naive <number of GPUs>
```

Fused/optimized multi-GPU 
```bash
./logregv4check --fused <number of GPUs>
```

## Results

| GPUs | Mode                    | Time (s) | Throughput (samples/s) | GFLOPS | Accuracy  |
|------|-------------------------|----------|--------------------------|--------|-----------|
| 1    | Single-GPU Baseline     | 51.782   | 96,558.52                | 4.0    | 100.00%   |
| 1    | Single-GPU Optimized    | 50.833   | 98,361.59                | 4.0    | 100.00%   |
| 1    | Single-GPU CM-Optimized | 53.972   | 92,640.92                | 3.8    | 100.00%   |
| 3    | Naive Multi-GPU         | 41.219   | 121,301.99               | 5.0    | 100.00%   |
| 5    | Naive Multi-GPU         | 38.903   | 128,525.15               | 5.3    | 100.00%   |
| 3    | Fused Multi-GPU         | 36.853   | 135,674.73               | 5.6    | 100.00%   |
| 5    | Fused Multi-GPU         | 36.577   | 136,697.65               | 5.6    | 100.00%   |

## Analysis
The lab results clearly demonstrate the benefits of multi-
GPU parallelism and pipeline-aware optimization for logistic
regression training (see Table 1 for results). As the number
of GPUs increases from 1 to 5, throughput improves signifi-
cantly—from 96K samples/sec to over 136K—while GFLOPS
rises from 4.0 to 5.6, showing better hardware utilization.
Before single-GPU optimization (at the kernel-level), the time
was 51.782 s for the computations, and GFLOPS was 4.0.
After optimization, the speed improved to 50.833 s, GLFOPS
remaining the same. This optimization still uses the row-
major layout for simplicity. We notice when we use column-
major layout optimization, the time increased to 53.972, and
throughput and GFLOPS drop. This could be because of
packing overhead; we explicitly transpose the input matrix in
pack_to_cm kernel, which is an extra kernel launch and
extra global memory write; the benefit from the CM kernels
does not appear significant. The extra synchronization over-
head could also cause the delay. On the other hand for multi-
GPU optimizations, we notice the fused multi-GPU strategy
consistently outperforms naive approaches by approximately
10% on 3 GPUs, validating the effectiveness of compute-
communication overlap and fused kernel execution. 
