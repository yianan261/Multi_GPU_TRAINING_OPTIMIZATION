/*%****************************************************************************
%  Code: 
%   logregv4check.cu
%
%  Purpose:
%   Optimize kernel-level performance for single-GPU training and scaling
%   performance with multi-GPUs optimization. This example uses logistic regression
%   for binary classification training on 8192 features and 1 million sample size.
%  
%  Modified:
%   May 12 2025 
%
%  Author:
%    Yian Chen
%
%  How to Compile:
%   nvcc -o logregv4check logregv4check.cu -lnccl -lcuda -lcudart -Xcompiler "-fopenmp"  
%
%  Execute:
%   ./logregv4check
%                             
%  See readme for CLI commands
%
%*****************************************************************************/

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define MAX_GPUS      8
#define NUM_FEATURES  8192    
#define NUM_SAMPLES   1000000
#define BATCH_SIZE    8192    
#define NUM_EPOCHS    5
#define LEARNING_RATE 0.01f
#define BLOCK_SIZE    256
#define TILE_F        256
#define PIPELINE_DEPTH 3   

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);}}while(0)
#define NCCL_CHECK(x) do{ncclResult_t r=(x); if(r!=ncclSuccess){ \
    printf("NCCL error %s:%d: %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); \
    exit(EXIT_FAILURE);}}while(0)

__device__ __forceinline__ float sigmoidf(float z){ return 1.f/(1.f+__expf(-z)); }


/* ───────────── Row-major kernels ───────────── */
__global__ void forward_rm(const float* X, const float* W, float* logit, int batch, int nf) {
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if(s >= batch) return;
    float acc = 0.f;
    for(int f = 0; f < nf; ++f)
        acc += X[(size_t)s*nf + f] * W[f];
    logit[s] = sigmoidf(acc);
}

__global__ void backward_rm(const float* X, const float* pred, const int* y, float* g, int batch, int nf) {
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    if(f >= nf) return;
    float acc = 0.f;
    for(int s = 0; s < batch; ++s) {
        float err = pred[s] - (float)y[s];
        acc += X[(size_t)s*nf + f] * err;
    }
    g[f] = acc / (float)batch;
}

/* ───────── Column-major kernels ───────── */
__global__ void forward_cm(const float* __restrict__ X, const float* __restrict__ W,
                          float* __restrict__ logit, int batch, int nf) {
    __shared__ float ws[TILE_F];
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if(s >= batch) return;
    float acc = 0.f;
    for(int tile = 0; tile < nf; tile += TILE_F) {
        int f = tile + threadIdx.x;
        if(f < nf) ws[threadIdx.x] = W[f];
        __syncthreads();
        #pragma unroll 4
        for(int k = 0; k < TILE_F && tile+k < nf; ++k)
            acc += X[(tile+k)*batch + s] * ws[k];
        __syncthreads();
    }
    logit[s] = sigmoidf(acc);
}

__global__ void backward_cm(const float* __restrict__ X, const float* __restrict__ pred,
                           const int* __restrict__ y, float* __restrict__ g,
                           int batch, int nf) {
    __shared__ float err_sh[BLOCK_SIZE];
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    if(f >= nf) return;
    float acc = 0.f;
    for(int base = 0; base < batch; base += BLOCK_SIZE) {
        int tid = threadIdx.x;
        if(base+tid < batch) err_sh[tid] = pred[base+tid] - (float)y[base+tid];
        __syncthreads();
        #pragma unroll 4
        for(int i = 0; i < BLOCK_SIZE && base+i < batch; ++i)
            acc += X[f*batch + (base+i)] * err_sh[i];
        __syncthreads();
    }
    g[f] = acc / (float)batch;
}

/* ─────── column-major ─────── */
__global__ void pack_to_cm(const float* src, float* dst, int batch, int nf, int ofs) {
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    if(f >= nf || s >= batch) return;
    dst[f*batch + s] = src[(size_t)(ofs+s)*nf + f];
}

//helper
__global__ void scale_kernel(float* g, int n, float s) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) g[i] *= s;
}

__global__ void sgd_kernel(float* w, const float* g, float lr, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) w[i] -= lr * g[i];
}

__global__ void scale_and_sgd_kernel(float* w, const float* g, float lr, float scale, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) w[i] -= lr * (g[i] * scale);
}

int main(int argc, char** argv) {
    int num_gpus = 1;
    bool use_opt = false, use_cm = false, use_fused = false;
    const char* mode = "Single-GPU Baseline";
    //CLI commands
    if(argc > 1) {
        if(!strcmp(argv[1], "--optimized")) { use_opt = true; mode = "Single-GPU Optimized"; }
        else if(!strcmp(argv[1], "--cm")) { use_opt = true; use_cm = true; mode = "Single-GPU CM-Optimized"; }
        else if(!strcmp(argv[1], "--naive")) { mode = "Naive Multi-GPU"; num_gpus = (argc>2)?atoi(argv[2]):3; }
        else if(!strcmp(argv[1], "--fused")) { 
            mode = "Fused Multi-GPU"; 
            num_gpus = (argc>2)?atoi(argv[2]):3;
            use_opt = true; 
            use_fused = (NUM_FEATURES > 2048); // disable for small feature size
        }
    }
    
    bool actually_fused = use_fused && (NUM_FEATURES > 2048);
    if (actually_fused) use_cm = true; // column-major for fused large problems
    
    printf("\n[Benchmark] %s | %d GPU(s) | Features: %d | Batch: %d\n", 
           mode, num_gpus, NUM_FEATURES, BATCH_SIZE);

    /* ----- host dataset ----- */
    const size_t bytes_X = (size_t)NUM_SAMPLES*NUM_FEATURES*sizeof(float);
    float *h_X = (float*)malloc(bytes_X);
    int   *h_y = (int*)malloc(NUM_SAMPLES*sizeof(int));
    CUDA_CHECK(cudaHostRegister(h_X, bytes_X, 0));
    CUDA_CHECK(cudaHostRegister(h_y, NUM_SAMPLES*sizeof(int), 0));

    // pinned mini buffers
    float* h_Xpin[PIPELINE_DEPTH];
    int*   h_ypin[PIPELINE_DEPTH];
    for(int i = 0; i < PIPELINE_DEPTH; ++i) {
        CUDA_CHECK(cudaHostAlloc(&h_Xpin[i], BATCH_SIZE*NUM_FEATURES*sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_ypin[i], BATCH_SIZE*sizeof(int), cudaHostAllocDefault));
    }

    float *d_Xrm[MAX_GPUS][PIPELINE_DEPTH] = {{nullptr}};
    float *d_Xcm[MAX_GPUS][PIPELINE_DEPTH] = {{nullptr}};
    float *d_W[MAX_GPUS], *d_logits[MAX_GPUS], *d_grad[MAX_GPUS];
    int   *d_y[MAX_GPUS];
    cudaStream_t h2d_stream[MAX_GPUS], comp_stream[MAX_GPUS], comm_stream[MAX_GPUS];
    cudaEvent_t copyDone[MAX_GPUS][PIPELINE_DEPTH], gradReady[MAX_GPUS][PIPELINE_DEPTH];
    int devs[MAX_GPUS];

    for(int g = 0; g < num_gpus; ++g) {
        devs[g] = g; 
        CUDA_CHECK(cudaSetDevice(g));
        
        CUDA_CHECK(cudaStreamCreate(&h2d_stream[g]));
        CUDA_CHECK(cudaStreamCreate(&comp_stream[g]));
        CUDA_CHECK(cudaStreamCreate(&comm_stream[g]));
        
        for(int b = 0; b < PIPELINE_DEPTH; ++b) {
            CUDA_CHECK(cudaMalloc(&d_Xrm[g][b], BATCH_SIZE*NUM_FEATURES*sizeof(float)));
            if(use_cm) CUDA_CHECK(cudaMalloc(&d_Xcm[g][b], NUM_FEATURES*BATCH_SIZE*sizeof(float)));
            CUDA_CHECK(cudaEventCreate(&copyDone[g][b]));
            CUDA_CHECK(cudaEventCreate(&gradReady[g][b]));
        }
        
        CUDA_CHECK(cudaMalloc(&d_y[g], BATCH_SIZE*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_logits[g], BATCH_SIZE*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W[g], NUM_FEATURES*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad[g], NUM_FEATURES*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_W[g], 0, NUM_FEATURES*sizeof(float)));
    }
    
    ncclComm_t comms[MAX_GPUS]; 
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

    dim3 blkF(BLOCK_SIZE), grdF((BATCH_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 blkB(BLOCK_SIZE), grdB((NUM_FEATURES+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 blkPk(32, 8);

    int total_batches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
    int iters_per_gpu = (total_batches + num_gpus - 1) / num_gpus;

    cudaEvent_t t0, t1; 
    CUDA_CHECK(cudaEventCreate(&t0)); 
    CUDA_CHECK(cudaEventCreate(&t1));
    cudaEventRecord(t0);
    long long total_flops = 0;

    /* ---------------- training ---------------- */
    for(int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        for(int it = 0; it < iters_per_gpu; ++it) {
            int buf = it % PIPELINE_DEPTH;  // 3-way ping-pong
            size_t global_off = (size_t)it * num_gpus * BATCH_SIZE;

            //host memcpy into pinned minibuffer
            for(int g = 0; g < num_gpus; ++g) {
                size_t off = global_off + (size_t)g * BATCH_SIZE;
                size_t rem = NUM_SAMPLES > off ? NUM_SAMPLES - off : 0;
                int batch_this = rem < BATCH_SIZE ? rem : BATCH_SIZE;
                if(!batch_this) continue;

                // async H2H copy
                memcpy(h_Xpin[buf], h_X + off*NUM_FEATURES, batch_this*NUM_FEATURES*sizeof(float));
                memcpy(h_ypin[buf], h_y + off, batch_this*sizeof(int));

                //async H2D copy
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyAsync(d_Xrm[g][buf], h_Xpin[buf],
                                batch_this*NUM_FEATURES*sizeof(float),
                                cudaMemcpyHostToDevice, h2d_stream[g]));
                CUDA_CHECK(cudaMemcpyAsync(d_y[g], h_ypin[buf],
                                batch_this*sizeof(int),
                                cudaMemcpyHostToDevice, h2d_stream[g]));
                CUDA_CHECK(cudaEventRecord(copyDone[g][buf], h2d_stream[g]));

                // optional cm
                if(use_cm) {
                    dim3 grdPk((NUM_FEATURES+blkPk.x-1)/blkPk.x,
                             (batch_this+blkPk.y-1)/blkPk.y);
                    CUDA_CHECK(cudaStreamWaitEvent(h2d_stream[g], copyDone[g][buf], 0));
                    pack_to_cm<<<grdPk, blkPk, 0, h2d_stream[g]>>>(d_Xrm[g][buf], d_Xcm[g][buf],
                                                                    batch_this, NUM_FEATURES, 0);
                }

                const float* X_in = use_cm ? d_Xcm[g][buf] : d_Xrm[g][buf];
                
                CUDA_CHECK(cudaStreamWaitEvent(comp_stream[g], copyDone[g][buf], 0));
                
                forward_cm<<<grdF, blkF, 0, comp_stream[g]>>>(X_in, d_W[g], d_logits[g],
                                                             batch_this, NUM_FEATURES);
                
                backward_cm<<<grdB, blkB, 0, comp_stream[g]>>>(X_in, d_logits[g], d_y[g],
                                                              d_grad[g], batch_this, NUM_FEATURES);
                
                CUDA_CHECK(cudaEventRecord(gradReady[g][buf], comp_stream[g]));
                total_flops += (long long)batch_this * NUM_FEATURES * 5;
            }

            if (actually_fused) {
                ncclGroupStart();
                for(int g = 0; g < num_gpus; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    CUDA_CHECK(cudaStreamWaitEvent(comm_stream[g], gradReady[g][buf], 0));
                    NCCL_CHECK(ncclAllReduce(d_grad[g], d_grad[g], NUM_FEATURES, 
                                           ncclFloat, ncclSum, comms[g], comm_stream[g]));
                }
                ncclGroupEnd();

                for(int g = 0; g < num_gpus; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    scale_and_sgd_kernel<<<grdB, blkB, 0, comm_stream[g]>>>(d_W[g], d_grad[g],
                                                                          LEARNING_RATE, 1.f/num_gpus,
                                                                          NUM_FEATURES);
                }
            } else {
                for(int g = 0; g < num_gpus; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    CUDA_CHECK(cudaStreamWaitEvent(comm_stream[g], gradReady[g][buf], 0));
                }
                
                ncclGroupStart();
                for(int g = 0; g < num_gpus; ++g) {
                    NCCL_CHECK(ncclAllReduce(d_grad[g], d_grad[g], NUM_FEATURES, 
                                           ncclFloat, ncclSum, comms[g], comm_stream[g]));
                }
                ncclGroupEnd();
                
                for(int g = 0; g < num_gpus; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    // wait for NCCL to complete
                    CUDA_CHECK(cudaStreamSynchronize(comm_stream[g]));
                    
                    scale_kernel<<<grdB, blkB, 0, 0>>>(d_grad[g], NUM_FEATURES, 1.f/num_gpus);
                    sgd_kernel<<<grdB, blkB, 0, 0>>>(d_W[g], d_grad[g], LEARNING_RATE, NUM_FEATURES);
                }
            }
        }
        printf("Epoch %d/%d complete (%s, %d GPU%s)\n",
               epoch+1, NUM_EPOCHS, mode, num_gpus, (num_gpus==1?"":"s"));
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double seconds = ms/1000.0;
    double gflops = (total_flops/1e9)/seconds;
    double throughput = (double)NUM_SAMPLES*NUM_EPOCHS/seconds;

    float* h_W = (float*)calloc(NUM_FEATURES, sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_W, d_W[0], NUM_FEATURES*sizeof(float), cudaMemcpyDeviceToHost));

    size_t correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for(int i = 0; i < NUM_SAMPLES; ++i) {
        float z = 0.f;
        for(int f = 0; f < NUM_FEATURES; ++f) 
            z += h_X[(size_t)i*NUM_FEATURES+f] * h_W[f];
        if((1.f/(1.f+expf(-z)) > 0.5f == h_y[i])) ++correct;
    }

    printf("\n| GPUs | Mode      |   Time | Throughput (samples/s) | GFLOPS | Acc |\n");
    printf("|------|-----------|-------:|-----------------------:|-------:|----:|\n");
    printf("| %3d  | %-9s | %6.3f | %23.2f | %6.1f | %4.2f%% |\n",
           num_gpus, mode, seconds, throughput, gflops,
           100.0*(double)correct/NUM_SAMPLES);

    //cleanup
    for(int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for(int b = 0; b < PIPELINE_DEPTH; ++b) {
            CUDA_CHECK(cudaFree(d_Xrm[g][b]));
            if(use_cm) CUDA_CHECK(cudaFree(d_Xcm[g][b]));
            CUDA_CHECK(cudaEventDestroy(copyDone[g][b]));
            CUDA_CHECK(cudaEventDestroy(gradReady[g][b]));
        }
        CUDA_CHECK(cudaFree(d_y[g]));
        CUDA_CHECK(cudaFree(d_logits[g]));
        CUDA_CHECK(cudaFree(d_W[g]));
        CUDA_CHECK(cudaFree(d_grad[g]));
        CUDA_CHECK(cudaStreamDestroy(h2d_stream[g]));
        CUDA_CHECK(cudaStreamDestroy(comp_stream[g]));
        CUDA_CHECK(cudaStreamDestroy(comm_stream[g]));
    }
    NCCL_CHECK(ncclCommDestroy(comms[0]));
    for(int i = 0; i < PIPELINE_DEPTH; ++i) {
        CUDA_CHECK(cudaFreeHost(h_Xpin[i]));
        CUDA_CHECK(cudaFreeHost(h_ypin[i]));
    }
    CUDA_CHECK(cudaHostUnregister(h_X));
    CUDA_CHECK(cudaHostUnregister(h_y));
    free(h_X); free(h_y); free(h_W);
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}