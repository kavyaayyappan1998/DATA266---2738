
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define TILE 16
#define CPU_TILE 32
#define CPU_RUNS 3
#define GPU_RUNS 5

// Stop if a CUDA command fails
#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        std::cerr << cudaGetErrorString(e) << std::endl; \
        std::exit(1); \
    } \
} while (0)

// Each block has 16 x 16 = 256 threads.
// Each thread calculates one value in output matrix C.
__global__ void matrixGPU(const float* A, const float* B, float* C, int N) {
    __shared__ float aTile[TILE][TILE];
    __shared__ float bTile[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        aTile[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0;

        bTile[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// CPU baseline
void matrixCPU(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C, int N) {

    std::fill(C.begin(), C.end(), 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += CPU_TILE)
        for (int jj = 0; jj < N; jj += CPU_TILE)
            for (int kk = 0; kk < N; kk += CPU_TILE)
                for (int i = ii; i < std::min(ii + CPU_TILE, N); i++)
                    for (int k = kk; k < std::min(kk + CPU_TILE, N); k++)
                        for (int j = jj; j < std::min(jj + CPU_TILE, N); j++)
                            C[i*N+j] += A[i*N+k] * B[k*N+j];
}

// Average CPU time over 3 runs
double timeCPU(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C, int N) {

    // Small warm-up
    std::vector<float> a(64*64, 1), b(64*64, 1), c(64*64);
    matrixCPU(a, b, c, 64);

    double total = 0;

    for (int r = 0; r < CPU_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        matrixCPU(A, B, C, N);
        auto stop = std::chrono::high_resolution_clock::now();

        total += std::chrono::duration<double, std::milli>(
            stop - start
        ).count();
    }

    return total / CPU_RUNS;
}

struct GPUTimes {
    double kernel;
    double transfer;
};

// Measure kernel and H2D+D2H transfer separately
GPUTimes timeGPU(const std::vector<float>& A,
                 const std::vector<float>& B,
                 std::vector<float>& C,
                 int N, int runs) {

    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    dim3 threads(TILE, TILE);
    dim3 blocks(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    // GPU warm-up
    CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    matrixGPU<<<blocks, threads>>>(dA, dB, dC, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    double kernelTotal = 0, transferTotal = 0;

    for (int r = 0; r < runs; r++) {
        float h2d, kernel, d2h;

        // CPU -> GPU
        CHECK(cudaEventRecord(start));
        CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&h2d, start, stop));

        // GPU kernel
        CHECK(cudaEventRecord(start));
        matrixGPU<<<blocks, threads>>>(dA, dB, dC, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&kernel, start, stop));

        // GPU -> CPU
        CHECK(cudaEventRecord(start));
        CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&d2h, start, stop));

        kernelTotal += kernel;
        transferTotal += h2d + d2h;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return {kernelTotal / runs, transferTotal / runs};
}

// Run one matrix size
void runTest(int N) {
    size_t count = (size_t)N * N;

    // Simple values are enough for a timing comparison
    std::vector<float> A(count, 1.0f);
    std::vector<float> B(count, 1.0f);
    std::vector<float> Ccpu(count);
    std::vector<float> Cgpu(count);

    double cpu = timeCPU(A, B, Ccpu, N);
    GPUTimes gpu = timeGPU(A, B, Cgpu, N, GPU_RUNS);

    double endToEnd = gpu.kernel + gpu.transfer;
    double speedup = cpu / endToEnd;

    // Keep the result observable so CPU work is not optimized away
    volatile float check = Ccpu[0] + Cgpu[0];
    (void)check;

    std::cout << N << ","
              << cpu << ","
              << gpu.kernel << ","
              << gpu.transfer << ","
              << endToEnd << ","
              << speedup << std::endl;
}

// Used when running the profiler
void runGPUOnly(int N) {
    size_t count = (size_t)N * N;

    std::vector<float> A(count, 1.0f);
    std::vector<float> B(count, 1.0f);
    std::vector<float> C(count);

    GPUTimes gpu = timeGPU(A, B, C, N, 1);

    std::cout << "GPU_ONLY,N=" << N
              << ",kernel_ms=" << gpu.kernel
              << ",H2D_plus_D2H_ms=" << gpu.transfer
              << std::endl;
}

int main(int argc, char** argv) {
    CHECK(cudaSetDevice(0));

    // Used by the timing table
    if (argc == 3 && std::string(argv[1]) == "--size") {
        std::cout << "N,CPU_ms,GPU_kernel_ms,H2D_plus_D2H_ms,"
                     "GPU_end_to_end_ms,Speedup\n";

        runTest(std::stoi(argv[2]));
        return 0;
    }

    // Used by Nsight Compute / profiler
    if (argc == 3 && std::string(argv[1]) == "--gpu-only") {
        runGPUOnly(std::stoi(argv[2]));
        return 0;
    }

    // Run all required sizes if no argument is given
    std::cout << "N,CPU_ms,GPU_kernel_ms,H2D_plus_D2H_ms,"
                 "GPU_end_to_end_ms,Speedup\n";

    for (int N : {256, 1024, 4096})
        runTest(N);

    return 0;
}
