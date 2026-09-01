#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define TILE 16
#define CPU_TILE 32

const unsigned int SEED = 2738;
const int CPU_REPEATS = 3;
const int GPU_REPEATS = 5;

// Simple CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// CUDA matrix multiplication kernel
// Each thread calculates one value in output matrix C
__global__ void matrixMultiplyGPU(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int numberOfTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numberOfTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        if (row < N && aCol < N)
            tileA[threadIdx.y][threadIdx.x] =
                A[(size_t)row * N + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < N && col < N)
            tileB[threadIdx.y][threadIdx.x] =
                B[(size_t)bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[(size_t)row * N + col] = sum;
    }
}

// CPU matrix multiplication used as baseline
void matrixMultiplyCPU(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N
) {
    std::fill(C.begin(), C.end(), 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += CPU_TILE) {
        for (int jj = 0; jj < N; jj += CPU_TILE) {

            int iEnd = std::min(ii + CPU_TILE, N);
            int jEnd = std::min(jj + CPU_TILE, N);

            for (int kk = 0; kk < N; kk += CPU_TILE) {

                int kEnd = std::min(kk + CPU_TILE, N);

                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {

                        float a = A[(size_t)i * N + k];

                        for (int j = jj; j < jEnd; j++) {
                            C[(size_t)i * N + j] +=
                                a * B[(size_t)k * N + j];
                        }
                    }
                }
            }
        }
    }
}

struct GPUTiming {
    double kernel_ms;
    double transfer_ms;
};

// Fill matrix with random values
void fillRandom(
    std::vector<float>& values,
    std::mt19937& rng
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& value : values) {
        value = dist(rng);
    }
}

// Check CPU and GPU results
double maxAbsError(
    const std::vector<float>& A,
    const std::vector<float>& B
) {
    double maxError = 0.0;

    for (size_t i = 0; i < A.size(); i++) {
        maxError = std::max(
            maxError,
            (double)std::fabs(A[i] - B[i])
        );
    }

    return maxError;
}

// Time CPU version
double timeCPU(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N
) {
    // Small warm-up
    int W = 128;

    std::vector<float> warmA((size_t)W * W, 1.0f);
    std::vector<float> warmB((size_t)W * W, 1.0f);
    std::vector<float> warmC((size_t)W * W, 0.0f);

    matrixMultiplyCPU(warmA, warmB, warmC, W);

    double total = 0.0;

    for (int r = 0; r < CPU_REPEATS; r++) {

        auto start =
            std::chrono::high_resolution_clock::now();

        matrixMultiplyCPU(A, B, C, N);

        auto end =
            std::chrono::high_resolution_clock::now();

        total +=
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    }

    return total / CPU_REPEATS;
}

// Time GPU kernel and transfer separately
GPUTiming timeGPU(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N,
    int repeats
) {
    size_t elements = (size_t)N * N;
    size_t bytes = elements * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    // GPU warm-up
    CUDA_CHECK(
        cudaMemcpy(
            dA,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dB,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    matrixMultiplyGPU<<<blocks, threads>>>(
        dA,
        dB,
        dC,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    double kernelTotal = 0.0;
    double transferTotal = 0.0;

    for (int r = 0; r < repeats; r++) {

        float h2d = 0.0f;
        float kernel = 0.0f;
        float d2h = 0.0f;

        // Host to device transfer
        CUDA_CHECK(cudaEventRecord(startEvent));

        CUDA_CHECK(
            cudaMemcpy(
                dA,
                A.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                dB,
                B.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        CUDA_CHECK(
            cudaEventElapsedTime(
                &h2d,
                startEvent,
                stopEvent
            )
        );

        // GPU kernel time
        CUDA_CHECK(cudaEventRecord(startEvent));

        matrixMultiplyGPU<<<blocks, threads>>>(
            dA,
            dB,
            dC,
            N
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        CUDA_CHECK(
            cudaEventElapsedTime(
                &kernel,
                startEvent,
                stopEvent
            )
        );

        // Device to host transfer
        CUDA_CHECK(cudaEventRecord(startEvent));

        CUDA_CHECK(
            cudaMemcpy(
                C.data(),
                dC,
                bytes,
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        CUDA_CHECK(
            cudaEventElapsedTime(
                &d2h,
                startEvent,
                stopEvent
            )
        );

        kernelTotal += kernel;
        transferTotal += h2d + d2h;
    }

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    GPUTiming result;

    result.kernel_ms =
        kernelTotal / repeats;

    result.transfer_ms =
        transferTotal / repeats;

    return result;
}

// Run one matrix size
void runTest(int N) {

    size_t elements = (size_t)N * N;

    std::vector<float> A(elements);
    std::vector<float> B(elements);
    std::vector<float> Ccpu(elements, 0.0f);
    std::vector<float> Cgpu(elements, 0.0f);

    std::mt19937 rng(SEED + N);

    fillRandom(A, rng);
    fillRandom(B, rng);

    double cpuTime =
        timeCPU(A, B, Ccpu, N);

    GPUTiming gpu =
        timeGPU(
            A,
            B,
            Cgpu,
            N,
            GPU_REPEATS
        );

    double gpuEndToEnd =
        gpu.kernel_ms + gpu.transfer_ms;

    double speedup =
        cpuTime / gpuEndToEnd;

    double error =
        maxAbsError(Ccpu, Cgpu);

    std::cout
        << std::fixed
        << std::setprecision(4)
        << N << ","
        << cpuTime << ","
        << gpu.kernel_ms << ","
        << gpu.transfer_ms << ","
        << gpuEndToEnd << ","
        << speedup << ","
        << error
        << std::endl;
}

// Used only for profiler
void runGPUOnly(int N) {

    size_t elements = (size_t)N * N;

    std::vector<float> A(elements);
    std::vector<float> B(elements);
    std::vector<float> C(elements, 0.0f);

    std::mt19937 rng(SEED + N);

    fillRandom(A, rng);
    fillRandom(B, rng);

    GPUTiming gpu =
        timeGPU(A, B, C, N, 1);

    std::cout
        << "GPU_ONLY,"
        << "N=" << N << ","
        << "kernel_ms=" << gpu.kernel_ms << ","
        << "H2D_plus_D2H_ms=" << gpu.transfer_ms
        << std::endl;
}

int main(int argc, char** argv) {

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    std::cout
        << "GPU="
        << prop.name
        << std::endl;

    std::cout
        << "SEED="
        << SEED
        << std::endl;

    std::cout
        << "BLOCK=16x16=256 threads"
        << std::endl;

    // Run only one size if requested
    if (
        argc == 3 &&
        std::string(argv[1]) == "--size"
    ) {

        std::cout
            << "N,CPU_ms,GPU_kernel_ms,"
               "H2D_plus_D2H_ms,"
               "GPU_end_to_end_ms,"
               "Speedup,MaxAbsError"
            << std::endl;

        runTest(
            std::stoi(argv[2])
        );

        return 0;
    }

    // Used for profiler
    if (
        argc == 3 &&
        std::string(argv[1]) == "--gpu-only"
    ) {

        runGPUOnly(
            std::stoi(argv[2])
        );

        return 0;
    }

    std::cout
        << "N,CPU_ms,GPU_kernel_ms,"
           "H2D_plus_D2H_ms,"
           "GPU_end_to_end_ms,"
           "Speedup,MaxAbsError"
        << std::endl;

    for (int N : {256, 1024, 4096}) {
        runTest(N);
    }

    return 0;
}
