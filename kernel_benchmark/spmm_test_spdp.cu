/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "SpMM_API.cuh"
#include "./Flashllm_utils.cuh"


int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("Wrong Inputs! Correct input format: ./spmm_test M K N Sparsity SplitK\n");
        return;
    }
    int M_GLOBAL                    = atoi(argv[1]);
    int K_GLOBAL                    = atoi(argv[2]);
    int N_GLOBAL                    = atoi(argv[3]);
    int MATRIX_A_PRUNING_PERCENTAGE = atoi(argv[4]);
    int SPLIT_K                     = atoi(argv[5]);
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major
    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    //
    A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (N_GLOBAL != 1){
        printf("This test is for SpMV kernel!");
        exit(-1);
    }
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    //
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    //
    printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
 
// CUBLAS
/////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuBlas...\n");
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    // Tensor core not enabled
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cudaDeviceSynchronize();
    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;
    float tflops_cublas =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.))
        / 1e12;
    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    milliseconds_cublas_tc = milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                 / (milliseconds_cublas_tc / 1000.))
                             / 1e12;
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);
/////////////////////////////////////////////////////////////////////////////////////////////////

    auto Split_K = SPLIT_K;

// Spdp
////////////////////////////////////////////////////////////////////////////////////////////////
    half* D_SpMM_bitmapv3 = NULL;
    // ### cudaError_t cudaMalloc(void** devPtr, size_t size);
    // - devPtr : ptr to point the GPU memory
    // - size : byte
    // - return value : 0 or error code, NULL
    // reinterpret_cast : for type casting without warning
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM_bitmapv3 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    // ### cudaError_t cudaMemset(void* devPtr, int value, size_t count);
    // - devPtr : ptr to initialize
    // - value : initialization value(only one byte, masked by 0xff)
    // - count : number of bytes to fill
    cudaMemset(D_SpMM_bitmapv3, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

    // Define the output pointer
    // - value
    half* Compressed_Val_cpu_v3 = nullptr;
    // - GTileOffset
    int* bitmap_TileOffsets_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_median_cpu_v3 = nullptr;
    int* bitmap_TileOffsets_global_cpu_v3 = nullptr;
    // - bitmap
    uint64_t* bitmap_cpu_v3 = nullptr;
    int max_nnz_intilev3 = 0;
    // Call the InitSparseMatrixA_bitmap_v6 function
    auto num_gtilesv3 = InitSparseMatrixA_bitmap_v6(A_h, M_GLOBAL, K_GLOBAL, 8, 16, 64, 8, 64, 64, &Compressed_Val_cpu_v3, &bitmap_TileOffsets_cpu_v3, &bitmap_TileOffsets_median_cpu_v3, &bitmap_TileOffsets_global_cpu_v3, &bitmap_cpu_v3, max_nnz_intilev3);
    // - auto : compiler determines the type
    auto local_tile_numv3 = 8*8;
    auto median_tile_numv3 = 4*1;
    auto num_ltilesv3 = num_gtilesv3*local_tile_numv3;
    auto num_mtilesv3 = num_gtilesv3*median_tile_numv3;
    // The offset of the last tile is equal to the total number of compressed non-zero values
    // - val_count_median_v3=0 always => condition of spmm_utils.h:521 is always false?
    int val_count_v3 = bitmap_TileOffsets_global_cpu_v3[num_gtilesv3]; 
    int val_count_median_v3 = bitmap_TileOffsets_median_cpu_v3[num_mtilesv3];
    // Adjust max_nnz_intilev3 to a multiple of 64
    if (max_nnz_intilev3 % 64 != 0) {
        max_nnz_intilev3 = ((max_nnz_intilev3 / 64) + 1) * 64;
    }
    printf("num_global_tiles: %d, bitmap v3 NNZ: %d, bitmap v3 median layer NNZ: %d,  max_nnz_intilev3: %d \n", num_gtilesv3, val_count_v3, val_count_median_v3, max_nnz_intilev3);
    half* Compressed_Val_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_median_gpu_v3 = nullptr;
    int* bitmap_TileOffsets_global_gpu_v3 = nullptr;
    uint64_t* bitmap_gpu_v3 = nullptr;
    cudaMalloc(&bitmap_TileOffsets_gpu_v3, sizeof(int) * (num_ltilesv3 + 1)); // for (16*64 tile specific)
    cudaMalloc(&bitmap_gpu_v3, sizeof(uint64_t) * (num_ltilesv3));
    cudaMalloc(&bitmap_TileOffsets_median_gpu_v3, sizeof(int) * (num_mtilesv3));
    cudaMalloc(&bitmap_TileOffsets_global_gpu_v3, sizeof(int) * (num_gtilesv3+1));
    if (val_count_v3 == 0)
         val_count_v3 = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&Compressed_Val_gpu_v3, sizeof(half) * val_count_v3);
    if (bitmap_TileOffsets_gpu_v3 == NULL || bitmap_gpu_v3 == NULL || Compressed_Val_gpu_v3 == NULL || bitmap_TileOffsets_global_gpu_v3 == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    // ### cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

    cudaMemcpy(bitmap_TileOffsets_gpu_v3, bitmap_TileOffsets_cpu_v3, sizeof(int) * (num_ltilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_global_gpu_v3, bitmap_TileOffsets_global_cpu_v3, sizeof(int) * (num_gtilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_median_gpu_v3, bitmap_TileOffsets_median_cpu_v3, sizeof(int) * (num_mtilesv3), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_gpu_v3, bitmap_cpu_v3, sizeof(uint64_t) * num_ltilesv3, cudaMemcpyHostToDevice);
    cudaMemcpy(Compressed_Val_gpu_v3, Compressed_Val_cpu_v3, sizeof(half) * val_count_v3, cudaMemcpyHostToDevice);
    free(bitmap_TileOffsets_cpu_v3);
    free(bitmap_cpu_v3);
    free(Compressed_Val_cpu_v3);
    free(bitmap_TileOffsets_global_cpu_v3);
    free(bitmap_TileOffsets_median_cpu_v3);
    printf("Done! Compressed A matrix for bitmap v3 GPU kernel.\n");
    
    printf("Launching bitmapv3 without Ahead of Time Sparse Data Reordering...\n");
    Split_K = SPLIT_K;
    printf("Split_K = %d\n", Split_K);
    half* Reduction_Workspace_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace_bitmapv3 == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    int* max_nnz_intilev3_gpu = nullptr;
    cudaMalloc(&max_nnz_intilev3_gpu, sizeof(int));
    if (max_nnz_intilev3_gpu == NULL) {
        printf("Error in cudaMalloc for max_nnz_intilev3_gpu\n");
        exit(-1);
    }
    cudaMemcpy(max_nnz_intilev3_gpu, &max_nnz_intilev3, sizeof(int), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < WARM_UP_ITERATION; i++)
        SpMV_SplitK_API_bitmap_v3(0,
                        A,
                        Compressed_Val_gpu_v3, // half
                        bitmap_TileOffsets_global_gpu_v3, // int
                        bitmap_TileOffsets_median_gpu_v3, // int
                        bitmap_gpu_v3, //uint64
                        max_nnz_intilev3_gpu, // int
                        B,
                        D_SpMM_bitmapv3,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace_bitmapv3,   //only malloced
                        Split_K);
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        SpMV_SplitK_API_bitmap_v3(0,
                        A,
                        Compressed_Val_gpu_v3,
                        bitmap_TileOffsets_global_gpu_v3,
                        bitmap_TileOffsets_median_gpu_v3,
                        bitmap_gpu_v3,
                        max_nnz_intilev3_gpu,
                        B,
                        D_SpMM_bitmapv3,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace_bitmapv3,
                        Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    // //
    float milliseconds_SpMM_bitmapv3 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM_bitmapv3, start, stop);
    milliseconds_SpMM_bitmapv3 = milliseconds_SpMM_bitmapv3 / BENCHMARK_ITERATION;
    float tflops_SpMM_bitmapv3 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM_bitmapv3 / 1000.))
        / 1e12;
    half* D_SpMM_hbitmapv3 = NULL;  // col major
    D_SpMM_hbitmapv3       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_hbitmapv3, D_SpMM_bitmapv3, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_SpMM_bitmapv3);
    cudaFree(bitmap_TileOffsets_gpu_v3);
    cudaFree(bitmap_TileOffsets_global_gpu_v3);
    cudaFree(bitmap_TileOffsets_median_gpu_v3);
    cudaFree(bitmap_gpu_v3);
    cudaFree(Compressed_Val_gpu_v3);
    cudaFree(Reduction_Workspace_bitmapv3);
    cudaFree(max_nnz_intilev3_gpu);
    /////////////////////////////////////////////////////////////////////////////////////////////////


    double totalError_SpMM2 = 0.0;
    double totalError_SpMM_bitmapv3 = 0.0;

    // totalError_SpMM2 = ComputeTotalError(D_cublas_h, D_SpMM_h2, M_GLOBAL, N_GLOBAL);
    totalError_SpMM_bitmapv3 = ComputeTotalError(D_cublas_h, D_SpMM_hbitmapv3, M_GLOBAL, N_GLOBAL);
    
    // free(D_SpMM_h2);
    free(D_SpMM_hbitmapv3);
    // PrintPerformance("FlashLLM_v1", milliseconds_SpMM2, tflops_SpMM2, totalError_SpMM2);
    PrintPerformance("SpInfer", milliseconds_SpMM_bitmapv3, tflops_SpMM_bitmapv3, totalError_SpMM_bitmapv3);
    PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);

    free(D_cublas_h);
    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
    SavePerformanceData_spdp("Preliminary_result.csv",
        M_GLOBAL, K_GLOBAL, N_GLOBAL, 
        SPLIT_K, MATRIX_A_PRUNING_PERCENTAGE,
        milliseconds_cublas_tc, tflops_cublas_tc,
        // milliseconds_SpMM2, tflops_SpMM2,
        milliseconds_SpMM_bitmapv3, tflops_SpMM_bitmapv3);
    return 0;
}
