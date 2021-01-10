#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

// code from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

static const int M = 16;
static const int N = 8;
static const int K = 32;
static const uint64_t repeat_time = 409600;
static const uint32_t total_thread_num = 32;
static const uint32_t warp_size = 32;
static const uint32_t block_num = 1;
static const uint32_t warp_num = total_thread_num / warp_size;

// TODO: use shared memory

__global__ void sp_mmad_16832_latency_test(__half *d, __half *a, __half *b,
                                           __half *c, uint32_t *metadata_p,
                                           uint32_t *duration) {
  uint32_t tid = threadIdx.x % warp_size;

  uint32_t metadata;
  uint32_t a01, a23, a45, a67;
  uint32_t b01, b23, b45, b67;
  uint32_t d01, d23;

  size_t mat_a_row = K / 2;
  size_t mat_b_row = N;

  __half *a_base_ptr = a + (tid % 4) * 2 + (tid / 4) * mat_a_row;
  __half *b_base_ptr = b + (tid % 4) * 2 * mat_b_row + tid / 4;
  __half *c_base_ptr = c + (tid % 4) * 2 + (tid / 4) * 8;
  __half *d_base_ptr = d + (tid % 4) * 2 + (tid / 4) * 8;

  uint32_t *metadata_ptr = metadata_p;
  __half *a_ptr = a_base_ptr;
  __half *b_ptr = b_base_ptr;
  __half *c_ptr = c_base_ptr;

  metadata = metadata_ptr[tid / 2];

  a01 = *((uint32_t *)a_ptr);
  a23 = *((uint32_t *)(a_ptr + 8));
  a45 = *((uint32_t *)(a_ptr + 128));
  a67 = *((uint32_t *)(a_ptr + 128 + 8));

  uint16_t b0, b1, b2, b3, b4, b5, b6, b7;

  b0 = *((uint16_t *)(b_ptr));
  b1 = *((uint16_t *)(b_ptr + 8));
  b2 = *((uint16_t *)(b_ptr + 64));
  b3 = *((uint16_t *)(b_ptr + 64 + 8));
  b4 = *((uint16_t *)(b_ptr + 64 * 2));
  b5 = *((uint16_t *)(b_ptr + 64 * 2 + 8));
  b6 = *((uint16_t *)(b_ptr + 64 * 3));
  b7 = *((uint16_t *)(b_ptr + 64 * 3 + 8));

  asm volatile("mov.b32 %0, {%4, %5};\n\t"
               "mov.b32 %1, {%6, %7};\n\t"
               "mov.b32 %2, {%8, %9};\n\t"
               "mov.b32 %3, {%10, %11};\n\t"
               : "=r"(b01), "=r"(b23), "=r"(b45), "=r"(b67)
               : "h"(b0), "h"(b1), "h"(b2), "h"(b3), "h"(b4), "h"(b5), "h"(b6),
                 "h"(b7)
               :);

  d01 = *((uint32_t *)c_ptr);
  d23 = *((uint32_t *)(c_ptr + 64));

  asm volatile("bar.sync 0;");

  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  #pragma unroll 1
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {
    asm volatile("{\n\t"
                 "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16\n\t"
                 "{%0, %1},\n\t"
                 "{%2, %3, %4, %5},\n\t"
                 "{%6, %7, %8, %9},\n\t"
                 "{%10, %11}, %12, 0x0;\n\t"
                 "}\n\t"
                 : "=r"(d01), "=r"(d23)
                 : "r"(a01), "r"(a23), "r"(a45), "r"(a67), "r"(b01), "r"(b23),
                   "r"(b45), "r"(b67), "r"(d01), "r"(d23), "r"(metadata)
                 :);
  }

  asm volatile("bar.sync 0;");

  uint32_t end = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(end)::"memory");

  __half *d_ptr = d_base_ptr;
  *((uint32_t *)d_ptr) = d01;
  *((uint32_t *)(d_ptr + 64)) = d23;

  uint32_t u32_duration = end - start;
  if (tid == 0) {
    duration[0] = u32_duration;
  }
}

int main(int argc, char **argv) {
  size_t mat_a_size = M * K / 2;
  size_t mat_b_size = N * K;
  size_t mat_c_size = M * N;
  size_t mat_d_size = M * N;
  size_t metadata_size_bytes = M * 2; // 16 bit per row

  __half *mat_a_host = new __half[mat_a_size];
  __half *mat_b_host = new __half[mat_b_size];
  __half *mat_c_host = new __half[mat_c_size];
  __half *mat_d_host = new __half[mat_d_size];
  uint32_t *metadata_host =
      new uint32_t[metadata_size_bytes / sizeof(uint32_t)];
  uint32_t *duration_host = new uint32_t[warp_num];

  std::ifstream a_fs("a.bin", std::ios_base::binary);
  a_fs.read((char *)mat_a_host, mat_a_size * sizeof(__half));
  std::ifstream b_fs("b.bin", std::ios_base::binary);
  b_fs.read((char *)mat_b_host, mat_b_size * sizeof(__half));
  std::ifstream c_fs("c.bin", std::ios_base::binary);
  c_fs.read((char *)mat_c_host, mat_c_size * sizeof(__half));
  std::ifstream metadata_fs("metadata.bin", std::ios_base::binary);
  metadata_fs.read((char *)metadata_host, metadata_size_bytes);

  __half *mat_a_dev;
  __half *mat_b_dev;
  __half *mat_c_dev;
  __half *mat_d_dev;
  uint32_t *metadata_dev;
  uint32_t *duration_dev;

  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_d_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&metadata_dev, metadata_size_bytes));
  gpuErrchk(cudaMalloc(&duration_dev, warp_num * sizeof(uint32_t)));

  // uncomment to use random data, but perfomance may decrease due to power
  // limit
  /*
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host,
                       mat_a_size * sizeof(__half), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host,
                       mat_b_size * sizeof(__half), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host,
                       mat_c_size * sizeof(__half), cudaMemcpyHostToDevice));
                       */
  gpuErrchk(cudaMemcpy(metadata_dev,
                       metadata_host, metadata_size_bytes,
                       cudaMemcpyHostToDevice));

  sp_mmad_16832_latency_test<<<block_num, total_thread_num>>>(
      mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, metadata_dev, duration_dev);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(duration_host, duration_dev, warp_num * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));

  double total_cycle = duration_host[0];

  std::cout << "mma.sp.sync.aligned.m16n8k32 latency: "
            << total_cycle / repeat_time << " cycle" << std::endl;

  gpuErrchk(cudaFree(mat_a_dev));
  gpuErrchk(cudaFree(mat_b_dev));
  gpuErrchk(cudaFree(mat_c_dev));
  gpuErrchk(cudaFree(mat_d_dev));
  gpuErrchk(cudaFree(metadata_dev));
  gpuErrchk(cudaFree(duration_dev));

  delete[] mat_a_host;
  delete[] mat_b_host;
  delete[] mat_c_host;
  delete[] mat_d_host;
  delete[] metadata_host;
  delete[] duration_host;

  return 0;
}