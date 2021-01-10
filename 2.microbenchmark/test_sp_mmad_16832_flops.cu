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
static const uint64_t total_repeat_time = 409600000;
static const uint32_t total_thread_num = 128;
static const uint32_t warp_size = 32;
static const uint32_t sm_count = 108;
static const uint32_t block_num = sm_count * 1;
static const uint32_t warp_num = total_thread_num / warp_size;
static const uint32_t max_unroll_times = 8;

// I didn't check result since it is just a throughput test
template <uint32_t unroll_times>
__global__ void sp_mmad_16832_throughput_test(__half *d, __half *a, __half *b,
                                              __half *c, uint32_t *metadata_p,
                                              uint64_t repeat_time) {
  uint32_t tid = threadIdx.x % warp_size;

  size_t mat_a_stride = M * K / 2;
  size_t mat_b_stride = N * K;
  size_t mat_c_stride = M * N;
  size_t mat_d_stride = M * N;
  size_t metadata_stride = M * 2 / sizeof(uint32_t); // 16 bit per row

  uint32_t metadata[max_unroll_times];
  uint32_t a01[max_unroll_times], a23[max_unroll_times], a45[max_unroll_times],
      a67[max_unroll_times];
  uint32_t b01[max_unroll_times], b23[max_unroll_times], b45[max_unroll_times],
      b67[max_unroll_times];
  uint32_t d01[max_unroll_times], d23[max_unroll_times];

  size_t mat_a_row = K / 2;
  size_t mat_b_row = N;

  __half *a_base_ptr = a + (tid % 4) * 2 + (tid / 4) * mat_a_row;
  __half *b_base_ptr = b + (tid % 4) * 2 * mat_b_row + tid / 4;
  __half *c_base_ptr = c + (tid % 4) * 2 + (tid / 4) * 8;
  __half *d_base_ptr = d + (tid % 4) * 2 + (tid / 4) * 8;

  for (int i = 0; i < unroll_times; ++i) {
    uint32_t *metadata_ptr = metadata_p + i * metadata_stride;
    __half *a_ptr = a_base_ptr + i * mat_a_stride;
    __half *b_ptr = b_base_ptr + i * mat_b_stride;
    __half *c_ptr = c_base_ptr + i * mat_c_stride;

    metadata[i] = metadata_ptr[tid / 2];

    a01[i] = *((uint32_t *)a_ptr);
    a23[i] = *((uint32_t *)(a_ptr + 128));
    a45[i] = *((uint32_t *)(a_ptr + 8));
    a67[i] = *((uint32_t *)(a_ptr + 128 + 8));

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
                 : "=r"(b01[i]), "=r"(b23[i]), "=r"(b45[i]), "=r"(b67[i])
                 : "h"(b0), "h"(b1), "h"(b2), "h"(b3), "h"(b4), "h"(b5),
                   "h"(b6), "h"(b7)
                 :);

    d01[i] = *((uint32_t *)c_ptr);
    d23[i] = *((uint32_t *)(c_ptr + 64));
  }

  asm volatile("bar.sync 0;");

  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {
    for (int i = 0; i < unroll_times; ++i) {
      asm volatile("{\n\t"
                   "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16\n\t"
                   "{%0, %1},\n\t"
                   "{%2, %3, %4, %5},\n\t"
                   "{%6, %7, %8, %9},\n\t"
                   "{%10, %11}, %12, 0x0;\n\t"
                   "}\n\t"
                   : "=r"(d01[i]), "=r"(d23[i])
                   : "r"(a01[i]), "r"(a23[i]), "r"(a45[i]), "r"(a67[i]),
                     "r"(b01[i]), "r"(b23[i]), "r"(b45[i]), "r"(b67[i]),
                     "r"(d01[i]), "r"(d23[i]), "r"(metadata[i])
                   :);
    }
  }

  asm volatile("bar.sync 0;");

  for (int i = 0; i < unroll_times; ++i) {
    __half *d_ptr = d_base_ptr + i * mat_d_stride;
    *((uint32_t *)d_ptr) = d01[i];
    *((uint32_t *)(d_ptr + 64)) = d23[i];
  }
}

template <uint32_t unroll_times> void test_sp_mmad_16832_flops() {
  size_t mat_a_size = M * K / 2;
  size_t mat_b_size = N * K;
  size_t mat_c_size = M * N;
  size_t mat_d_size = M * N;
  size_t metadata_size_bytes = M * K / 8; // 32 bit per row

  size_t mat_a_unroll_size = mat_a_size * unroll_times;
  size_t mat_b_unroll_size = mat_b_size * unroll_times;
  size_t mat_c_unroll_size = mat_c_size * unroll_times;
  size_t mat_d_unroll_size = mat_d_size * unroll_times;
  size_t metadata_unroll_size_bytes = metadata_size_bytes * unroll_times;

  uint32_t *duration_host = new uint32_t[warp_num];

  __half *mat_a_host = new __half[mat_a_size];
  __half *mat_b_host = new __half[mat_b_size];
  __half *mat_c_host = new __half[mat_c_size];
  __half *mat_d_host = new __half[mat_d_size];
  uint32_t *metadata_host =
      new uint32_t[metadata_size_bytes / sizeof(uint32_t)];

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

  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_unroll_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_unroll_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_unroll_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_d_unroll_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&metadata_dev, metadata_unroll_size_bytes));

  for (uint32_t i = 0; i < unroll_times; ++i) {
    // uncomment to use random data, but perfomance may decrease due to power
    // limit
    /*
    gpuErrchk(cudaMemcpy(mat_a_dev + i * mat_a_size, mat_a_host,
                         mat_a_size * sizeof(__half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mat_b_dev + i * mat_b_size, mat_b_host,
                         mat_b_size * sizeof(__half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mat_c_dev + i * mat_c_size, mat_c_host,
                         mat_c_size * sizeof(__half), cudaMemcpyHostToDevice));
                         */
    gpuErrchk(
        cudaMemcpy(metadata_dev + i * metadata_size_bytes / sizeof(uint32_t),
                   metadata_host, metadata_size_bytes, cudaMemcpyHostToDevice));
  }

  uint64_t repeat_time = total_repeat_time / unroll_times;

  auto t_start = std::chrono::high_resolution_clock::now();

  sp_mmad_16832_throughput_test<unroll_times><<<block_num, total_thread_num>>>(
      mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, metadata_dev, repeat_time);
  gpuErrchk(cudaDeviceSynchronize());
  auto t_end = std::chrono::high_resolution_clock::now();
  double gpu_ns = (t_end - t_start).count();
  std::cout << "kernel duration: " << gpu_ns << " ns" << std::endl;

  double flop_per_repeat =
      unroll_times * warp_num * M * K * N * 2; // x2 because fma is 2ops
  double total_flop = flop_per_repeat * repeat_time * block_num;
  std::cout << "unroll: " << unroll_times
            << " flops(whole GPU): " << total_flop / (gpu_ns / 1E9) / 1E12
            << " TFLOPS" << std::endl;

  gpuErrchk(cudaFree(mat_a_dev));
  gpuErrchk(cudaFree(mat_b_dev));
  gpuErrchk(cudaFree(mat_c_dev));
  gpuErrchk(cudaFree(mat_d_dev));
  gpuErrchk(cudaFree(metadata_dev));

  delete[] mat_a_host;
  delete[] mat_b_host;
  delete[] mat_c_host;
  delete[] mat_d_host;
  delete[] metadata_host;
}

int main(int argc, char **argv) {
  test_sp_mmad_16832_flops<1>();
  test_sp_mmad_16832_flops<2>();
  test_sp_mmad_16832_flops<3>();
  test_sp_mmad_16832_flops<4>();
  test_sp_mmad_16832_flops<5>();
  test_sp_mmad_16832_flops<6>();
  test_sp_mmad_16832_flops<7>();
  test_sp_mmad_16832_flops<8>();
  return 0;
}