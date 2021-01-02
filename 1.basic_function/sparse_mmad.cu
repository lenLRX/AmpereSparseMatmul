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
static const int K = 16;

__global__ void sparse_mmad(__half *d, __half *a, __half *b, __half *c,
                            uint32_t *metadata_p) {
  uint32_t tid = threadIdx.x;
  uint32_t metadata = metadata_p[tid / 4];
  __half *a_ptr = a + (tid % 4) * 2 + (tid / 4) * 8;
  __half *b_ptr = b + (tid % 4) * 2 * 8 + tid / 4;
  __half *c_ptr = c + (tid % 4) * 2 + (tid / 4) * 8;
  __half *d_ptr = d + (tid % 4) * 2 + (tid / 4) * 8;
  asm volatile("{\n\t"
               ".reg .f16 %Ra_single<4>, %Rb_single<4>;\n\t"
               ".reg .f16x2 %Ra<2>, %Rb<2>, %Rc<2>, %Rd<2>;\n\t"
               "ld.global.ca.b32 %Ra0, [%1];\n\t"
               "ld.global.ca.b32 %Ra1, [%1 + 128];\n\t"
               "ld.global.ca.b16 %Rb_single0, [%2];\n\t"
               "ld.global.ca.b16 %Rb_single1, [%2 + 16];\n\t"
               "ld.global.ca.b16 %Rb_single2, [%2 + 128];\n\t"
               "ld.global.ca.b16 %Rb_single3, [%2 + 144];\n\t"
               "ld.global.ca.b32 %Rc0, [%3];\n\t"
               "ld.global.ca.b32 %Rc1, [%3 + 128];\n\t"
               "mov.b32 %Rb0, {%Rb_single0, %Rb_single1};\n\t"
               "mov.b32 %Rb1, {%Rb_single2, %Rb_single3};\n\t"
               "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\n\t"
               "{%Rd0, %Rd1},\n\t"
               "{%Ra0, %Ra1},\n\t"
               "{%Rb0, %Rb1},\n\t"
               "{%Rc0, %Rc1}, %4, 0x0;\n\t"
               "st.global.wb.b32 [%0], %Rd0;\n\t"
               "st.global.wb.b32 [%0 + 128], %Rd1;\n\t"
               "}\n\t"
               :
               : "l"(d_ptr), "l"(a_ptr), "l"(b_ptr), "l"(c_ptr), "r"(metadata)
               : "memory");
}

int main(int argc, char **argv) {
  size_t mat_a_size = M * K / 2;
  size_t mat_b_size = N * K;
  size_t mat_c_d_size = M * N;
  size_t metadata_size_bytes = M * 2; // 16 bit per row

  __half *mat_a_host = new __half[mat_a_size];
  __half *mat_b_host = new __half[mat_b_size];
  __half *mat_c_host = new __half[mat_c_d_size];
  __half *mat_d_host = new __half[mat_c_d_size];
  uint32_t *metadata_host =
      new uint32_t[metadata_size_bytes / sizeof(uint32_t)];
  std::ifstream a_fs("a.bin", std::ios_base::binary);
  a_fs.read((char *)mat_a_host, mat_a_size * sizeof(__half));
  std::ifstream b_fs("b.bin", std::ios_base::binary);
  b_fs.read((char *)mat_b_host, mat_b_size * sizeof(__half));
  std::ifstream c_fs("c.bin", std::ios_base::binary);
  c_fs.read((char *)mat_c_host, mat_c_d_size * sizeof(__half));
  std::ifstream metadata_fs("metadata.bin", std::ios_base::binary);
  metadata_fs.read((char *)metadata_host, metadata_size_bytes);

  __half *mat_a_dev;
  __half *mat_b_dev;
  __half *mat_c_dev;
  __half *mat_d_dev;
  uint32_t *metadata_dev;

  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_d_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_c_d_size * sizeof(__half)));
  gpuErrchk(cudaMalloc(&metadata_dev, metadata_size_bytes));

  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__half),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__half),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host, mat_c_d_size * sizeof(__half),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(metadata_dev, metadata_host, metadata_size_bytes,
                       cudaMemcpyHostToDevice));

  sparse_mmad<<<1, 32>>>(mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev,
                         metadata_dev);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(mat_d_host, mat_d_dev, mat_c_d_size * sizeof(__half),
                       cudaMemcpyDeviceToHost));
  std::ofstream d_fs("d_gpu.bin", std::ios_base::binary);
  d_fs.write((char *)mat_d_host, mat_c_d_size * sizeof(__half));

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

  return 0;
}