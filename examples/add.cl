// CUDA
#ifdef __CUDACC__
  #define GLOBAL
  #define KERNEL extern "C" __global__
// OpenCL
#else
  #define GLOBAL __global
  #define KERNEL __kernel
#endif

KERNEL void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}