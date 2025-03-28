#include <metal_stdlib>
using namespace metal;

kernel void add_vectors(device const float *inA,
                         device const float *inB,
                         device float *result,
                         uint index [[thread_position_in_grid]])
{
    // Add the corresponding elements of both vectors
    result[index] = inA[index] + inB[index];
}