#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <numeric>

constexpr size_t simd_width_float = 1;
constexpr size_t simd_width_int32 = 4;

using simd_float = float;
using simd_int32 = int32_t;
using simd_mask = bool;
constexpr simd_float (*simd_fexp)(simd_float) = expf;
simd_float simd_fbroadcast(float val) { return val; }
#define simd_fget_lane(val, lane) val
simd_float simd_fadd(simd_float a, simd_float b) { return a + b; }
simd_float simd_fsub(simd_float a, simd_float b) { return a - b; }
simd_float simd_fmul(simd_float a, simd_float b) { return a * b; }
simd_float simd_fmax(simd_float a, simd_float b) { return std::max(a, b); }
simd_mask simd_fgt_mask(simd_float a, simd_float b) { return a > b; }
float simd_fhadd(simd_float val) { return val; }
simd_float simd_fgather_int32(float const *base, simd_int32 indices) { return *(base + indices); }
simd_float simd_fgather_int32_masked(float const *base, simd_int32 indices, int value, simd_mask mask) {
    if (mask) return *(base + indices);
    return value; 
}
void simd_fscatter_int32_masked(float *base, simd_int32 indices, simd_float values, simd_mask mask) {
    if (mask) *(base + indices) = values;
}
void simd_iscatter_scalar_int32_masked(int *base, simd_int32 indices, int value, simd_mask mask) {
    if (mask) *(base + indices) = value;
}

#define SIMD_INCREASING (0)
#define SIMD_ITERATE_LANES(__LOOP__) __LOOP__(0);
