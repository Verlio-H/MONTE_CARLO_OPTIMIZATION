#include <cstdint>
#include <cstdlib>
#include "avx512_mathfun.h"

constexpr size_t simd_width_float = 16;
constexpr size_t simd_width_int32 = 16;

using simd_float = __m512;
using simd_int32 = __m512i;
using simd_mask = __mmask16;
constexpr simd_float (*simd_fexp)(simd_float) = exp_ps;
constexpr simd_float (*simd_fbroadcast)(float val) = _mm512_set1_ps;
float simd_fget_lane(simd_float v, const int i) {
     __m512i shuf = _mm512_castsi128_si512(_mm_cvtsi32_si128(i));
    return _mm512_cvtss_f32(_mm512_permutexvar_ps(shuf, v));
}
constexpr simd_float (*simd_fadd)(simd_float a, simd_float b) = _mm512_add_ps;
constexpr simd_float (*simd_fsub)(simd_float a, simd_float b) = _mm512_sub_ps;
constexpr simd_float (*simd_fmul)(simd_float a, simd_float b) = _mm512_mul_ps;
constexpr simd_float (*simd_fmax)(simd_float a, simd_float b) = _mm512_max_ps;
simd_mask simd_fgt_mask(simd_float a, simd_float b) {
    return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}

constexpr float (*simd_fhadd)(simd_float x) = _mm512_reduce_add_ps;

simd_float simd_fgather_int32(float const *base, simd_int32 indices) {
    return _mm512_i32gather_ps(indices, base, 4);
}

simd_float simd_fgather_int32_masked(float const *base, simd_int32 indices, float def, simd_mask mask) {
    return _mm512_mask_i32gather_ps(simd_fbroadcast(def), mask, indices, (__m512 *)base, 4);
}

void simd_fscatter_int32_masked(float *base, simd_int32 indices, simd_float values, simd_mask mask) {
    _mm512_mask_i32scatter_ps((__m512 *)base, mask, indices, values, 4);
}
void simd_iscatter_scalar_int32_masked(int *base, simd_int32 indices, int value, simd_mask mask) {
    _mm512_mask_i32scatter_epi32((__m512 *)base, mask, indices, _mm512_set1_epi32(value), 4);
}

#define SIMD_INCREASING (_mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define SIMD_ITERATE_LANES(__LOOP__) \
    __LOOP__(0); \
    __LOOP__(1); \
    __LOOP__(2); \
    __LOOP__(3); \
    __LOOP__(4); \
    __LOOP__(5); \
    __LOOP__(6); \
    __LOOP__(7); \
    __LOOP__(8); \
    __LOOP__(9); \
    __LOOP__(10); \
    __LOOP__(11); \
    __LOOP__(12); \
    __LOOP__(13); \
    __LOOP__(14); \
    __LOOP__(15);


