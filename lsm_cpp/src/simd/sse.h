
#pragma once

#include <cstdint>
#include <cstdlib>
#include "sse_mathfun.h"

constexpr size_t simd_width_float = 4;
constexpr size_t simd_width_int32 = 4;

using simd_float = __m128;
using simd_int32 = __m128i;
using simd_mask = __m128i;
constexpr simd_float (*simd_fexp)(simd_float) = sse_mathfun_exp_ps;
constexpr simd_float (*simd_fbroadcast)(float val) = _mm_set_ps1;
#ifdef __SSE_4_1__
    #define simd_fget_lane(val, lane) _mm_extract_epi32((val), (lane))
#else
    #define simd_fget_lane(val, lane) ((__m128)_mm_cvtsi128_si32(_mm_shuffle_epi32((__m128i)(val), (lane)*0x55)
#endif
constexpr simd_float (*simd_fadd)(simd_float a, simd_float b) = _mm_add_ps;
constexpr simd_float (*simd_fsub)(simd_float a, simd_float b) = _mm_sub_ps;
constexpr simd_float (*simd_fmul)(simd_float a, simd_float b) = _mm_mul_ps;
constexpr simd_float (*simd_fmax)(simd_float a, simd_float b) = _mm_max_ps;
simd_mask simd_fgt_mask(simd_float a, simd_float b) {
    return (simd_mask)_mm_cmpgt_ps(a, b);
}


inline int get_x(__m128i vec) { return _mm_cvtsi128_si32(vec); }

#ifdef __SSE4_1__
    inline int get_y(__m128i vec) { return _mm_extract_epi32(vec, 1); }
    inline int get_z(__m128i vec) { return _mm_extract_epi32(vec, 2); }
    inline int get_w(__m128i vec) { return _mm_extract_epi32(vec, 3); }
#else
    inline int get_y(__m128i vec) { return _mm_cvtsi128_si32(_mm_shuffle_epi32(vec, 0x55)); }
    inline int get_z(__m128i vec) { return _mm_cvtsi128_si32(_mm_shuffle_epi32(vec, 0xAA)); }
    inline int get_w(__m128i vec) { return _mm_cvtsi128_si32(_mm_shuffle_epi32(vec, 0xFF)); }
#endif


#ifdef __SSE3__
    #define MOVEHDUP _mm_movehdup_ps
#else
    inline __m128 MOVEHDUP(__m128 value) { return _mm_shuffle_ps(value, value, _MM_SHUFFLE(3,3, 1,1)); }
#endif

float simd_fhadd(simd_float val) {
    __m128 shuf = MOVEHDUP(val);
    __m128 sums = _mm_add_ps(val, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

simd_float simd_fgather_int32(float const *base, simd_int32 indices) {
    return _mm_set_ps(*(base + get_x(indices)),
                      *(base + get_y(indices)),
                      *(base + get_z(indices)),
                      *(base + get_w(indices)));
}

simd_float simd_fgather_int32_masked(float const *base, simd_int32 indices, float def, simd_mask mask) {
    return _mm_set_ps(get_x(mask) ? *(base + get_x(indices)) : def,
                      get_y(mask) ? *(base + get_y(indices)) : def,
                      get_z(mask) ? *(base + get_z(indices)) : def,
                      get_w(mask) ? *(base + get_w(indices)) : def);
}
void simd_fscatter_int32_masked(float *base, simd_int32 indices, simd_float values, simd_mask mask) {
    uint32_t lane0 = get_x(mask);
    if (lane0) *(base + get_x(indices)) = get_x((__m128i)values);
    uint32_t lane1 = get_y(mask);
    if (lane1) *(base + get_y(indices)) = get_y((__m128i)values);
    uint32_t lane2 = get_z(mask);
    if (lane2) *(base + get_z(indices)) = get_z((__m128i)values);
    uint32_t lane3 = get_w(mask);
    if (lane3) *(base + get_w(indices)) = get_w((__m128i)values);
}
void simd_iscatter_scalar_int32_masked(int *base, simd_int32 indices, int value, simd_mask mask) {
    uint32_t lane0 = get_x(mask);
    if (lane0) *(base + get_x(indices)) = value;
    uint32_t lane1 = get_y(mask);
    if (lane1) *(base + get_y(indices)) = value;
    uint32_t lane2 = get_z(mask);
    if (lane2) *(base + get_z(indices)) = value;
    uint32_t lane3 = get_w(mask);
    if (lane3) *(base + get_w(indices)) = value;
}


#define SIMD_INCREASING (_mm_set_ps(0, 1, 2, 3))
#define SIMD_ITERATE_LANES(__LOOP__) \
    __LOOP__(0); \
    __LOOP__(1); \
    __LOOP__(2); \
    __LOOP__(3);

