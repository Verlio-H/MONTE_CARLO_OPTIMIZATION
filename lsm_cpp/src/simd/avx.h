#include <cstdint>
#include <cstdlib>
#include "avx_mathfun.h"

constexpr size_t simd_width_float = 8;
constexpr size_t simd_width_int32 = 8;

using simd_float = __m256;
using simd_int32 = __m256i;
using simd_mask = __m256i;
constexpr simd_float (*simd_fexp)(simd_float) = exp256_ps;
constexpr simd_float (*simd_fbroadcast)(float val) = _mm256_set1_ps;
float simd_fget_lane(simd_float a, const int i) {
    float ret = 0;
    switch (i){
        case 0: {
            ret = _mm_cvtss_f32(_mm256_extractf128_ps(a, 0));
        } break;
        case 1: {
            __m128 lo = _mm256_extractf128_ps(a, 0);
            ret = _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, 1));
        } break;
        case 2: {
            __m128 lo = _mm256_extractf128_ps(a, 0);
            ret = _mm_cvtss_f32(_mm_movehl_ps(lo, lo));
        } break;
        case 3: {
            __m128 lo = _mm256_extractf128_ps(a, 0);                    
            ret = _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, 3));
        } break;
        case 4: {
            ret = _mm_cvtss_f32(_mm256_extractf128_ps(a, 1));
        } break;
        case 5: {
            __m128 hi = _mm256_extractf128_ps(a, 1);
            ret = _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 1));
        } break;
        case 6: {
            __m128 hi = _mm256_extractf128_ps(a, 1);
            ret = _mm_cvtss_f32(_mm_movehl_ps(hi, hi));
        } break;
        case 7: {
            __m128 hi = _mm256_extractf128_ps(a, 1);
            ret = _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 3));
        } break;
    }

    return ret;
}
constexpr simd_float (*simd_fadd)(simd_float a, simd_float b) = _mm256_add_ps;
constexpr simd_float (*simd_fsub)(simd_float a, simd_float b) = _mm256_sub_ps;
constexpr simd_float (*simd_fmul)(simd_float a, simd_float b) = _mm256_mul_ps;
constexpr simd_float (*simd_fmax)(simd_float a, simd_float b) = _mm256_max_ps;
simd_mask simd_fgt_mask(simd_float a, simd_float b) {
    return (simd_mask)_mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

#define extract _mm256_extract_epi32

float simd_fhadd(simd_float x) {
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

simd_float simd_fgather_int32(float const *base, simd_int32 indices) {
#ifdef __AVX2__
    return _mm256_i32gather_ps(base, indices, 4);
#else
    return _mm256_set_ps(*(base + extract(indices, 7)),
                         *(base + extract(indices, 6)),
                         *(base + extract(indices, 5)),
                         *(base + extract(indices, 4)),
                         *(base + extract(indices, 3)),
                         *(base + extract(indices, 2)),
                         *(base + extract(indices, 1)),
                         *(base + extract(indices, 0)));
#endif
}

simd_float simd_fgather_int32_masked(float const *base, simd_int32 indices, float def, simd_mask mask) {
#ifdef __AVX2__
    return _mm256_mask_i32gather_ps(simd_fbroadcast(def), base, indices, (__m256)mask, 4);
#else
    return _mm256_set_ps(extract(mask, 7) ? *(base + extract(indices, 7)) : def,
                         extract(mask, 6) ? *(base + extract(indices, 6)) : def,
                         extract(mask, 5) ? *(base + extract(indices, 5)) : def,
                         extract(mask, 4) ? *(base + extract(indices, 4)) : def,
                         extract(mask, 3) ? *(base + extract(indices, 3)) : def,
                         extract(mask, 2) ? *(base + extract(indices, 2)) : def,
                         extract(mask, 1) ? *(base + extract(indices, 1)) : def,
                         extract(mask, 0) ? *(base + extract(indices, 0)) : def);
#endif
}
void simd_fscatter_int32_masked(float *base, simd_int32 indices, simd_float values, simd_mask mask) {
    uint32_t lane0 = extract(mask, 0);
    if (lane0) *(base + extract(indices, 0)) = simd_fget_lane(values, 0);
    uint32_t lane1 = extract(mask, 1);
    if (lane1) *(base + extract(indices, 1)) = simd_fget_lane(values, 1);
    uint32_t lane2 = extract(mask, 2);
    if (lane2) *(base + extract(indices, 2)) = simd_fget_lane(values, 2);
    uint32_t lane3 = extract(mask, 3);
    if (lane3) *(base + extract(indices, 3)) = simd_fget_lane(values, 3);
    uint32_t lane4 = extract(mask, 4);
    if (lane4) *(base + extract(indices, 4)) = simd_fget_lane(values, 4);
    uint32_t lane5 = extract(mask, 5);
    if (lane5) *(base + extract(indices, 5)) = simd_fget_lane(values, 5);
    uint32_t lane6 = extract(mask, 6);
    if (lane6) *(base + extract(indices, 6)) = simd_fget_lane(values, 6);
    uint32_t lane7 = extract(mask, 7);
    if (lane7) *(base + extract(indices, 7)) = simd_fget_lane(values, 7);
}
void simd_iscatter_scalar_int32_masked(int *base, simd_int32 indices, int value, simd_mask mask) {
    uint32_t lane0 = extract(mask, 0);
    if (lane0) *(base + extract(indices, 0)) = value;
    uint32_t lane1 = extract(mask, 1);
    if (lane1) *(base + extract(indices, 1)) = value;
    uint32_t lane2 = extract(mask, 2);
    if (lane2) *(base + extract(indices, 2)) = value;
    uint32_t lane3 = extract(mask, 3);
    if (lane3) *(base + extract(indices, 3)) = value;
    uint32_t lane4 = extract(mask, 4);
    if (lane4) *(base + extract(indices, 4)) = value;
    uint32_t lane5 = extract(mask, 5);
    if (lane5) *(base + extract(indices, 5)) = value;
    uint32_t lane6 = extract(mask, 6);
    if (lane6) *(base + extract(indices, 6)) = value;
    uint32_t lane7 = extract(mask, 7);
    if (lane7) *(base + extract(indices, 7)) = value;
}


#define SIMD_INCREASING (_mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0))
#define SIMD_ITERATE_LANES(__LOOP__) \
    __LOOP__(0); \
    __LOOP__(1); \
    __LOOP__(2); \
    __LOOP__(3); \
    __LOOP__(4); \
    __LOOP__(5); \
    __LOOP__(6); \
    __LOOP__(7);

