#pragma once

#include <cstdint>
#include <cstdlib>
#include "neon_mathfun.h"

constexpr size_t simd_width_float = 4;
constexpr size_t simd_width_int32 = 4;

using simd_float = float32x4_t;
using simd_int32 = int32x4_t;
using simd_mask = uint32x4_t;
constexpr simd_float (*simd_fexp)(simd_float) = exp_ps;
constexpr simd_float (*simd_fbroadcast)(float val) = vdupq_n_f32;
#define simd_fget_lane(val, lane) vgetq_lane_f32(val, lane)
constexpr simd_float (*simd_fadd)(simd_float a, simd_float b) = vaddq_f32;
constexpr simd_float (*simd_fsub)(simd_float a, simd_float b) = vsubq_f32;
constexpr simd_float (*simd_fmul)(simd_float a, simd_float b) = vmulq_f32;
constexpr simd_float (*simd_fmax)(simd_float a, simd_float b) = vmaxq_f32;
constexpr simd_mask (*simd_fgt_mask)(simd_float a, simd_float b) = vcgtq_f32;
constexpr float (*simd_fhadd)(simd_float val) = vaddvq_f32;
simd_float simd_fgather_int32(float const *base, simd_int32 indices) {
    simd_float result = vdupq_n_f32(0);
    result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 0), result, 0);
    result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 1), result, 1);
    result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 2), result, 2);
    result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 3), result, 3);
    return result;
}
simd_float simd_fgather_int32_masked(float const *base, simd_int32 indices, float def, simd_mask mask) {
    simd_float result = vdupq_n_f32(def);
    uint32_t lane0 = vgetq_lane_u32(mask, 0);
    if (lane0) result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 0), result, 0);
    uint32_t lane1 = vgetq_lane_u32(mask, 1);
    if (lane1) result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 1), result, 1);
    uint32_t lane2 = vgetq_lane_u32(mask, 2);
    if (lane2) result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 2), result, 2);
    uint32_t lane3 = vgetq_lane_u32(mask, 3);
    if (lane3) result = vld1q_lane_f32(base + vgetq_lane_s32(indices, 3), result, 3);
    return result;
}
void simd_fscatter_int32_masked(float *base, simd_int32 indices, simd_float values, simd_mask mask) {
    uint32_t lane0 = vgetq_lane_u32(mask, 0);
    if (lane0) *(base + vgetq_lane_s32(indices, 0)) = vgetq_lane_f32(values, 0);
    uint32_t lane1 = vgetq_lane_u32(mask, 1);
    if (lane1) *(base + vgetq_lane_s32(indices, 1)) = vgetq_lane_f32(values, 1);
    uint32_t lane2 = vgetq_lane_u32(mask, 2);
    if (lane2) *(base + vgetq_lane_s32(indices, 2)) = vgetq_lane_f32(values, 2);
    uint32_t lane3 = vgetq_lane_u32(mask, 3);
    if (lane3) *(base + vgetq_lane_s32(indices, 3)) = vgetq_lane_f32(values, 3);
}
void simd_iscatter_scalar_int32_masked(int *base, simd_int32 indices, int value, simd_mask mask) {
    uint32_t lane0 = vgetq_lane_u32(mask, 0);
    if (lane0) *(base + vgetq_lane_s32(indices, 0)) = value;
    uint32_t lane1 = vgetq_lane_u32(mask, 1);
    if (lane1) *(base + vgetq_lane_s32(indices, 1)) = value;
    uint32_t lane2 = vgetq_lane_u32(mask, 2);
    if (lane2) *(base + vgetq_lane_s32(indices, 2)) = value;
    uint32_t lane3 = vgetq_lane_u32(mask, 3);
    if (lane3) *(base + vgetq_lane_s32(indices, 3)) = value;
    
}


#define SIMD_INCREASING (vsetq_lane_f32(3, vsetq_lane_f32(2, vsetq_lane_f32(1, vdupq_n_f32(0), 3), 2), 1))
#define SIMD_ITERATE_LANES(__LOOP__) \
    __LOOP__(0); \
    __LOOP__(1); \
    __LOOP__(2); \
    __LOOP__(3);
