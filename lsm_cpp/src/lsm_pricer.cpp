// lsm_cpp/src/lsm_pricer.cpp

#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#if defined(__aarch64__)
    #ifdef TARGET_OS_MAC
    #if TARGET_OS_MAC
        #define USE_APPLE_AMX
        #include "simd/amx.h"
    #endif
    #endif

    #define USE_ARM_NEON
    #include "simd/neon.h"
#elif defined(__SSE__)
    #define USE_INTEL_SSE
    #include "simd/sse.h"
#else
    #define USE_FALLBACK
    #include "simd/fallback.h"
#endif



struct vec3 {
    union {
        double vals[3];
        struct {
            union {
                double x;
                double r;
                double c0;
            };
            union {
                double y;
                double g;
                double c1;
            };
            union {
                double z;
                double b;
                double c2;
            };
        };
    };
};

// Simple least-squares polynomial fit (degree 2)
vec3 polyfit(const float *x, const float *y, size_t n) {
    double S_x = 0, S_y = 0, S_xx = 0, S_xy = 0, S_xxx = 0, S_xxy = 0, S_xxxx = 0;

    
    size_t i = 0;
    #ifdef USE_APPLE_AMX
        amx_begin();
        // accumulate onto singular one so that 1024 bit ops can be used
        constexpr amx_reg s_x = 0;
        constexpr amx_reg s_y = 4;
        constexpr amx_reg s_xx = 8;
        constexpr amx_reg s_xy = 12;
        constexpr amx_reg s_xxx = 16;
        constexpr amx_reg s_xxy = 20;
        constexpr amx_reg s_xxxx = 24;
        for (; i <= n - 32; i += 32) {
            constexpr amx_reg xi = 0;
            constexpr amx_reg xi_y = 4;
            constexpr amx_reg xi_2_y = 4;
            constexpr amx_reg yi = 0;
            constexpr amx_reg xi_2_z = 32;
            constexpr amx_reg xi_2_x = 4;
            //xi = x[i];
            amx_load1024_x(xi, (void *)(x + i));
            //yi = y[i];
            amx_load1024_y(yi, (void *)(y + i));
            amx_movyx(xi_y + 0, xi + 0);
            amx_movyx(xi_y + 1, xi + 1);
            //S_xy += xi * yi;
            amx_vfma32(s_xy + 0, xi + 0, yi + 0);
            amx_vfma32(s_xy + 1, xi + 1, yi + 1);
            //xi_2 = xi * xi;
            amx_vfma32_old(xi_2_z + 0, xi + 0, xi_y + 0, 0, 0, 1);
            amx_vfma32_old(xi_2_z + 1, xi + 1, xi_y + 1, 0, 0, 1);
            //S_x += xi;
            amx_vfma32_old(s_x + 0, xi + 0, 0, 0, 1, 0);
            amx_vfma32_old(s_x + 1, xi + 1, 0, 0, 1, 0);
            //S_y += yi;
            amx_vfma32_old(s_y + 0, 0, yi + 0, 1, 0, 0);
            amx_vfma32_old(s_y + 1, 0, yi + 1, 1, 0, 0);
            amx_movxz(xi_2_x + 0, xi_2_z + 0);
            amx_movxz(xi_2_x + 1, xi_2_z + 1);
            amx_movyz(xi_2_y + 0, xi_2_z + 0);
            amx_movyz(xi_2_y + 1, xi_2_z + 1);
            //S_xx += xi_2;
            amx_vfma32_old(s_xx + 0, xi_2_x + 0, 0, 0, 1, 0);
            amx_vfma32_old(s_xx + 1, xi_2_x + 1, 0, 0, 1, 0);
            //S_xxy += xi_2 * yi
            amx_vfma32(s_xxy + 0, xi_2_x + 0, yi + 0);
            amx_vfma32(s_xxy + 1, xi_2_x + 1, yi + 1);
            //S_xxx += xi * xi_2
            amx_vfma32(s_xxx + 0, xi + 0, xi_2_y + 0);
            amx_vfma32(s_xxx + 1, xi + 1, xi_2_y + 1);
            //S_xxxx += xi_2 * xi_2
            amx_vfma32(s_xxxx + 0, xi_2_x + 0, xi_2_y + 0);
            amx_vfma32(s_xxxx + 1, xi_2_x + 1, xi_2_y + 1);
        }
        amx_movxz(0, s_x + 1);
        amx_movxz(1, s_y + 1);
        amx_movxz(2, s_xx + 1);
        amx_movxz(3, s_xy + 1);
        amx_movxz(4, s_xxx + 1);
        amx_movxz(5, s_xxy + 1);
        amx_movxz(6, s_xxxx + 1);
        amx_vfma32_old(s_x, 0, 0, 0, 1, 0);
        amx_vfma32_old(s_y, 1, 0, 0, 1, 0);
        amx_vfma32_old(s_xx, 2, 0, 0, 1, 0);
        amx_vfma32_old(s_xy, 3, 0, 0, 1, 0);
        amx_vfma32_old(s_xxx, 4, 0, 0, 1, 0);
        amx_vfma32_old(s_xxy, 5, 0, 0, 1, 0);
        amx_vfma32_old(s_xxxx, 6, 0, 0, 1, 0);

        float S_x_result[16], S_y_result[16], S_xx_result[16], S_xy_result[16], S_xxx_result[16], S_xxy_result[16], S_xxxx_result[16];

        amx_store512_z(s_x, S_x_result);
        amx_store512_z(s_y, S_y_result);
        amx_store512_z(s_xx, S_xx_result);
        amx_store512_z(s_xy, S_xy_result);
        amx_store512_z(s_xxx, S_xxx_result);
        amx_store512_z(s_xxy, S_xxy_result);
        amx_store512_z(s_xxx, S_xxxx_result);
        for (int j = 0; j < 16; j += simd_width_float) {
            S_x += simd_fhadd(*(simd_float *)(&S_x_result[j]));
            S_y += simd_fhadd(*(simd_float *)(&S_y_result[j]));
            S_xx += simd_fhadd(*(simd_float *)(&S_xx_result[j]));
            S_xy += simd_fhadd(*(simd_float *)(&S_xy_result[j]));
            S_xxx += simd_fhadd(*(simd_float *)(&S_xxx_result[j]));
            S_xxy += simd_fhadd(*(simd_float *)(&S_xxy_result[j]));
            S_xxxx += simd_fhadd(*(simd_float *)(&S_xxxx_result[j]));
        }

        amx_end();
    #endif
    for (; i <= n - simd_width_float; i += simd_width_float) {
        simd_float xi = *(simd_float *)(x + i); //x0, x1 // convert to doubles for testing purposes
        simd_float yi = *(simd_float *)(y + i); //y0, y1
        //copy xi into y4, y5
        simd_float xi_2 = simd_fmul(xi, xi); //z0 -> x4, z1 -> x5
        S_x += simd_fhadd(xi); //z0, z1
        S_y += simd_fhadd(yi); //z4, z5
        S_xx += simd_fhadd(xi_2); //z8, z9
        S_xy += simd_fhadd(simd_fmul(xi, yi)); //z12, z13
        S_xxx += simd_fhadd(simd_fmul(xi, xi_2)); //z16, z17
        S_xxy += simd_fhadd(simd_fmul(xi_2, yi)); //z20, x21
        S_xxxx += simd_fhadd(simd_fmul(xi_2, xi_2)); //z24, z25
    }
  
    
    for (; simd_width_float != 1 && i < n; ++i) {
        float xi = x[i];
        float yi = y[i];
        float xi_2 = xi * xi;
        S_x += xi;
        S_y += yi;
        S_xx += xi_2;
        S_xy += xi * yi;
        S_xxx += xi * xi_2;
        S_xxy += xi_2 * yi;
        S_xxxx += xi_2 * xi_2;
    }
    vec3 A[3] = {
        (vec3){.x = static_cast<double>(n), .y = S_x, .z = S_xx},
        (vec3){.x = S_x, .y = S_xx, .z = S_xxx},
        (vec3){.x = S_xx, .y = S_xxx, .z = S_xxxx}
    };

    vec3 b = (vec3){.x = S_y, .y = S_xy, .z = S_xxy};
    
    // Simple Gaussian elimination for 3x3 system
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int j = i + 1; j < 3; ++j) {
            if (std::abs(A[j].vals[i]) > std::abs(A[pivot].vals[i])) {
                pivot = j;
            }
        }
        std::swap(A[i], A[pivot]);
        std::swap(b.vals[i], b.vals[pivot]);
        for (int j = i + 1; j < 3; ++j) {
            double factor = A[j].vals[i] / A[i].vals[i];
            for (int k = i; k < 3; ++k) {
                A[j].vals[k] -= factor * A[i].vals[k];
            }
            b.vals[j] -= factor * b.vals[i];
        }
    }

    vec3 coeffs = {};
    for (int i = 2; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < 3; ++j) {
            sum += A[i].vals[j] * coeffs.vals[j];
        }
        coeffs.vals[i] = (b.vals[i] - sum) / A[i].vals[i];
    }
    return coeffs;
}

double price_american_put_lsm_cpp(
    double S0, double K, double T, double r, double sigma,
    int num_paths, int num_steps, int seed) {

    const simd_float simd_increasing = SIMD_INCREASING;

    num_paths = (num_paths + (simd_width_float - 1)) & ~(simd_width_float - 1);
    num_steps = (num_steps + (simd_width_float - 1)) & ~(simd_width_float - 1);

    const double dt = T / static_cast<double>(num_steps);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0, 1.0);

    std::vector<float> S(num_paths * num_steps);

    const float factor1 = (r - 0.5 * sigma * sigma) * dt;
    const float factor2 = sigma * std::sqrt(dt);
    for (int i = 0; i < num_paths; ++i) {
        S[i] = S0 * expf(factor1 + factor2 * dist(rng));
    }
    for (int j = 1; j < num_steps; ++j) {
        for (int i = 1; i < num_paths; ++i) {
            double Z = dist(rng);
            S[j * num_paths + i] = S[(j - 1) * num_paths + i] * expf(factor1 + factor2 * Z);
        }
    }

    std::vector<float> exp_vals(num_steps);

    for (int i = 0; i < num_steps; i += simd_width_float) {
        simd_float offset = simd_fadd(simd_increasing, simd_fbroadcast(i));
        simd_float input = simd_fmul(simd_fbroadcast(-r * dt), offset);
        *(simd_float *)(exp_vals.data() + i) = simd_fexp(input);
    }

    std::vector<float> last_cash_flow(num_paths);
    std::vector<int32_t> time_cash_flow(num_paths, num_steps - 1);

    const simd_float simd_K = simd_fbroadcast(K);
    const simd_float simd_0 = simd_fbroadcast(0);
    for (int i = 0; i < num_paths; i += simd_width_float) {
        simd_float simd_S = *(simd_float *)(S.data() + (num_steps - 1) * num_paths + i);
        *(simd_float *)(last_cash_flow.data() + i) = simd_fmax(simd_0, simd_fsub(simd_K, simd_S));
    }

    int *in_the_money_paths = (int *)malloc(num_paths * sizeof(int));
    #ifdef USE_APPLE_AMX
        float *x_itm = (float *)aligned_alloc(128, (num_paths * sizeof(float) + 127) & ~127);
        float *y_itm = (float *)aligned_alloc(128, (num_paths * sizeof(float) + 127) & ~127);
    #else
        float *x_itm = (float *)malloc(num_paths * sizeof(float));
        float *y_itm = (float *)malloc(num_paths * sizeof(float));
    #endif
    for (int t = num_steps - 2; t >= 0; --t) {
        size_t in_the_money_count = 0;
        for (int i = 0; (simd_width_int32 == simd_width_float) && i < num_paths; i += simd_width_float) {
            simd_float simd_S = *(simd_float *)(S.data() + t * num_paths + i);
            simd_mask mask = simd_fgt_mask(simd_fsub(simd_K, simd_S), simd_0);
            simd_int32 exp_indices = *(simd_int32 *)(time_cash_flow.data() + i);
            simd_float last = *(simd_float *)(last_cash_flow.data() + i);
            simd_float future_cf = simd_fmul(last, simd_fgather_int32_masked(exp_vals.data() - t, exp_indices, std::nanf(""), mask)); 

            #define PATHS_UPDATE_LOOP(__LANE__) do { \
                float future = simd_fget_lane(future_cf, __LANE__); \
                if (!std::isnan(future)) { \
                    in_the_money_paths[in_the_money_count] = i + __LANE__; \
                    x_itm[in_the_money_count] = simd_fget_lane(simd_S, __LANE__); \
                    y_itm[in_the_money_count] = future; \
                    ++in_the_money_count; \
                } \
            } while (0)
            SIMD_ITERATE_LANES(PATHS_UPDATE_LOOP);
        }
        for (int i = 0; (simd_width_float != simd_width_int32) && i < num_paths; ++i) {
            if (K - S[t * num_paths + i] > 0.0) {
                float future_cf = last_cash_flow[i] * exp_vals[time_cash_flow[i] - t];

                in_the_money_paths[in_the_money_count] = i;
                x_itm[in_the_money_count] = S[t * num_paths + i];
                y_itm[in_the_money_count] = future_cf;

                ++in_the_money_count;
            }
        }

        if (in_the_money_count == 0) continue;
        vec3 coeffs = polyfit(x_itm, y_itm, in_the_money_count);
        
        simd_float simd_c0 = simd_fbroadcast(coeffs.c0);
        simd_float simd_c1 = simd_fbroadcast(coeffs.c1);
        simd_float simd_c2 = simd_fbroadcast(coeffs.c2);
        size_t i = 0;
        for (; false && (simd_width_int32 == simd_width_float) && i <= in_the_money_count - simd_width_float; i += simd_width_float) {
            simd_int32 path_idx = *(simd_int32 *)(in_the_money_paths + i);
            simd_float x_val = *(simd_float *)(x_itm + i);
            simd_float continuation_value = simd_fadd(simd_fmul(simd_fmul(x_val, x_val), simd_c2), simd_fadd(simd_fmul(x_val, simd_c1), simd_c0));
            simd_float simd_S = simd_fgather_int32(S.data() + t * num_paths, path_idx);
            simd_float intrinsic_value = simd_fmax(simd_0, simd_fsub(simd_K, simd_S));

            simd_mask mask = simd_fgt_mask(intrinsic_value, continuation_value);
            simd_fscatter_int32_masked(last_cash_flow.data(), path_idx, intrinsic_value, mask);
            simd_iscatter_scalar_int32_masked(time_cash_flow.data(), path_idx, t, mask);
        }
        for (;(simd_width_float != 1 || simd_width_int32 != simd_width_float) && i < in_the_money_count; ++i) {
            int path_idx = in_the_money_paths[i];
            float x_val = x_itm[i];
            double continuation_value = coeffs.c2 * x_val * x_val + coeffs.c1 * x_val + coeffs.c0;
            double intrinsic_value = std::max(0.0, K - S[t * num_paths + path_idx]);

            if (intrinsic_value > continuation_value) {
                last_cash_flow[path_idx] = intrinsic_value;
                time_cash_flow[path_idx] = t;
            }
        }
    }
    free(in_the_money_paths);
    free(x_itm);
    free(y_itm);

    float total_payoff = 0.0;
    if (simd_width_int32 != simd_width_float) {
        for (int i = 0; i < num_paths; ++i) {
            total_payoff += last_cash_flow[i] * exp_vals[time_cash_flow[i]];
        }
    } else {
        for (int i = 0; i < num_paths; i += simd_width_int32) {
            simd_int32 simd_time_cash_flow = *(simd_int32 *)(time_cash_flow.data() + i);
            simd_float simd_exp_vals = simd_fgather_int32(exp_vals.data(), simd_time_cash_flow);
            simd_float simd_last_cash_flow = *(simd_float *)(last_cash_flow.data() + i);
            simd_float payouts = simd_fmul(simd_last_cash_flow, simd_exp_vals);
            total_payoff += simd_fhadd(payouts);
        }
    }

    return total_payoff / num_paths;
}
