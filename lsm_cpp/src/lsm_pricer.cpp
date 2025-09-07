// lsm_cpp/src/lsm_pricer.cpp

#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdlib>

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
vec3 polyfit(const double *x, const double *y, size_t n) {
    double S_x = 0, S_y = 0, S_xx = 0, S_xy = 0, S_xxx = 0, S_xxy = 0, S_xxxx = 0;

    for (size_t i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double xi_2 = xi * xi;
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
    
    const double dt = T / static_cast<double>(num_steps);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> S(num_paths * num_steps);

    const double factor1 = (r - 0.5 * sigma * sigma) * dt;
    const double factor2 = sigma * std::sqrt(dt);
    for (int i = 0; i < num_paths; ++i) {
        S[i * num_steps + 0] = S0 * std::exp(factor1 + factor2 * dist(rng));
        for (int j = 1; j < num_steps; ++j) {
            double Z = dist(rng);
            S[i * num_steps + j] = S[i * num_steps + j - 1] * std::exp(factor1 + factor2 * Z);
        }
    }

    std::vector<double> exp_vals(num_steps);

    for (int i = 0; i < num_steps; ++i) {
        exp_vals[i] = std::exp(-r * i * dt);
    }

    std::vector<double> last_cash_flow(num_paths);
    std::vector<int> time_cash_flow(num_paths, num_steps - 1);
    for (int i = 0; i < num_paths; ++i) {
        last_cash_flow[i] = std::max(0.0, K - S[i * num_steps + num_steps - 1]);
    }

    int *in_the_money_paths = (int *)malloc(num_paths * sizeof(int));
    double *x_itm = (double *)malloc(num_paths * sizeof(double));
    double *y_itm = (double *)malloc(num_paths * sizeof(double));
    for (int t = num_steps - 2; t >= 0; --t) {
        size_t in_the_money_count = 0;
        for (int i = 0; i < num_paths; ++i) {
            if (K - S[i * num_steps + t] > 0.0) {
                double future_cf = last_cash_flow[i] * exp_vals[time_cash_flow[i] - t];

                in_the_money_paths[in_the_money_count] = i;
                x_itm[in_the_money_count] = S[i * num_steps + t];
                y_itm[in_the_money_count] = future_cf;

                ++in_the_money_count;
            }
        }

        if (in_the_money_count == 0) continue;
        vec3 coeffs = polyfit(x_itm, y_itm, in_the_money_count);
        
        for (size_t i = 0; i < in_the_money_count; ++i) {
            int path_idx = in_the_money_paths[i];
            double x_val = x_itm[i];
            double continuation_value = coeffs.c2 * x_val * x_val + coeffs.c1 * x_val + coeffs.c0;
            double intrinsic_value = std::max(0.0, K - S[path_idx * num_steps + t]);

            if (intrinsic_value > continuation_value) {
                last_cash_flow[path_idx] = intrinsic_value;
                time_cash_flow[path_idx] = t;
            }
        }
    }
    free(in_the_money_paths);
    free(x_itm);
    free(y_itm);

    double total_payoff = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        total_payoff += last_cash_flow[i] * exp_vals[time_cash_flow[i]];
    }

    return total_payoff / num_paths;
}
