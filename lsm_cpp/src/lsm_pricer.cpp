// lsm_cpp/src/lsm_pricer.cpp

#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>

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
vec3 polyfit(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    double S_x = 0, S_y = 0, S_xx = 0, S_xy = 0, S_xxx = 0, S_xxy = 0, S_xxxx = 0;

    for (int i = 0; i < n; ++i) {
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
        {static_cast<double>(n), S_x, S_xx},
        {S_x, S_xx, S_xxx},
        {S_xx, S_xxx, S_xxxx}
    };

    vec3 b = {S_y, S_xy, S_xxy};
    
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
    
    double dt = T / static_cast<double>(num_steps);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<std::vector<double>> S(num_paths, std::vector<double>(num_steps + 1));
    for (int i = 0; i < num_paths; ++i) {
        S[i][0] = S0;
        for (int j = 1; j <= num_steps; ++j) {
            double Z = dist(rng);
            S[i][j] = S[i][j - 1] * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        }
    }

    std::vector<std::vector<double>> cash_flows(num_paths, std::vector<double>(num_steps + 1, 0.0));
    for (int i = 0; i < num_paths; ++i) {
        cash_flows[i][num_steps] = std::max(0.0, K - S[i][num_steps]);
    }

    for (int t = num_steps - 1; t > 0; --t) {
        std::vector<int> in_the_money_paths;
        std::vector<double> x_itm, y_itm;

        in_the_money_paths.reserve(num_paths);
        x_itm.reserve(num_paths);
        y_itm.reserve(num_paths);
        for (int i = 0; i < num_paths; ++i) {
            if (K - S[i][t] > 0.0) {
                in_the_money_paths.push_back(i);
                x_itm.push_back(S[i][t]);
                
                double future_cf = 0.0;
                for (int j = t + 1; j <= num_steps; ++j) {
                    if (cash_flows[i][j] > 0.0) {
                        future_cf = cash_flows[i][j] * std::exp(-r * (j - t) * dt);
                        break;
                    }
                }
                y_itm.push_back(future_cf);
            }
        }

        if (x_itm.empty()) continue;

        vec3 coeffs = polyfit(x_itm, y_itm);
        
        for (size_t i = 0; i < in_the_money_paths.size(); ++i) {
            int path_idx = in_the_money_paths[i];
            double x_val = x_itm[i];
            double continuation_value = coeffs.c2 * x_val * x_val + coeffs.c1 * x_val + coeffs.c0;
            double intrinsic_value = std::max(0.0, K - S[path_idx][t]);

            if (intrinsic_value > continuation_value) {
                cash_flows[path_idx][t] = intrinsic_value;
                for (int j = t + 1; j <= num_steps; ++j) {
                    cash_flows[path_idx][j] = 0.0;
                }
            }
        }
    }

    double total_payoff = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        for (int j = 1; j <= num_steps; ++j) {
            if (cash_flows[i][j] > 0.0) {
                total_payoff += cash_flows[i][j] * std::exp(-r * j * dt);
                break;
            }
        }
    }

    return total_payoff / num_paths;
}
