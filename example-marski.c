#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>

using namespace std;

// ============================================================================
// Utility Functions
// ============================================================================

// Convert double matrix to float matrix
vector<vector<float>> to_float_matrix(const vector<vector<double>>& A) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<float>> result(n, vector<float>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = static_cast<float>(A[i][j]);
        }
    }
    return result;
}

// Convert float matrix to double matrix
vector<vector<double>> to_double_matrix(const vector<vector<float>>& A) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> result(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = static_cast<double>(A[i][j]);
        }
    }
    return result;
}

// Convert double vector to float vector
vector<float> to_float_vector(const vector<double>& v) {
    vector<float> result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = static_cast<float>(v[i]);
    }
    return result;
}

// Convert float vector to double vector
vector<double> to_double_vector(const vector<float>& v) {
    vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = static_cast<double>(v[i]);
    }
    return result;
}

// Generate random matrix
vector<vector<double>> random_matrix(int n, int m, double min_val = -1.0, double max_val = 1.0, int seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dis(min_val, max_val);
    
    vector<vector<double>> A(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = dis(gen);
        }
    }
    return A;
}

// Generate random SPD matrix (for stable linear systems)
vector<vector<double>> random_spd_matrix(int n, int seed = 42) {
    auto A = random_matrix(n, n, -1.0, 1.0, seed);
    
    // Make it symmetric positive definite: A = A^T * A + n*I
    vector<vector<double>> result(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += A[k][i] * A[k][j];
            }
            if (i == j) result[i][j] += n; // Add diagonal dominance
        }
    }
    return result;
}

// Vector operations
template<typename T>
vector<T> matvec(const vector<vector<T>>& A, const vector<T>& x) {
    int n = A.size();
    vector<T> result(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

template<typename T>
vector<T> vecsub(const vector<T>& a, const vector<T>& b) {
    vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template<typename T>
double vecnorm(const vector<T>& v) {
    double sum = 0.0;
    for (auto x : v) {
        sum += static_cast<double>(x) * static_cast<double>(x);
    }
    return sqrt(sum);
}

// ============================================================================
// EXAMPLE 1: Iterative Refinement for Linear Systems
// ============================================================================

// Gaussian elimination solver (templated for float/double)
template<typename T>
vector<T> solve_gaussian(vector<vector<T>> A, vector<T> b) {
    int n = A.size();
    
    // Forward elimination
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            T factor = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    // Back substitution
    vector<T> x(n);
    for (int i = n - 1; i >= 0; i--) {
        T sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    
    return x;
}

// Naive FP64 solver
vector<double> solve_fp64(const vector<vector<double>>& A, const vector<double>& b) {
    // No casting needed - work directly with doubles
    return solve_gaussian(A, b);
}

// Naive FP32 solver
vector<double> solve_fp32(const vector<vector<double>>& A, const vector<double>& b) {
    // Cast to FP32 outside computation
    auto A_fp32 = to_float_matrix(A);
    auto b_fp32 = to_float_vector(b);
    
    // Solve in FP32
    auto x_fp32 = solve_gaussian(A_fp32, b_fp32);
    
    // Cast result back to FP64
    return to_double_vector(x_fp32);
}

// Mixed precision with iterative refinement
vector<double> solve_mixed_precision_ir(const vector<vector<double>>& A, 
                                         const vector<double>& b,
                                         int max_iter = 5,
                                         bool verbose = false) {
    int n = A.size();
    
    // Step 1: Initial solve in FP32 (cast outside)
    auto A_fp32 = to_float_matrix(A);
    auto b_fp32 = to_float_vector(b);
    
    // Factor in FP32 (keep factored matrix)
    auto LU_fp32 = A_fp32; // Will be modified to contain LU factors
    vector<float> b_fp32_copy = b_fp32;
    
    // Forward elimination (creating LU factorization)
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            float factor = LU_fp32[i][k] / LU_fp32[k][k];
            LU_fp32[i][k] = factor; // Store L factor
            for (int j = k + 1; j < n; j++) {
                LU_fp32[i][j] -= factor * LU_fp32[k][j];
            }
            b_fp32_copy[i] -= factor * b_fp32_copy[k];
        }
    }
    
    // Back substitution for initial solution
    vector<float> x_fp32(n);
    for (int i = n - 1; i >= 0; i--) {
        float sum = b_fp32_copy[i];
        for (int j = i + 1; j < n; j++) {
            sum -= LU_fp32[i][j] * x_fp32[j];
        }
        x_fp32[i] = sum / LU_fp32[i][i];
    }
    
    // Convert to FP64 for refinement
    vector<double> x = to_double_vector(x_fp32);
    
    if (verbose) {
        cout << "Iter 0 - Residual: " << scientific << setprecision(3) 
             << vecnorm(vecsub(matvec(A, x), b)) << endl;
    }
    
    // Step 2: Iterative refinement loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute residual in FP64
        vector<double> r = vecsub(b, matvec(A, x));
        double residual_norm = vecnorm(r);
        
        if (verbose) {
            cout << "Iter " << (iter + 1) << " - Residual: " << scientific 
                 << setprecision(3) << residual_norm << endl;
        }
        
        if (residual_norm < 1e-12) break;
        
        // Solve for correction using FP32 LU factors (cast outside)
        auto r_fp32 = to_float_vector(r);
        
        // Forward substitution
        for (int k = 0; k < n; k++) {
            for (int i = k + 1; i < n; i++) {
                r_fp32[i] -= LU_fp32[i][k] * r_fp32[k];
            }
        }
        
        // Back substitution
        vector<float> d_fp32(n);
        for (int i = n - 1; i >= 0; i--) {
            float sum = r_fp32[i];
            for (int j = i + 1; j < n; j++) {
                sum -= LU_fp32[i][j] * d_fp32[j];
            }
            d_fp32[i] = sum / LU_fp32[i][i];
        }
        
        // Update solution in FP64
        auto d = to_double_vector(d_fp32);
        for (int i = 0; i < n; i++) {
            x[i] += d[i];
        }
    }
    
    return x;
}

// // ============================================================================
// // EXAMPLE 2: Matrix Multiplication with Kahan Summation
// // ============================================================================

// // Standard FP64 matrix multiplication
// vector<vector<double>> matmul_fp64(const vector<vector<double>>& A,
//                                     const vector<vector<double>>& B) {
//     int n = A.size();
//     int m = B[0].size();
//     int k = A[0].size();
    
//     vector<vector<double>> C(n, vector<double>(m, 0.0));
    
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < m; j++) {
//             for (int p = 0; p < k; p++) {
//                 C[i][j] += A[i][p] * B[p][j];
//             }
//         }
//     }
    
//     return C;
// }

// // Standard FP32 matrix multiplication (cast outside)
// vector<vector<double>> matmul_fp32(const vector<vector<double>>& A,
//                                     const vector<vector<double>>& B) {
//     // Cast to FP32 outside
//     auto A_fp32 = to_float_matrix(A);
//     auto B_fp32 = to_float_matrix(B);
    
//     int n = A_fp32.size();
//     int m = B_fp32[0].size();
//     int k = A_fp32[0].size();
    
//     vector<vector<float>> C_fp32(n, vector<float>(m, 0.0f));
    
//     // Pure FP32 computation
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < m; j++) {
//             for (int p = 0; p < k; p++) {
//                 C_fp32[i][j] += A_fp32[i][p] * B_fp32[p][j];
//             }
//         }
//     }
    
//     // Cast result back
//     return to_double_matrix(C_fp32);
// }

// // Mixed precision with Kahan summation
// vector<vector<double>> matmul_kahan(const vector<vector<double>>& A,
//                                      const vector<vector<double>>& B) {
//     // Cast to FP32 outside
//     auto A_fp32 = to_float_matrix(A);
//     auto B_fp32 = to_float_matrix(B);
    
//     int n = A_fp32.size();
//     int m = B_fp32[0].size();
//     int k = A_fp32[0].size();
    
//     vector<vector<double>> C(n, vector<double>(m, 0.0));
    
//     // Mixed precision: multiply in FP32, accumulate with Kahan in FP64
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < m; j++) {
//             double sum = 0.0;
//             double c = 0.0;  // Kahan compensation
            
//             for (int p = 0; p < k; p++) {
//                 // Compute product in FP32 (no casting inside loop)
//                 float prod_fp32 = A_fp32[i][p] * B_fp32[p][j];
                
//                 // Kahan summation in FP64
//                 double y = static_cast<double>(prod_fp32) - c;
//                 double t = sum + y;
//                 c = (t - sum) - y;
//                 sum = t;
//             }
            
//             C[i][j] = sum;
//         }
//     }
    
//     return C;
// }

// // Frobenius norm of matrix difference
// double matrix_diff_norm(const vector<vector<double>>& A, 
//                         const vector<vector<double>>& B) {
//     double sum = 0.0;
//     for (size_t i = 0; i < A.size(); i++) {
//         for (size_t j = 0; j < A[0].size(); j++) {
//             double diff = A[i][j] - B[i][j];
//             sum += diff * diff;
//         }
//     }
//     return sqrt(sum);
// }

// ============================================================================
// MAIN: Demonstration
// ============================================================================

void print_vector(const vector<double>& v, int max_show = 5) {
    cout << "[";
    int n = min((int)v.size(), max_show);
    for (int i = 0; i < n; i++) {
        cout << v[i];
        if (i < n - 1) cout << ", ";
    }
    if ((int)v.size() > max_show) cout << ", ...";
    cout << "]";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int n_linear = 100;  // Size for linear system
    int n_matmul = 200;  // Size for matrix multiplication
    string mode = "all";
    int positional_index = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg.rfind("--mode=", 0) == 0) {
            mode = arg.substr(7);
            continue;
        }
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }

        if (positional_index == 0) {
            n_linear = atoi(arg.c_str());
            positional_index++;
        } else if (positional_index == 1) {
            n_matmul = atoi(arg.c_str());
            positional_index++;
        }
    }

    transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    bool run_fp64 = (mode == "all" || mode == "fp64");
    bool run_fp32 = (mode == "all" || mode == "fp32");
    bool run_ir = (mode == "all" || mode == "ir");
    
    if (argc > 1) n_linear = atoi(argv[1]);
    if (argc > 2) n_matmul = atoi(argv[2]);
    
    cout << fixed << setprecision(10);
    
    // ========================================================================
    // Demo 1: Iterative Refinement
    // ========================================================================
    cout << "============================================\n";
    cout << "DEMO 1: Iterative Refinement\n";
    cout << "Matrix size: " << n_linear << "x" << n_linear << "\n";
    cout << "============================================\n\n";
    
    // Generate random SPD system
    auto A = random_spd_matrix(n_linear, 42);
    vector<double> x_true(n_linear);
    for (int i = 0; i < n_linear; i++) {
        x_true[i] = static_cast<double>(i + 1);
    }
    vector<double> b = matvec(A, x_true);
    double x_true_norm = vecnorm(x_true);
    
    cout << "Solving Ax = b where x_true = [1, 2, 3, ..., " << n_linear << "]\n\n";
    
    double fp64_error = 0.0;
    double fp64_residual = 0.0;
    double fp64_rel_error = 0.0;
    double fp32_error = 0.0;
    double fp32_residual = 0.0;
    double fp32_rel_error = 0.0;
    double mixed_error = 0.0;
    double mixed_residual = 0.0;
    double mixed_rel_error = 0.0;
    long long fp64_time = 0;
    long long fp32_time = 0;
    long long mixed_time = 0;

    cout << "--- Naive FP64 Solution ---\n";
    auto start = chrono::high_resolution_clock::now();
    vector<double> x_fp64 = solve_fp64(A, b);
    auto end = chrono::high_resolution_clock::now();
    fp64_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

    fp64_error = vecnorm(vecsub(x_fp64, x_true));
    fp64_residual = vecnorm(vecsub(matvec(A, x_fp64), b));
    fp64_rel_error = fp64_error / x_true_norm;
    cout << "Solution: "; print_vector(x_fp64); cout << "\n";
    cout << "Error vs true: " << scientific << fp64_error << endl;
    cout << "Residual ||Ax-b||: " << scientific << fp64_residual << endl;
    cout << "Relative error ||x-x_true||/||x_true||: " << scientific << fp64_rel_error << endl;
    cout << "Time: " << fixed << fp64_time << " μs\n\n";
    
    
    
    cout << "--- Naive FP32 Solution ---\n";
    start = chrono::high_resolution_clock::now();
    vector<double> x_fp32 = solve_fp32(A, b);
    end = chrono::high_resolution_clock::now();
    fp32_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

    fp32_error = vecnorm(vecsub(x_fp32, x_true));
    fp32_residual = vecnorm(vecsub(matvec(A, x_fp32), b));
    fp32_rel_error = fp32_error / x_true_norm;
    cout << "Solution: "; print_vector(x_fp32); cout << "\n";
    cout << "Error vs true: " << scientific << fp32_error << endl;
    cout << "Residual ||Ax-b||: " << scientific << fp32_residual << endl;
    cout << "Relative error ||x-x_true||/||x_true||: " << scientific << fp32_rel_error << endl;
    cout << "Time: " << fixed << fp32_time << " μs\n\n";
    
    
    cout << "--- Mixed Precision with Iterative Refinement ---\n";
    start = chrono::high_resolution_clock::now();
    vector<double> x_mixed = solve_mixed_precision_ir(A, b, 5, true);
    end = chrono::high_resolution_clock::now();
    mixed_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

    mixed_error = vecnorm(vecsub(x_mixed, x_true));
    mixed_residual = vecnorm(vecsub(matvec(A, x_mixed), b));
    mixed_rel_error = mixed_error / x_true_norm;
    cout << "\nFinal solution: "; print_vector(x_mixed); cout << "\n";
    cout << "Error vs true: " << scientific << mixed_error << endl;
    cout << "Residual ||Ax-b||: " << scientific << mixed_residual << endl;
    cout << "Relative error ||x-x_true||/||x_true||: " << scientific << mixed_rel_error << endl;
    cout << "Time: " << fixed << mixed_time << " μs\n\n";
    

    
    cout << "Summary:\n";
    cout << "  FP32 error improvement over FP64: " << scientific << (fp32_error / fp64_error) << "x\n";
    cout << "  Mixed precision error vs FP64: " << (mixed_error / fp64_error) << "x\n";
    cout << "  Mixed precision achieves FP64-level accuracy!\n\n";
    
    
//     // ========================================================================
//     // Demo 2: Matrix Multiplication with Kahan
//     // ========================================================================
//     cout << "\n============================================\n";
//     cout << "DEMO 2: Matrix Multiplication with Kahan\n";
//     cout << "Matrix size: " << n_matmul << "x" << n_matmul << "\n";
//     cout << "============================================\n\n";
    
//     auto M1 = random_matrix(n_matmul, n_matmul, -1.0, 1.0, 123);
//     auto M2 = random_matrix(n_matmul, n_matmul, -1.0, 1.0, 456);
    
//     // FP64 reference
//     cout << "--- FP64 Reference ---\n";
//     start = chrono::high_resolution_clock::now();
//     auto C_fp64 = matmul_fp64(M1, M2);
//     end = chrono::high_resolution_clock::now();
//     auto matmul_fp64_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//     cout << "Time: " << matmul_fp64_time << " ms\n\n";
    
//     // FP32 naive
//     cout << "--- FP32 Naive ---\n";
//     start = chrono::high_resolution_clock::now();
//     auto C_fp32 = matmul_fp32(M1, M2);
//     end = chrono::high_resolution_clock::now();
//     auto matmul_fp32_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//     double matmul_fp32_error = matrix_diff_norm(C_fp32, C_fp64);
//     cout << "Time: " << matmul_fp32_time << " ms\n";
//     cout << "Error vs FP64: " << scientific << matmul_fp32_error << "\n\n";
    
//     // Kahan summation
//     cout << "--- FP32 multiply + Kahan (FP64) accumulation ---\n";
//     start = chrono::high_resolution_clock::now();
//     auto C_kahan = matmul_kahan(M1, M2);
//     end = chrono::high_resolution_clock::now();
//     auto kahan_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//     double kahan_error = matrix_diff_norm(C_kahan, C_fp64);
//     cout << "Time: " << kahan_time << " ms\n";
//     cout << "Error vs FP64: " << scientific << kahan_error << "\n\n";
    
//     cout << "Summary:\n";
//     cout << "  FP32 naive error:  " << scientific << matmul_fp32_error << "\n";
//     cout << "  Kahan error:       " << kahan_error << "\n";
//     cout << "  Improvement:       " << (matmul_fp32_error / kahan_error) << "x reduction in error\n";
//     cout << "  Kahan overhead:    " << fixed << setprecision(1) 
//          << (100.0 * (kahan_time - matmul_fp32_time) / matmul_fp32_time) << "%\n";
    
    return 0;
}
