#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>

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
// Gaussian Elimination Solver (templated for float/double)
// ============================================================================

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

vector<double> solve_fp64(const vector<vector<double>>& A, const vector<double>& b) {
    return solve_gaussian<double>(A, b);
}

// Naive FP32 solver
vector<double> solve_fp32(const vector<vector<double>>& A, const vector<double>& b) {
    // Cast to FP32 outside computation
    auto A_fp32 = to_float_matrix(A);
    auto b_fp32 = to_float_vector(b);

    // Solve in FP32
    auto x_fp32 = solve_gaussian<float>(A_fp32, b_fp32);

    // Cast result back to FP64
    return to_double_vector(x_fp32);
}

// ============================================================================
// MAIN: Demonstration (FP64 only)
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
    int n_linear = 100;
    if (argc > 1) n_linear = atoi(argv[1]);

    cout << fixed << setprecision(10);

    cout << "============================================\n";
    cout << "DEMO: Gaussian Solve (FP64)\n";
    cout << "Matrix size: " << n_linear << "x" << n_linear << "\n";
    cout << "============================================\n\n";

    // Generate random SPD system
    auto A = random_spd_matrix(n_linear, 42);
    vector<double> x_true(n_linear);
    for (int i = 0; i < n_linear; i++) {
        x_true[i] = static_cast<double>(i + 1);
    }
    vector<double> b = matvec(A, x_true);

    cout << "Solving Ax = b where x_true = [1, 2, 3, ..., " << n_linear << "]\n\n";

    cout << "--- FP64 Solution ---\n";
    auto start = chrono::high_resolution_clock::now();
    vector<double> x_fp64 = solve_fp64(A, b);
    auto end = chrono::high_resolution_clock::now();
    auto fp64_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

    double fp64_error = vecnorm(vecsub(x_fp64, x_true));
    double fp64_residual = vecnorm(vecsub(matvec(A, x_fp64), b));
    double x_true_norm = vecnorm(x_true);
    double fp64_rel_error = fp64_error / x_true_norm;

    cout << "Solutions: "; print_vector(x_fp64); cout << "\n";
    cout << "Error vs true: " << scientific << fp64_error << endl;
    cout << "Residual ||Ax-b||: " << scientific << fp64_residual << endl;
    cout << "Relative error ||x-x_true||/||x_true||: " << scientific << fp64_rel_error << endl;
    cout << "Time: " << fixed << fp64_time << " μs\n\n";

    // Naive FP32
    cout << "--- FP32 Solution ---\n";
    start = chrono::high_resolution_clock::now();
    vector<double> x_fp32 = solve_fp32(A, b);
    end = chrono::high_resolution_clock::now();
    auto fp32_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

    double fp32_error = vecnorm(vecsub(x_fp32, x_true));
    double fp32_residual = vecnorm(vecsub(matvec(A, x_fp32), b));
    double fp32_rel_error = fp32_error / x_true_norm;

    cout << "Solution: "; print_vector(x_fp32); cout << "\n";
    cout << "Error vs true: " << scientific << fp32_error << endl;
    cout << "Residual ||Ax-b||: " << scientific << fp32_residual << endl;
    cout << "Relative error ||x-x_true||/||x_true||: " << scientific << fp32_rel_error << endl;
    cout << "Time: " << fixed << fp32_time << " μs\n\n";

    return 0;
}
