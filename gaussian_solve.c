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
vector<double> matvec(const vector<vector<double>>& A, const vector<double>& x) {
    int n = A.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

vector<double> vecsub(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double vecnorm(const vector<double>& v) {
    double sum = 0.0;
    for (auto x : v) {
        sum += x * x;
    }
    return sqrt(sum);
}

// ============================================================================
// Gaussian Elimination Solver (FP64)
// ============================================================================

// Compute elimination factor (division operation)
double compute_factor(double pivot_elem, double target_elem) {
    return target_elem / pivot_elem;
}

// Update matrix row during elimination
void update_matrix_row(vector<double>& row_i, const vector<double>& row_k, double factor, int start_col, int n) {
    for (int j = start_col; j < n; j++) {
        row_i[j] -= factor * row_k[j];
    }
}

// Update RHS vector during elimination
void update_rhs_element(double& b_i, double b_k, double factor) {
    b_i -= factor * b_k;
}

// Perform back substitution step
double back_substitution_step(const vector<vector<double>>& A, const vector<double>& x, 
                               double b_i, int i, int n) {
    double sum = b_i;
    for (int j = i + 1; j < n; j++) {
        sum -= A[i][j] * x[j];
    }
    return sum / A[i][i];
}

vector<double> solve_gaussian(vector<vector<double>> A, vector<double> b) {
    int n = A.size();

    // Forward elimination
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = compute_factor(A[k][k], A[i][k]);
            update_matrix_row(A[i], A[k], factor, k, n);
            update_rhs_element(b[i], b[k], factor);
        }
    }

    // Back substitution
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = back_substitution_step(A, x, b[i], i, n);
    }

    return x;
}

vector<double> solve_fp64(const vector<vector<double>>& A, const vector<double>& b) {
    return solve_gaussian(A, b);
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

    cout << "--- Solutions ---\n";
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

    return 0;
}
