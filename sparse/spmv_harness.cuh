/*
 * spmv_harness.cuh: shared host-side driver for the large-matrix SpMV steps
 * (04-07).
 *
 * Steps 00-03 use small matrices with uniformly random sparsity, where every
 * row has about the same length. Real sparse matrices (web graphs, social
 * networks, finite-element meshes with refinement) are irregular, and that
 * irregularity is what separates SpMV formats. This harness provides:
 * - a large generated matrix with power-law row lengths: most rows hold a
 *   handful of nonzeros, a few hold thousands;
 * - or any real matrix in Matrix Market format with --mtx path/to/matrix.mtx
 *   (for example from the SuiteSparse Matrix Collection, sparse.tamu.edu);
 * - a double-precision CPU reference for y = A x, validation, and timing
 *   reported as GFLOP/s (2 flops per nonzero) and GB/s of the minimum traffic
 *   (the matrix, x, and y each moved once).
 *
 * A step provides `setup`, which builds its own format from the CSR matrix
 * once (outside the timing), and `run`, which computes y = A x.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "lab.cuh"

struct CsrHost {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> vals;
  int nnz() const { return static_cast<int>(vals.size()); }
};

struct CsrDevice {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  int *row_ptr = nullptr;
  int *col_idx = nullptr;
  float *vals = nullptr;
};

using SpmvSetup = void (*)(const CsrHost &host, const CsrDevice &device);
using SpmvRun = void (*)(const float *d_x, float *d_y);

// Rows with Pareto-distributed lengths (min 4, shape 1.5, capped), random
// distinct sorted columns, values in [-1, 1).
inline CsrHost power_law_matrix(int rows, int cols, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::uniform_int_distribution<int> col(0, cols - 1);
  std::uniform_real_distribution<float> val(-1.f, 1.f);
  const int cap = std::min(cols, 20000);
  CsrHost m;
  m.rows = rows;
  m.cols = cols;
  m.row_ptr.push_back(0);
  std::vector<int> cols_in_row;
  for (int r = 0; r < rows; ++r) {
    const double len = 4.0 * std::pow(1.0 - u(gen), -1.0 / 1.5);
    const int target = static_cast<int>(std::min<double>(len, cap));
    cols_in_row.clear();
    for (int k = 0; k < target; ++k) cols_in_row.push_back(col(gen));
    std::sort(cols_in_row.begin(), cols_in_row.end());
    cols_in_row.erase(std::unique(cols_in_row.begin(), cols_in_row.end()),
                      cols_in_row.end());
    for (int c : cols_in_row) {
      m.col_idx.push_back(c);
      m.vals.push_back(val(gen));
    }
    m.row_ptr.push_back(static_cast<int>(m.col_idx.size()));
  }
  return m;
}

// Reads a Matrix Market "coordinate" file (real, integer, or pattern;
// general or symmetric) into CSR with sorted columns.
inline CsrHost read_matrix_market(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  std::getline(in, line);
  if (line.find("coordinate") == std::string::npos) {
    std::fprintf(stderr, "%s: only 'coordinate' Matrix Market files are supported\n",
                 path.c_str());
    std::exit(EXIT_FAILURE);
  }
  const bool pattern = line.find("pattern") != std::string::npos;
  const bool symmetric = line.find("symmetric") != std::string::npos;
  while (std::getline(in, line) && !line.empty() && line[0] == '%') {
  }
  int rows = 0, cols = 0;
  long long entries = 0;
  std::istringstream(line) >> rows >> cols >> entries;

  std::vector<std::vector<std::pair<int, float>>> by_row(rows);
  for (long long e = 0; e < entries; ++e) {
    int r = 0, c = 0;
    double v = 1.0;
    in >> r >> c;
    if (!pattern) in >> v;
    by_row[r - 1].push_back({c - 1, static_cast<float>(v)});
    if (symmetric && r != c) by_row[c - 1].push_back({r - 1, static_cast<float>(v)});
  }
  CsrHost m;
  m.rows = rows;
  m.cols = cols;
  m.row_ptr.push_back(0);
  for (auto &row : by_row) {
    std::sort(row.begin(), row.end());
    for (auto &[c, v] : row) {
      m.col_idx.push_back(c);
      m.vals.push_back(v);
    }
    m.row_ptr.push_back(static_cast<int>(m.col_idx.size()));
  }
  return m;
}

inline int run_spmv(const char *name, int argc, char **argv, SpmvSetup setup,
                    SpmvRun run) {
  lab::Args args(argc, argv);
  lab::print_device();

  CsrHost A;
  std::string source = "power-law generated";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--mtx") {
      source = argv[i + 1];
      A = read_matrix_market(source);
    }
  }
  if (A.rows == 0) {
    const int rows = static_cast<int>(args.get_int("rows", args.quick() ? 5000 : 500000));
    A = power_law_matrix(rows, rows, 71);
  }
  int longest = 0;
  for (int r = 0; r < A.rows; ++r) longest = std::max(longest, A.row_ptr[r + 1] - A.row_ptr[r]);
  std::printf("%s: %d x %d matrix (%s), %d nonzeros, %.1f per row on average, "
              "longest row %d\n",
              name, A.rows, A.cols, source.c_str(), A.nnz(),
              static_cast<double>(A.nnz()) / A.rows, longest);

  const std::vector<float> x = lab::random_uniform<float>(A.cols, -1.f, 1.f, 72);
  // Reference, plus each row's sum of |a * x|: rounding error in a float dot
  // product is bounded relative to that sum, not to the (possibly cancelling)
  // result, which matters for rows with thousands of terms.
  std::vector<double> expected(A.rows, 0.0), magnitude(A.rows, 0.0);
  for (int r = 0; r < A.rows; ++r) {
    for (int k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k) {
      const double term = static_cast<double>(A.vals[k]) * x[A.col_idx[k]];
      expected[r] += term;
      magnitude[r] += std::fabs(term);
    }
  }

  CsrDevice dA;
  dA.rows = A.rows;
  dA.cols = A.cols;
  dA.nnz = A.nnz();
  CUDA_CHECK(cudaMalloc(&dA.row_ptr, (A.rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dA.col_idx, std::max(1, A.nnz()) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dA.vals, std::max(1, A.nnz()) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA.row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA.col_idx, A.col_idx.data(), A.nnz() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA.vals, A.vals.data(), A.nnz() * sizeof(float),
                        cudaMemcpyHostToDevice));
  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, A.cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, A.rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), A.cols * sizeof(float), cudaMemcpyHostToDevice));

  setup(A, dA);
  run(d_x, d_y);
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(A.rows);
  CUDA_CHECK(cudaMemcpy(got.data(), d_y, A.rows * sizeof(float), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  double max_err = 0.0;
  for (int r = 0; r < A.rows; ++r) {
    const double err = std::fabs(got[r] - expected[r]);
    max_err = std::max(max_err, err);
    if (!(err <= 1e-4 * magnitude[r] + 1e-6)) {
      if (mismatches < 5) {
        std::fprintf(stderr, "  y[%d]: got %.9g, expected %.9g\n", r, got[r], expected[r]);
      }
      ++mismatches;
    }
  }
  const bool pass = mismatches == 0;
  std::printf("Check %-24s %s (max abs error %.3g", "y = A x", pass ? "ok" : "FAILED", max_err);
  if (!pass) std::printf(", %zu of %d rows mismatched", mismatches, A.rows);
  std::printf(")\n");

  const float ms = lab::time_ms([&] { run(d_x, d_y); });
  const double bytes = static_cast<double>(A.nnz()) * (sizeof(int) + sizeof(float)) +
                       (A.rows + 1.0) * sizeof(int) + A.cols * sizeof(float) +
                       A.rows * sizeof(float);
  lab::report(name, ms, 2.0 * A.nnz(), bytes);

  CUDA_CHECK(cudaFree(dA.row_ptr));
  CUDA_CHECK(cudaFree(dA.col_idx));
  CUDA_CHECK(cudaFree(dA.vals));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return lab::finish(pass);
}
