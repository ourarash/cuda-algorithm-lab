#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>
denseToCSR(std::vector<std::vector<float>>& denseMatrix,
           std::vector<int>& rowPtr, std::vector<int>& colInd,
           std::vector<float>& values) {
  if (denseMatrix.empty() || denseMatrix[0].empty()) {
    return;
  }
  int numRows = denseMatrix.size();
  int numCols = denseMatrix[0].size();

  rowPtr.resize(numRows + 1, 0);
  colInd.clear();
  values.clear();

  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      if (denseMatrix[i][j] != 0.0f) {
        colInd.push_back(j);
        values.push_back(denseMatrix[i][j]);
      }
    }
    rowPtr[i + 1] = colInd.size();
  }
}

void denseToCSC(const std::vector<std::vector<float>>& denseMatrix,
                std::vector<int>& colPtr, std::vector<int>& rowInd,
                std::vector<float>& values) {
  if (denseMatrix.empty() || denseMatrix[0].empty()) {
    return;
  }
  int numRows = denseMatrix.size();
  int numCols = denseMatrix[0].size();

  colPtr.resize(numCols + 1, 0);
  rowInd.clear();
  values.clear();

  // Outer loop must be over COLUMNS for CSC format
  for (int j = 0; j < numCols; ++j) {
    // Inner loop is over rows
    for (int i = 0; i < numRows; ++i) {
      if (denseMatrix[i][j] != 0.0f) {
        // Add the value and its corresponding row index
        values.push_back(denseMatrix[i][j]);
        rowInd.push_back(i);
      }
    }
    // After processing a full column, update the column pointer.
    // It marks the start of the *next* column.
    colPtr[j + 1] = values.size();
  }
}

void sparseMatrixMultiplyCSRByCSCSequential(
    const std::vector<int>& rowPtrA, const std::vector<int>& colIndA,
    const std::vector<float>& valuesA,  // Input: Matrix A (CSR)

    const std::vector<int>& colPtrB, const std::vector<int>& rowIndB,
    const std::vector<float>& valuesB,  // Input: Matrix B (CSC)

    std::vector<int>& rowPtrC, std::vector<int>& colIndC,
    std::vector<float>& valuesC) {  // Output: Matrix C (CSR)

  // 1. Initialization
  int numRowsA = rowPtrA.size() - 1;  // Get the number of rows in matrix A
                                      // (rowPtrA has numRowsA + 1 elements)

  rowPtrC.resize(numRowsA + 1, 0);  // Initialize rowPtrC for the result matrix
                                    // C. It will have the same number of rows
                                    // as A. All elements are initialized to 0.

  colIndC.clear();  // Clear previous data in output column indices
  valuesC.clear();  // Clear previous data in output values

  // 2. Main Loop: Iterate over each row of Matrix A
  // This outer loop processes one row of the result matrix C at a time.
  for (int i = 0; i < numRowsA; ++i) {
    // 3. Accumulation Map for the current row of C
    // std::map<int, float> acc;
    // Changed to std::map<int, float> acc; from std::unordered_map
    // to ensure sorted column indices in the output CSR.
    // This map will store the accumulated values for each column in the current
    // row 'i' of C. Key: Column index in C (colC) Value: Accumulated value
    // C[i][colC]
    std::map<int, float>
        acc;  // using std::map ensures keys are sorted, which is good for CSR

    // 4. Inner Loop: Iterate through non-zero elements in row 'i' of Matrix A
    // rowPtrA[i] gives the start index for row 'i' in colIndA and valuesA.
    // rowPtrA[i+1] gives the end index (exclusive).
    for (int j = rowPtrA[i]; j < rowPtrA[i + 1]; ++j) {
      int colA = colIndA[j];    // colA is the column index of the current
                                // non-zero element in A[i][colA]
      float valA = valuesA[j];  // valA is the value A[i][colA]

      // 5. Innermost Loop: Iterate through non-zero elements in the
      // corresponding column of Matrix B The `colA` from Matrix A's non-zero
      // element (A[i][colA]) becomes the row index `j` in `B[j][k]`. So, we
      // need to look at column `colA` of B. colPtrB[colA] gives the start index
      // for column `colA` in rowIndB and valuesB. colPtrB[colA + 1] gives the
      // end index (exclusive).
      for (int k = colPtrB[colA]; k < colPtrB[colA + 1]; ++k) {
        int rowB =
            rowIndB[k];  // rowB is the row index of the current non-zero
                         // element in B. Since B is CSC, rowIndB[k] effectively
                         // gives the 'k' in B_jk, which becomes the column
                         // index in C_ik. So, rowB represents the column index
                         // `colC` in the result matrix C.
        float valB = valuesB[k];  // valB is the value B[rowB][colA]

        // Accumulate the product: (A[i][colA] * B[colA][rowB])
        // This product contributes to C[i][rowB].
        acc[rowB] += valA * valB;
      }
    }

    // 6. Store accumulated values for the current row of C
    // After processing all contributions to row 'i' of C,
    // iterate through the accumulated map 'acc'.
    // std::map iterates its elements in sorted key order, which is perfect for
    // CSR's requirement that column indices within a row are sorted.
    for (const auto& [colC, valC] : acc) {
      colIndC.push_back(colC);  // Add the column index to C's column index list
      valuesC.push_back(valC);  // Add the corresponding value to C's value list
    }

    // 7. Update rowPtrC for the next row
    // rowPtrC[i+1] stores the total number of non-zero elements found so far
    // (which is also the starting index for the next row's non-zero elements).
    // colIndC.size() gives the current total number of non-zero elements stored
    // in C.
    rowPtrC[i + 1] = colIndC.size();
  }
}

__global__ void sparseMatrixMultiply(const int* rowPtrA, const int* colIndA,
                                     const float* valuesA, const int* rowPtrB,
                                     const int* colIndB, const float* valuesB,
                                     int* rowPtrC, int* colIndC, float* valuesC,
                                     int numRowsA, int numColsB) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {}