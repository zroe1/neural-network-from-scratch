#include <criterion/criterion.h>
#include "../network.h"

Test(matmul, matmul00)
{
  Matrix *A = allocate_empty(2, 1);
  Matrix *B = allocate_empty(1, 2);

  A->values[0][0] = 1;
  B->values[0][0] = 1;

  double solution_arr[2][2] = {
    {1, 0},
    {0, 0}
  };

  Matrix *solution = matmul(A, B);

  for (unsigned int i = 0; i < 2; i++) {
    for (unsigned int j = 0; j < 2; j++) {
      if (solution->values[i][j] - solution_arr[i][j] > 0.0001) {
        free_matrix(A);
        free_matrix(B);
        free_matrix(solution);
        cr_assert(0);
      }
    }
  }

  free_matrix(A);
  free_matrix(B);
  free_matrix(solution);

  cr_assert(1);
}

Test(matmul, matmul01)
{
  double A_arr[2][1] = {
    {1},
    {2}
  };
  double B_arr[1][2] = {
    {1, 2}
  };

  Matrix *A = allocate_from_2D_arr(2, 1, A_arr);
  Matrix *B = allocate_from_2D_arr(1, 2, B_arr);

  double solution_arr[2][2] = {
    {1, 2},
    {2, 4}
  };

  Matrix *solution = matmul(A, B);

  for (unsigned int i = 0; i < 2; i++) {
    for (unsigned int j = 0; j < 2; j++) {
      if (solution->values[i][j] - solution_arr[i][j] > 0.0001) {
        free_matrix(A);
        free_matrix(B);
        free_matrix(solution);
        cr_assert(0);
      }
    }
  }

  free_matrix(A);
  free_matrix(B);
  free_matrix(solution);

  cr_assert(1);
}

Test(flatten_matrix, flatten00)
{
  Matrix *m = allocate_empty(2, 2);
  Matrix *flattened = flatten_matrix(m);
  // print_matrix_verbose(m);
  // print_matrix_verbose(flattened);

  cr_assert(flattened->rows == 1);
  cr_assert(flattened->columns == 4);

  free_matrix(m);
  free_matrix(flattened);
}

Test(flatten_matrix, flatten01)
{
  double arr[3][2] = {
    {1, 2},
    {3, 4},
    {5, 6}
  };

  Matrix *m = allocate_from_2D_arr(3, 2, arr);
  Matrix *flattened = flatten_matrix(m);

  cr_assert(flattened->rows == 1);
  cr_assert(flattened->columns == 6);
  cr_assert(flattened->values[0][0] == 1);
  cr_assert(flattened->values[0][5] == 6);

  free_matrix(m);
  free_matrix(flattened);
}