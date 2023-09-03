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