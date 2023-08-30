#include <stdio.h>
#include <stdlib.h>
#include "network.h"

Matrix *allocate_empty(unsigned int rows, unsigned int columns) {
  double **matrix_values = (double **)malloc(sizeof(double *) * rows);
  for (unsigned int i = 0; i < rows; i++) {
    double *current_column = (double *)calloc(columns, sizeof(double));
    matrix_values[i] = current_column;
  }
  
  Matrix *rv = (Matrix *)malloc(sizeof(Matrix));
  rv->rows = rows;
  rv->columns = columns;
  rv->values = matrix_values;
  return rv;
}

Matrix *allocate_from_2D_arr(unsigned int rows, unsigned int columns, 
                             double arr[rows][columns]) {
  double **matrix_values = (double **)malloc(sizeof(double *) * rows);
  for (unsigned int i = 0; i < rows; i++) {
    double *current_column = (double *)malloc(columns * sizeof(double));
    for (unsigned int j = 0; j < columns; j++) {
      current_column[j] = arr[i][j];
    }
    matrix_values[i] = current_column;
  }
  
  Matrix *rv = (Matrix *)malloc(sizeof(Matrix));
  rv->rows = rows;
  rv->columns = columns;
  rv->values = matrix_values;
  return rv;
  
}

void free_matrix(Matrix *matrix) {
  for (unsigned int i = 0; i < matrix->rows; i++) {
    free(matrix->values[i]);
  }
  free(matrix->values);
  free(matrix);
}

/* Assumes the the vectors are both rows or both columns */
double dot_product(Matrix *vec1, Matrix *vec2) {
  double rv = 0;

  for (unsigned int i = 0; i < vec1->rows; i++) {
    for (unsigned int j = 0; j < vec1->columns; j++) {
      rv += vec1->values[i][j] * vec2->values[i][j];
    }
  }

  return rv;
}

static void print_dot_operation(Matrix *A, Matrix *B, double result) {
  unsigned int output_height, output_height_half;
  output_height = A->rows + 2;
  output_height_half = (output_height - 1) / 2;

  for (unsigned int i = 0; i < output_height; i++) {
    print_matrix_row(A, i);
    if (i == output_height_half)
      printf("  .  ");
    else
      printf("     ");

    print_matrix_row(B, i);
    if (i == output_height_half)
      printf("  =  %f\n", result);
    else
      putchar('\n');
  }
}

void print_dot(Matrix *A, Matrix *B) {
  double scalar = dot_product(A, B);
  print_dot_operation(A, B, scalar);
}


/*
 * NOTE: The following functions are for printing matrices and operations on
 * matrices. The code is writen so it is possible to print whole matrix 
 * operations over the same lines. If this code is refactored it should preserve that quality.
 */
static unsigned int len_of_num(double num) {
  char num_str[50];
  sprintf(num_str, "%.2f", num);
  
  unsigned int rv = 0;
  while (num_str[rv] != '\0') {
    rv++;
  }
  return rv;
}

/* NOTE: this helper function exists and is written like this becuase it 
makes it easier to print out operations involving multiple matrices */
void print_matrix_row(Matrix *matrix, int row) {
  unsigned int space_per_num = 8;

  if (row == -1) {
    // prints empty row
    for (unsigned int col = 0; col < matrix->columns; col++) {
      for (unsigned int i = 0; i < space_per_num; i++) {
        putchar(' ');
      }
    }
    printf("  ");
    return;
  }

  if (row == 0) { 
    // prints top line
    putchar('/');
    for (unsigned int col = 0; col < matrix->columns; col++) {
      for (unsigned int i = 0; i < space_per_num; i++) {
        putchar(' ');
      }
    }
    putchar('\\');
    return;
  }

  if (row == matrix->rows + 1) {
    // prints bottom line
    putchar('\\');
    for (unsigned int col = 0; col < matrix->columns; col++) {
      for (unsigned int i = 0; i < space_per_num; i++) {
        putchar(' ');
      }
    }
    putchar('/');
    return;
  }

  // print row of values of matrix
  putchar('|');
  for (unsigned int j = 0; j < matrix->columns; j++) {
    unsigned int num_len = len_of_num(matrix->values[row - 1][j]);
    unsigned int left_padding = (space_per_num - num_len) / 2;
    unsigned int right_padding = space_per_num - num_len - left_padding;
    
    while (left_padding-- > 0)
      putchar(' ');
    printf("%.2f", matrix->values[row - 1][j]);
    while (right_padding-- > 0)
      putchar(' ');
  }
  printf("|");
}

void print_matrix(Matrix *matrix) {
  for (int i = 0; i <= matrix->rows + 1; i++) {
    print_matrix_row(matrix, i);
    putchar('\n');
  }
}

void print_matrix_verbose(Matrix *matrix) {
  printf("%d * %d matrix:\n", matrix->rows, matrix->columns);
  print_matrix(matrix);
}

