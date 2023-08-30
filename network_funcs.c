#include <stdio.h>
#include <stdlib.h>
#include "network.h"

Matrix *calc_layer_output(Layer *layer, Matrix *input) {
  if (input->columns != layer->weights->rows) {
    fprintf(stderr, "inproper matrix multiplication");
    exit(1);
  }
  print_matmul(input, layer->weights);
  return matmul(input, layer->weights);
}

void calc_weights_gradient(Layer *layer) {
  unsigned int rows = layer->weights->rows;
  unsigned int cols = layer->weights->columns;

  double modified_weights[rows][cols - 1];
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      double output_grad = layer->output_grads->values[0][j];
      if (i == rows - 1) {
        modified_weights[i][j] = output_grad;
      } else {
        modified_weights[i][j] = layer->weights->values[i][j] * output_grad;
      }
    }
  }
  Matrix *weight_grads = allocate_from_2D_arr(rows, cols - 1, modified_weights);
  layer->weight_grads = weight_grads;
  print_matrix_verbose(weight_grads);
}