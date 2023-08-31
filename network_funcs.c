#include <stdio.h>
#include <stdlib.h>
#include "network.h"

void calc_layer_output(Layer *layer, Layer *input_layer) {
  Matrix *input = input_layer->output;
  if (input->columns != layer->weights->rows) {
    fprintf(stderr, "inproper matrix multiplication");
    exit(1);
  }
  layer->output = matmul(input, layer->weights);
  print_matmul(input, layer->weights);
}

Matrix *weights_gradients_subtotal(Layer *layer) {
  unsigned int rows = layer->weights->rows - 1;
  unsigned int cols = layer->weights->columns - 1;

  double rv_arr[rows][cols];
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      double output_grad = layer->output_grads->values[0][j];
      rv_arr[i][j] = layer->weights->values[i][j] * output_grad;
    }
  }
  Matrix *weight_grads = allocate_from_2D_arr(rows, cols, rv_arr);
  // print_matrix_verbose(weight_grads);
  return weight_grads;
}

void calc_layer_output_gradients(Layer *layer, Layer *above_layer) {
  Matrix *weight_grads_above = weights_gradients_subtotal(above_layer);
  unsigned int rv_cols = layer->output->columns - 1;
  Matrix *output_grads = allocate_empty(1, rv_cols);

  for (unsigned int i = 0; i < rv_cols; i++) {
    for (unsigned int j = 0; j < weight_grads_above->columns; j++) {
      output_grads->values[0][i] += weight_grads_above->values[i][j];
    }
  }

  layer->output_grads = output_grads;
  // print_matrix_verbose(output_grads);
}