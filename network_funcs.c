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