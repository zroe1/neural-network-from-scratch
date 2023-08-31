#include <stdio.h>
#include <stdlib.h>
#include "network.h"

Layer *init_layer(Matrix *output, 
                  Matrix *output_grads, 
                  Matrix *weights, 
                  Matrix *weight_grads) 
{
  Layer *rv = (Layer *)malloc(sizeof(Layer));
  rv->output = output;
  rv->output_grads = output_grads;
  rv->weights = weights;
  rv->weight_grads = weight_grads;
  return rv;
}

RELU_Layer *init_RELU_layer(Matrix *output, Matrix *output_grads) {
  RELU_Layer *rv = (RELU_Layer *)malloc(sizeof(RELU_Layer));
  rv->output = output;
  rv->output_grads = output_grads;
  return rv;
}

void calc_layer_output(Layer *layer, Layer *input_layer) {
  Matrix *input = input_layer->output;
  if (input->columns != layer->weights->rows) {
    fprintf(stderr, "inproper matrix multiplication");
    exit(1);
  }
  layer->output = matmul(input, layer->weights);
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
  return weight_grads;
}

void calc_layer_output_gradients(Layer *layer, Layer *above_layer) {
  Matrix *weight_grads_above = weights_gradients_subtotal(above_layer);
  unsigned int rv_cols = above_layer->weights->rows - 1;
  Matrix *output_grads = allocate_empty(1, rv_cols);

  for (unsigned int i = 0; i < rv_cols; i++) {
    for (unsigned int j = 0; j < weight_grads_above->columns; j++) {
      output_grads->values[0][i] += weight_grads_above->values[i][j];
    }
  }

  layer->output_grads = output_grads;
}

void calc_RELU_layer(RELU_Layer *relu, Matrix *input) {
  Matrix *output = allocate_empty(1, input->columns);

  for (unsigned int i = 0; i < input->columns; i++) {
    if (input->values[0][i] > 0) {
      output->values[0][i] = input->values[0][i];
    } else {
      output->values[0][i] = 0;
    }
  }
  relu->output = output;
}

void calc_layer_gradients_from_RELU(Layer *input_layer, RELU_Layer* RELU_Layer) {
  Matrix *grads = allocate_empty(1, input_layer->output->columns - 1);

  for (unsigned int i = 0; i < grads->columns; i++) {
    if (input_layer->output->values[0][i] > 0) {
      grads->values[0][i] = RELU_Layer->output_grads->values[0][i];
    } else {
      grads->values[0][i] = 0;
    }
  }
  input_layer->output_grads = grads;
}

/**
 * NOTE: below functions are only for debugging purposes and printing small
 * layers. They are not suitable for printing out large layers of networks to
 * the console.
 */

void print_layer(Layer *layer, char *layer_name) {
  printf("*****************************************************************\n");
  printf("                      LAYER: %s\n", layer_name);
  printf("*****************************************************************\n");
  printf("OUTPUT MATRIX:\n");
  if (layer->output != NULL)
    print_matrix_verbose(layer->output);
  else
    printf("--> NULL pointer\n");

  putchar('\n');
  printf("OUTPUT GRADIENTS:\n");
  if (layer->output_grads != NULL)
    print_matrix_verbose(layer->output_grads);
  else
    printf("--> NULL pointer\n");

  putchar('\n');
  printf("WEIGHTS:\n");
  if (layer->weights != NULL)
    print_matrix_verbose(layer->weights);
  else
    printf("--> NULL pointer\n");

  putchar('\n');
  printf("WEIGHTS GRADIENTS:\n");
  if (layer->weight_grads != NULL)
    print_matrix_verbose(layer->weight_grads);
  else
    printf("--> NULL pointer\n");
  
  printf("_________________________________________________________________\n\n");
}

void print_RELU_layer(RELU_Layer *layer, char *layer_name) {
  printf("*****************************************************************\n");
  printf("                      LAYER: %s\n", layer_name);
  printf("*****************************************************************\n");
  printf("OUTPUT MATRIX:\n");
  if (layer->output != NULL)
    print_matrix_verbose(layer->output);
  else
    printf("--> NULL pointer\n");

  putchar('\n');
  printf("OUTPUT GRADIENTS:\n");
  if (layer->output_grads != NULL)
    print_matrix_verbose(layer->output_grads);
  else
    printf("--> NULL pointer\n");
  
  printf("_________________________________________________________________\n\n");
}