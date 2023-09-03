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

void free_layer(Layer *l) {
  if (l->output != NULL) {
    free_matrix(l->output);
  }
  if (l->output_grads != NULL) {
    free_matrix(l->output_grads);
  }
  if (l->weights != NULL) {
    free_matrix(l->weights);
  }
  if (l->weight_grads != NULL) {
    free_matrix(l->weight_grads);
  }
  free(l);
}

void free_RELU_layer(RELU_Layer *l) {
  if (l->output != NULL) {
    free_matrix(l->output);
  }
  if (l->output_grads != NULL) {
    free_matrix(l->output_grads);
  }
  free(l);
}

Matrix *init_random_weights(unsigned int rows, unsigned int cols) {
  Matrix *rv = allocate_empty(rows, cols);

  /* rightmost row is always set as zeros with one 1.0 at the bottom to append a 
    1 to output */
  for (unsigned int i = 0; i < rows - 1; i++) {
    rv->values[i][cols - 1] = 0.0;
  }
  rv->values[rows - 1][cols - 1] = 1.0;

  // rest of values are set to a random double between -1 and 1
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols - 1; j++) {
      double val = (double)rand() / RAND_MAX;
      if (rand() % 2 == 0) {
        val *= -1;
      }
      rv->values[i][j] = val;
    }
  }
  return rv;
}

void zero_gradients(Matrix *grads) {
  unsigned int rows = grads->rows;
  unsigned int cols = grads->columns;

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      grads->values[i][j] = 0;
    }
  }
}

void calc_layer_output(Layer *layer, Matrix *input) {
  if (input->columns != layer->weights->rows) {
    fprintf(stderr, "inproper matrix multiplication");
    exit(1);
  }
  layer->output = matmul(input, layer->weights);
}

Matrix *weights_gradients_subtotal(Layer *layer) {
  unsigned int rows = layer->weights->rows - 1;
  unsigned int cols = layer->weights->columns - 1;

  if (layer->output_grads->columns != cols) {
    fprintf(stderr, "ERROR: ouput gradients != number of sets of weights.\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  double rv_arr[rows][cols];
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      double output_grad = layer->output_grads->values[0][j];
      rv_arr[i][j] = layer->weights->values[i][j] * output_grad;
    }
  }
  return allocate_from_2D_arr(rows, cols, rv_arr);
}

Matrix *calc_layer_input_gradients(Layer *above_layer, Matrix *rv) {
  Matrix *subtotals = weights_gradients_subtotal(above_layer);
  unsigned int rv_cols = above_layer->weights->rows - 1; // -1 to ignore bias values
  // Matrix *output_grads = allocate_empty(1, rv_cols);

  for (unsigned int i = 0; i < rv_cols; i++) {
    for (unsigned int j = 0; j < subtotals->columns; j++) {
      rv->values[0][i] += subtotals->values[i][j];
    }
  }

  return rv;
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

void calc_layer_gradients_from_RELU(Layer *input_layer, Matrix *RELU_grads) {
  if (RELU_grads->columns != input_layer->output->columns -1) {
    fprintf(stderr, "ERROR: RELU gradients != number of input layer outputs\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  Matrix *grads = allocate_empty(1, input_layer->output->columns - 1);

  for (unsigned int i = 0; i < grads->columns; i++) {
    if (input_layer->output->values[0][i] > 0) {
      grads->values[0][i] += RELU_grads->values[0][i];
    } else {
      grads->values[0][i] += 0;
    }
  }
  input_layer->output_grads = grads;
}

/* Assumes layer ouput gradients have been set correctly */
void calc_weight_gradients(Layer *layer, Matrix *layer_inputs) {
  if (layer_inputs->columns != layer->weights->rows) {
    fprintf(stderr, "ERROR: inputs != number of wieghts per set\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  if (layer->output_grads->columns != layer->weights->columns - 1) {
    fprintf(stderr, "ERROR: output gradients != sets of weights\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  unsigned int rv_rows = layer->weights->rows;
  unsigned int rv_cols = layer->weights->columns - 1;

  Matrix *rv = layer->weight_grads;

  for (unsigned int i = 0; i < layer_inputs->columns; i++) {
    for (unsigned int j = 0; j < layer->output_grads->columns; j++) {
      if (i == layer_inputs->columns - 1) {
        rv->values[i][j] += layer->output_grads->values[0][j];
      } else {
        rv->values[i][j] += layer_inputs->values[0][i] * layer->output_grads->values[0][j];
      }
    }
  }
  // layer->weight_grads = rv;
}

void gradient_descent_on_layer(Layer *layer, double learning_rate) {
  Matrix *weight_grads = layer->weight_grads;
  Matrix *weights = layer->weights;

  for (unsigned int i = 0; i < weight_grads->columns; i++) {
    for (unsigned int j = 0; j < weight_grads->rows; j++) {
      weights->values[i][j] -= weight_grads->values[i][j] * learning_rate;
    }
  }
}

void forward_pass(Layer *input_layer,
                  Layer *layer1,
                  RELU_Layer *layer1_RELU,
                  Layer *layer2,
                  RELU_Layer *layer2_RELU,
                  Layer *output_layer)
{
  calc_layer_output(layer1, input_layer->output);
  calc_RELU_layer(layer1_RELU, layer1->output);
  calc_layer_output(layer2, layer1_RELU->output);
  calc_RELU_layer(layer2_RELU, layer2->output);
  calc_layer_output(output_layer, layer2_RELU->output);
}

void backward_pass(Layer *input_layer,
                  Layer *layer1,
                  RELU_Layer *layer1_RELU,
                  Layer *layer2,
                  RELU_Layer *layer2_RELU,
                  Layer *output_layer,
                  double loss,
                  double correct)
{
  /* calculates the gradient for the output */
  double output_grad = loss / output_layer->output->values[0][0];
  if (output_layer->output->values[0][0] - correct < 0) {
    if (output_grad > 0)
      output_grad *= -1;
  }
  double output_grads[1][1] = {{output_grad}};
  output_layer->output_grads = allocate_from_2D_arr(1, 1, output_grads);

  calc_weight_gradients(output_layer, layer2_RELU->output);

  layer2_RELU->output_grads = calc_layer_input_gradients(output_layer, layer2_RELU->output_grads);
  calc_layer_gradients_from_RELU(layer2, layer2_RELU->output_grads);
  calc_weight_gradients(layer2, layer1_RELU->output);
  
  layer1_RELU->output_grads = calc_layer_input_gradients(layer2, layer1_RELU->output_grads);
  calc_layer_gradients_from_RELU(layer1, layer1_RELU->output_grads);
  calc_weight_gradients(layer1, input_layer->output);
}

double calc_mean_squared_loss(double output, double correct) {
  return (output - correct) * (output - correct);
}

double calc_grad_of_input_to_loss(double output, double correct) {
  return 2.0 * (output - correct);
}

void calc_squish_layer(Squish_Layer *layer, Matrix *inputs) {
  if (inputs->rows > 1) {
    fprintf(stderr, "ERROR: input matrix is not 1 * n.\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  double min_input = inputs->values[0][0];
  for (unsigned int i = 0; i < inputs->columns; i++) {
    if (inputs->values[0][i] < min_input) {
      min_input = inputs->values[0][i];
    }
  }

  free(layer->ouput);
  Matrix *output = allocate_empty(inputs->rows, inputs->columns);
  if (min_input < 0) {
    double to_add = -min_input;
    for (unsigned int i = 0; i < inputs->columns; i++) {
      output->values[0][i] = inputs->values[0][i] + to_add;
    }
  }

  double output_sum = 0;
  for (unsigned int i = 0; i < output->columns; i++) {
    output_sum += output->values[0][i];
  }
  for (unsigned int i = 0; i < output->columns; i++) {
    output->values[0][i] /= output_sum;
  }

  layer->output_sum = output_sum;
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