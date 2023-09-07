#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double *load_MNIST_lables(char *filename, unsigned int num_lines) {
  FILE* file = fopen(filename, "r");
  double *rv = (double *)malloc(sizeof(double) * num_lines);
  unsigned int counter = 0;

  char *end_pointer;
  char line[3];

  while (fgets(line, sizeof(line), file)) {
    line[1] = '\0';
    rv[counter] = strtod(line, &end_pointer);

    if (*end_pointer != '\0') {
      printf("ERROR: line read incorrectly\n");
      exit(1);
    }
    counter++;
  }

  if (counter != num_lines) {
    free(rv);
    printf("ERROR: read incorrect number of lines.\n");
    exit(1);
  }
  return rv;
}

Matrix **load_MNIST_images(char *filename, unsigned int num_imgs) {
  unsigned int LINES_PER_IMG = 28;

  FILE* file = fopen(filename, "r");
  // initialized with calloc so it is possible to check for NULL pointers
  Matrix **rv = (Matrix **)calloc(num_imgs, sizeof(Matrix *));
  rv[0] = allocate_empty(28, 28);
  unsigned int current_img = 0;
  unsigned int current_img_line = 0;
  unsigned int current_img_col = 0;

  char *end_pointer;
  char line[100];
  char num_temp[5];
  unsigned int num_temp_idx = 0;

  while (fgets(line, sizeof(line), file)) {
    // a line with only a newline indicates a new image
    if (line[0] == '\n') {
      current_img_line = 0;
      current_img++;
      rv[current_img] = allocate_empty(28, 28);

      // else statement reads a line from an image
    } else {
      current_img_col = 0;
      unsigned int current_line_idx = 0;

      while (current_img_col < 28){
        char current_char = line[current_line_idx];

        if (current_char == ' ') {
          num_temp[num_temp_idx] = '\0';
          num_temp_idx = 0;
          double current_num = strtod(num_temp, &end_pointer);

          if (*end_pointer != '\0') {
            printf("ERROR: line read incorrectly\n");
            printf("*%s*\n", num_temp);
            exit(1);
          }

          rv[current_img]->values[current_img_line][current_img_col++] = current_num;
        } else {
          num_temp[num_temp_idx++] = current_char;
        }
        current_line_idx++;
      }
      current_img_line++;
    }
  }
  if (current_img != num_imgs) {
    printf("ERROR: read incorrect number of images.\n");
    printf("Check \"num_imgs\" passed into function call\n");
    exit(1);
  }
  return rv;
}

void free_matrix_array(Matrix **matrix_arr, unsigned int len) {
  for (unsigned int i = 0; i< len; i++) {
    if (matrix_arr[i] != NULL) {
      free_matrix(matrix_arr[i]);
    }
  }
  free(matrix_arr);
}

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

void free_squish_layer(Squish_Layer *l) {
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

  // if rv matrix passed in is NULL, a new one is created to be returned
  if (rv == NULL) {
    rv = allocate_empty(1, rv_cols);
  }  else {
    // zeros out previous gradients
    for (unsigned int i = 0; i < rv->columns; i++) {
      rv->values[0][i] = 0;
    }
  }

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

  Matrix *grads = input_layer->output_grads;
  // if input_layer->output_grads doesn't exist, a new one must be created
  if (grads == NULL) {
    grads = allocate_empty(1, input_layer->output->columns - 1);
  } else {
    // otherwise, gradients are zeroed
    for (unsigned int i = 0; i < grads->columns - 1; i++) {
      grads->values[0][i] = 0;
    }
  }

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
/* Assumes weight gradients for "layer" have been initialized */
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

  if (layer->weight_grads == NULL) {
    fprintf(stderr, "ERROR: layer->weight_grads is NULL\n");
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
}

void gradient_descent_on_layer(Layer *layer, double learning_rate) {
  Matrix *weight_grads = layer->weight_grads;
  Matrix *weights = layer->weights;

  for (unsigned int i = 0; i < weight_grads->rows; i++) {
    for (unsigned int j = 0; j < weight_grads->columns; j++) {
      weights->values[i][j] -= weight_grads->values[i][j] * learning_rate;
    }
  }
}

void forward_pass(Layer *input_layer,
                  Layer *layer1,
                  RELU_Layer *layer1_RELU,
                  Layer *layer2,
                  Squish_Layer *squish)
{
  calc_layer_output(layer1, input_layer->output);
  calc_RELU_layer(layer1_RELU, layer1->output);
  calc_layer_output(layer2, layer1_RELU->output);
  calc_squish_layer(squish, layer2->output);
}

void backward_pass(Layer *input_layer,
                   Layer *layer1,
                   RELU_Layer *layer1_RELU,
                   Layer *layer2,
                   Squish_Layer *squish,
                   double correct)
{
  // checks if the output gradients for the squish layer aren't initalialized
  if (squish->output_grads == NULL) {
    squish->output_grads = allocate_empty(1, 10);
  }
  
  for (unsigned int i = 0; i < squish->output->columns; i++) {
    double output_grad;
    if (i + 1 == correct) {
      output_grad = calc_grad_of_input_to_loss(squish->output->values[0][i], 1);
    } else {
      output_grad = calc_grad_of_input_to_loss(squish->output->values[0][i], 0);
    }
    squish->output_grads->values[0][i] = output_grad;
  }

  calc_layer_gradients_from_squish(layer2, squish);
  calc_weight_gradients(layer2, layer1_RELU->output);
  layer1_RELU->output_grads = calc_layer_input_gradients(layer2, layer1_RELU->output_grads);
  calc_layer_gradients_from_RELU(layer1, layer1_RELU->output_grads);
  calc_weight_gradients(layer1, input_layer->output);
}

double calc_mean_squared_loss(double output, double correct) {
  return (output - correct) * (output - correct);
}

/* Assumes loss used is mean squared */
double calc_grad_of_input_to_loss(double output, double correct) {
  return 2.0 * (output - correct);
}

Squish_Layer *init_squish_layer(Matrix *output, Matrix *output_grads) {
  Squish_Layer *rv = (Squish_Layer *)malloc(sizeof(Squish_Layer));
  rv->output_sum = 0;
  rv->output = output;
  rv->output_grads = output_grads;
  return rv;
}

void calc_squish_layer(Squish_Layer *layer, Matrix *inputs) {
  if (inputs->rows > 1) {
    fprintf(stderr, "ERROR: input matrix is not 1 * n.\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  double min_input = 0;
  for (unsigned int i = 0; i < inputs->columns - 1; i++) {
    if (inputs->values[0][i] < min_input) {
      min_input = inputs->values[0][i];
    }
  }

  free(layer->output);
  Matrix *output = allocate_empty(inputs->rows, inputs->columns);
  if (min_input < 0) {
    double to_add = -min_input;
    for (unsigned int i = 0; i < inputs->columns - 1; i++) {
      output->values[0][i] = inputs->values[0][i] + to_add;
    }
  } else {
    for (unsigned int i = 0; i < inputs->columns - 1; i++) {
      output->values[0][i] = inputs->values[0][i];
    }
  }

  double output_sum = 0;
  for (unsigned int i = 0; i < output->columns - 1; i++) {
    output_sum += output->values[0][i];
  }
  
  if (output_sum == 0) {
    output_sum = 1; // this is to prevent divide by 0 error
  }
  for (unsigned int i = 0; i < output->columns - 1; i++) {
    output->values[0][i] = output->values[0][i] / output_sum;
  }
  output->values[0][output->columns - 1] = 1;
  layer->output = output;
  layer->output_sum = output_sum;
}

void calc_layer_gradients_from_squish(Layer *input_layer, Squish_Layer *squish) {
  if (squish->output_grads->columns != input_layer->output->columns -1) {
    fprintf(stderr, "ERROR: squish gradients != number of input layer outputs\n");
    fprintf(stderr, "(EXIT EARLY)\n");
    exit(1);
  }

  Matrix *grads;
  if (input_layer->output_grads == NULL) {
    grads = allocate_empty(1, input_layer->output->columns - 1);
  } else {
    /* intermediate neuron activations aren't added up over backward passes */
    grads = input_layer->output_grads;
    for (unsigned int i = 0; i < grads->rows; i++) {
      for (unsigned int j = 0; j < grads->columns; j++) {
        grads->values[i][j] = 0;
      }
    }
  }

  for (unsigned int i = 0; i < grads->columns; i++) {
    double squish_grad = squish->output_grads->values[0][i];
    double layer_total = squish->output_sum;
    double input_val = input_layer->output->values[0][i];
    double temp = (layer_total - input_val) / (layer_total * layer_total);
    grads->values[0][i] += temp * squish_grad;
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

void print_squish_layer(Squish_Layer *layer, char *layer_name) {
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
  printf("OUTPUT SUM:\n");
  printf("--> %f\n", layer->output_sum);
  
  printf("_________________________________________________________________\n\n");
}