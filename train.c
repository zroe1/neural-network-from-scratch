#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double LEARNING_RATE = 0.001;

void gradient_descent_on_layer(Layer *layer, double learning_rate) {
  Matrix *weight_grads = layer->weight_grads;
  Matrix *weights = layer->weights;

  for (unsigned int i = 0; i < weight_grads->columns; i++) {
    for (unsigned int j = 0; j < weight_grads->rows; j++) {
      weights->values[i][j] -= weight_grads->values[i][j] * learning_rate;
    }
  }
}

int main() {
  double ys[4] = {1, -1, -1, 1};

  double x0[1][4] = {
    {2, 3, -1, 1}
  };
  double x1[1][4] = {
    {3, -1, 0.5, 1}
  };
  double x2[1][4] = {
    {0.5, 1.0, 1.0, 1}
  };
  double x3[1][4] = {
    {1.0, 1.0, -1, 1}
  };

  Layer **inputs = (Layer **)malloc(sizeof(Layer *) * 4);

  inputs[0] = init_layer(allocate_from_2D_arr(1, 4, x0), NULL, NULL, NULL);
  inputs[1] = init_layer(allocate_from_2D_arr(1, 4, x1), NULL, NULL, NULL);
  inputs[2] = init_layer(allocate_from_2D_arr(1, 4, x2), NULL, NULL, NULL);
  inputs[3] = init_layer(allocate_from_2D_arr(1, 4, x3), NULL, NULL, NULL);
  
  Layer *layer1 = init_layer(NULL, NULL, init_random_weights(4, 5), NULL);
  RELU_Layer *layer1_RELU = init_RELU_layer(NULL, NULL);
  Layer *layer2 = init_layer(NULL, NULL, init_random_weights(5, 5), NULL);
  RELU_Layer *layer2_RELU = init_RELU_layer(NULL, NULL);
  Layer *output_layer = init_layer(NULL, NULL, init_random_weights(5, 2), NULL);

  calc_layer_output(layer1, inputs[0]->output);
  calc_RELU_layer(layer1_RELU, layer1->output);
  calc_layer_output(layer2, layer1_RELU->output);
  calc_RELU_layer(layer2_RELU, layer2->output);
  calc_layer_output(output_layer, layer2_RELU->output);

  double loss = (output_layer->output->values[0][0] - ys[0]) * (output_layer->output->values[0][0] - ys[0]);
  printf("loss: %f\n", loss);

  /* calculates the gradient for the output */
  double output_grad = loss / output_layer->output->values[0][0];
  if (output_layer->output->values[0][0] - ys[0] < 0) {
    if (output_grad > 0)
      output_grad *= -1;
  }
  double output_grads[1][1] = {{output_grad}};
  output_layer->output_grads = allocate_from_2D_arr(1, 1, output_grads);

  calc_weight_gradients(output_layer, layer2_RELU->output);
  gradient_descent_on_layer(output_layer, LEARNING_RATE);

  layer2_RELU->output_grads = calc_layer_input_gradients(output_layer);
  calc_layer_gradients_from_RELU(layer2, layer2_RELU->output_grads);
  calc_weight_gradients(layer2, layer1_RELU->output);
  gradient_descent_on_layer(layer2, LEARNING_RATE);
  
  layer1_RELU->output_grads = calc_layer_input_gradients(layer2);
  calc_layer_gradients_from_RELU(layer1, layer1_RELU->output_grads);
  calc_weight_gradients(layer1, inputs[0]->output);
  gradient_descent_on_layer(layer1, LEARNING_RATE);


  calc_layer_output(layer1, inputs[0]->output);
  calc_RELU_layer(layer1_RELU, layer1->output);
  calc_layer_output(layer2, layer1_RELU->output);
  calc_RELU_layer(layer2_RELU, layer2->output);
  calc_layer_output(output_layer, layer2_RELU->output);
  loss = (output_layer->output->values[0][0] - ys[0]) * (output_layer->output->values[0][0] - ys[0]);
  printf("loss: %f\n", loss);

  return 0;
}