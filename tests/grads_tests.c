#include <criterion/criterion.h>
#include "../network.h"

Test(zero_grad, zeros0)
{
  Matrix *weights = init_random_weights(5, 5);
  zero_gradients(weights);

  for (unsigned int i = 0; i < weights->rows; i++) {
    for (unsigned int j = 0; j < weights->columns; j++) {
      if (weights->values[i][j] != 0) {
        free_matrix(weights);
        cr_assert(0);
      }
    }
  }
  free_matrix(weights);
  cr_assert(1);
}

Test(input_grads, ingrad00)
{
  double input_arr[1][2] = {
    {2, 1}
  };
  double weights_arr[2][2] = {
    {2, 0},
    {0, 1}
  };
  double output_grad[1][1] = {
    {1}
  };
  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), allocate_empty(1, 1), NULL, NULL);
  Layer *l = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), allocate_empty(2, 1)); 
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));

  calc_layer_output(l, in->output);
  calc_RELU_layer(relu, l->output);
  calc_layer_gradients_from_RELU(l, relu->output_grads);
  calc_weight_gradients(l, in->output);
  in->output_grads = calc_layer_input_gradients(l, in->output_grads);

  cr_assert(relu->output_grads->values[0][0] == 1);
  cr_assert(l->output_grads->values[0][0] == 1);
  cr_assert(l->weight_grads->values[0][0] == 2);
  cr_assert(in->output_grads->values[0][0] == 2);

  free_layer(in);
  free_layer(l);
  free_RELU_layer(relu);
}

Test(relu_grads, relu00)
{
  double input_arr[1][2] = {
    {2, 1}
  };
  double output_grad[1][1] = {
    {1}
  };

  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), allocate_empty(1, 1), NULL, NULL);
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));
  
  calc_RELU_layer(relu, in->output);
  calc_layer_gradients_from_RELU(in, relu->output_grads);

  cr_assert(in->output_grads->values[0][0] == 1);
  cr_assert(relu->output_grads->values[0][0] == 1);

  free_layer(in);
  free_RELU_layer(relu);
}

/* tests for zero gradient in layer before relu */
Test(relu_grads, relu01)
{
  double input_arr[1][2] = {
    {-2, 1}
  };
  double output_grad[1][1] = {
    {1}
  };

  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), allocate_empty(1, 1), NULL, NULL);
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));
  
  calc_RELU_layer(relu, in->output);
  calc_layer_gradients_from_RELU(in, relu->output_grads);

  cr_assert(in->output_grads->values[0][0] == 0);
  cr_assert(relu->output_grads->values[0][0] == 1);

  free_layer(in);
  free_RELU_layer(relu);
}