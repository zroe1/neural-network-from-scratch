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

Test(input_grads, ingrad00) {
  double weights_arr[2][2] = {
    {2, 0},
    {0, 1}
  };

  double output_grads[1][1] = {
    {1}
  };

  Layer *in = init_layer(NULL, NULL, NULL, NULL);
  Layer *l = init_layer(NULL, allocate_from_2D_arr(1, 1, output_grads), allocate_from_2D_arr(2, 2, weights_arr), NULL);

  in->output_grads = calc_layer_input_gradients(l, in->output_grads);
  cr_assert(in->output_grads->values[0][0] == 2);

  free_layer(in);
  free_layer(l);
}

Test(input_grads, ingrad01)
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
  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), NULL, NULL, NULL);
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

  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), NULL, NULL, NULL);
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

  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), NULL, NULL, NULL);
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));
  
  calc_RELU_layer(relu, in->output);
  calc_layer_gradients_from_RELU(in, relu->output_grads);

  cr_assert(in->output_grads->values[0][0] == 0);
  cr_assert(relu->output_grads->values[0][0] == 1);

  free_layer(in);
  free_RELU_layer(relu);
}

/* 
 * NOTE: Loss tests are inlcuded in this file because loss is used to calculate 
 * all gradients in the network
 */

Test(loss_test, loss00)
{
  double loss = calc_mean_squared_loss(0.5, 1);
  cr_assert(loss - 0.25 < 0.0001);
}

Test(loss_test, loss01)
{
  double loss = calc_mean_squared_loss(0.5, 0);
  cr_assert(loss - 0.25 < 0.0001);
}

Test(loss_grad, lossgrad00)
{
  double grad = calc_grad_of_input_to_loss(0.5, 0);
  cr_assert(grad - 1 < 0.0001);
}

Test(loss_grad, lossgrad01)
{
  double grad = calc_grad_of_input_to_loss(0.5, 1);
  cr_assert(grad + 1 < 0.0001);
}

Test(loss_grad, loss_grad2)
{
  double grad = calc_grad_of_input_to_loss(0, 1);
  cr_assert(grad + 2 < 0.0001);
}

Test(loss_grad, loss_grad3)
{
  double grad = calc_grad_of_input_to_loss(1, 1);
  cr_assert(grad < 0.0001);
}

Test(loss_grad, loss_grad4)
{
  double grad = calc_grad_of_input_to_loss(1, 1);
  cr_assert(grad < 0.0001);
}

Test(loss_grad, loss_grad5)
{
  double input_arr[1][2] = {
    {-1, 1}
  };
  double weights[2][2] = {
    {-0.25, 0},
    {0, 1}
  };

  Matrix *in = allocate_from_2D_arr(1, 2, input_arr);
  Layer *l = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights), allocate_empty(2, 1));
  calc_layer_output(l, in);

  double output_grad = calc_grad_of_input_to_loss(l->output->values[0][0], 1);
  double output_grad_arr[1][1] = {
    {output_grad}
  };
  l->output_grads = allocate_from_2D_arr(1, 1, output_grad_arr);
  calc_weight_gradients(l, in);

  cr_assert(l->weight_grads->values[0][0] - 1.5 < 0.0001);
  cr_assert(l->output_grads->values[0][0] + 1.5 < 0.0001);

  free_matrix(in);
  free_layer(l);
}

Test(squish, squish00)
{
  double input_arr[1][3] = {
    {1, 3, 1}
  };
  double output_grads[1][2] = {
    {1, 1}
  };

  Layer *in = init_layer(allocate_from_2D_arr(1, 3, input_arr), NULL, NULL, NULL);
  Squish_Layer *l = init_squish_layer(NULL, allocate_from_2D_arr(1, 2, output_grads));
  calc_squish_layer(l, allocate_from_2D_arr(1, 3, input_arr));
  calc_layer_gradients_from_squish(in, l);

  cr_assert(l->output->values[0][0] - 0.25 < 0.0001);
  cr_assert(l->output->values[0][1] - 0.75 < 0.0001);

  cr_assert(in->output_grads->values[0][0] - 0.1875 < 0.001);
  cr_assert(in->output_grads->values[0][1] - 0.0625 < 0.001);
  free_squish_layer(l);
}