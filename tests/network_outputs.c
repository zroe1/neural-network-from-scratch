#include <criterion/criterion.h>
#include "../network.h"

/* test for relu layer to keep positive values unchanged */
Test(ouputs, outputs00)
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

  Matrix *in = allocate_from_2D_arr(1, 2, input_arr);
  Layer *l = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), NULL); 
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));

  calc_layer_output(l, in);
  calc_RELU_layer(relu, l->output);
  cr_assert(l->output->values[0][0] == 4);
  cr_assert(relu->output->values[0][0] == 4);

  free_RELU_layer(relu);
  free_layer(l);
}

/* test for relu layer to turn negative to zero */
Test(ouputs, outputs01)
{
  double input_arr[1][2] = {
    {2, 1}
  };
  double weights_arr[2][2] = {
    {-2, 0},
    {0, 1}
  };
  double output_grad[1][1] = {
    {1}
  };

  Matrix *in = allocate_from_2D_arr(1, 2, input_arr);
  Layer *l = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), NULL); 
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));

  calc_layer_output(l, in);
  calc_RELU_layer(relu, l->output);
  cr_assert(l->output->values[0][0] == -4);
  cr_assert(relu->output->values[0][0] == 0);

  free_RELU_layer(relu);
  free_layer(l);
}