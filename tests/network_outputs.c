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

Test(squish, squish00)
{
  double input_arr[1][3] = {
    {1, 3, 1}
  };
  Squish_Layer *l = init_squish_layer(NULL, NULL);
  calc_squish_layer(l, allocate_from_2D_arr(1, 3, input_arr));

  cr_assert(l->output->values[0][0] - 0.25 < 0.0001);
  cr_assert(l->output->values[0][1] - 0.75 < 0.0001);

  free_squish_layer(l);
}

Test(squish, squish01)
{
  double input_arr[1][3] = {
    {0, 0, 1}
  };
  Squish_Layer *l = init_squish_layer(NULL, NULL);
  calc_squish_layer(l, allocate_from_2D_arr(1, 3, input_arr));

  cr_assert(l->output->values[0][0] - 0 < 0.0001);
  cr_assert(l->output->values[0][1] - 0 < 0.0001);

  free_squish_layer(l);
}

Test(squish, squish02)
{
  double input_arr[1][5] = {
    {1, 1, 1, 1, 1}
  };
  Squish_Layer *l = init_squish_layer(NULL, NULL);
  calc_squish_layer(l, allocate_from_2D_arr(1, 5, input_arr));

  cr_assert(l->output->values[0][0] - 1 < 0.0001);
  cr_assert(l->output->values[0][1] - 1 < 0.0001);
  cr_assert(l->output->values[0][2] - 1 < 0.0001);
  cr_assert(l->output->values[0][3] - 1 < 0.0001);
  cr_assert(l->output->values[0][4] - 1 < 0.0001);

  free_squish_layer(l);
}

Test(forward_pass, forward00)
{
  double input_arr[1][2] = {
    {2, 1}
  };
  double weights_arr[2][2] = {
    {-2, 0},
    {1, 1}
  };

  Layer *in = init_layer(allocate_from_2D_arr(1, 2, input_arr), NULL, NULL, NULL);
  Layer *l1 = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), NULL);
  RELU_Layer *relu = init_RELU_layer(NULL, NULL);
  Layer *l2 = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), NULL);
  Squish_Layer *s = init_squish_layer(NULL, NULL);

  forward_pass(in, l1, relu, l2, s);

  cr_assert(l1->output->values[0][0] == -3);
  cr_assert(relu->output->values[0][0] == 0);
  cr_assert(l2->output->values[0][0] == 1);
  cr_assert(s->output->values[0][0] == 1);

  free_layer(in);
  free_layer(l1);
  free_RELU_layer(relu);
  free_layer(l2);
  free_squish_layer(s);
}