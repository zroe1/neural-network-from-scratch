#include <criterion/criterion.h>
#include "../network.h"

Test(ouputs, outputs1)
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

  Layer *l = init_layer(NULL, NULL, allocate_from_2D_arr(2, 2, weights_arr), NULL); 
  RELU_Layer *relu = init_RELU_layer(NULL, allocate_from_2D_arr(1, 1, output_grad));

  calc_layer_output(l, allocate_from_2D_arr(1, 2, input_arr));
  cr_assert(l->output->values[0][0] == 4);

  free_RELU_layer(relu);
  free_layer(l);
}