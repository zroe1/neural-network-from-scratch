#include <stdlib.h>
#include "network.h"

int main() {
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
  Layer *layer2 = init_layer(NULL, NULL, init_random_weights(5, 5), NULL);
  Layer *output_layer = init_layer(NULL, NULL, init_random_weights(5, 2), NULL);

  calc_layer_output(layer1, inputs[0]->output);
  calc_layer_output(layer2, layer1->output);
  calc_layer_output(output_layer, layer2->output);

  print_layer(output_layer, "output");
  return 0;
}