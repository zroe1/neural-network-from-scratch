#include <criterion/criterion.h>
#include "../network.h"

Test(zero_grad, zeros0)
{
  Matrix *weights = init_random_weights(5, 5);
  zero_gradients(weights);

  for (unsigned int i = 0; i < weights->rows; i++) {
    for (unsigned int j = 0; j < weights->columns; j++) {
      if (weights->values[i][j] != 0) {
        cr_assert(0);
      }
    }
  }
  cr_assert(1);
}