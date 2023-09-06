#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double LEARNING_RATE = 0.001;

void gradient_descent(Layer *layer1,
                      Layer *layer2,
                      Layer *output_layer)
{
  gradient_descent_on_layer(output_layer, LEARNING_RATE);
  gradient_descent_on_layer(layer2, LEARNING_RATE);
  gradient_descent_on_layer(layer1, LEARNING_RATE);
}

int main() {
  unsigned int NUM_IMGS = 60000;

  Matrix **imgs = load_MNIST_images("TRAINING_DATA/x_train.txt", NUM_IMGS);
  Matrix **flattened_imgs = (Matrix **)calloc(NUM_IMGS, sizeof(Matrix *));
  for (unsigned int i = 0; i < NUM_IMGS; i++) {
    flattened_imgs[i] = flatten_matrix(imgs[i]);
  }

  // original images are no longer needed because model uses the flattened form
  free_matrix_array(imgs, NUM_IMGS);

  double *labels = load_MNIST_lables("TRAINING_DATA/y_train.txt", NUM_IMGS);
  Layer *input_layer = init_layer(NULL, NULL, NULL, NULL);

  free(labels);
  free(matrix_arr);
  free(input_layer); // output matrix for this layer has already been freed
  return 0;
}