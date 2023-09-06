#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double LEARNING_RATE = 0.00001;

void gradient_descent(Layer *layer1, Layer *layer2) {
  // gradient_descent_on_layer(layer2, LEARNING_RATE);
  gradient_descent_on_layer(layer1, LEARNING_RATE);
}

int main() {
  unsigned int NUM_IMGS = 60000;

  Matrix **imgs = load_MNIST_images("TRAINING_DATA/x_train.txt", NUM_IMGS);
  Matrix **flattened_imgs = (Matrix **)calloc(NUM_IMGS, sizeof(Matrix *));
  for (unsigned int i = 0; i < NUM_IMGS; i++) {
    flattened_imgs[i] = flatten_matrix_and_append_one(imgs[i]);
  }

  // original images are no longer needed because model uses the flattened form
  free_matrix_array(imgs, NUM_IMGS);

  double *labels = load_MNIST_lables("TRAINING_DATA/y_train.txt", NUM_IMGS);
  Layer *in = init_layer(NULL, NULL, NULL, NULL);
  Layer *l1 = init_layer(NULL, NULL, init_random_weights(785, 129), allocate_empty(785, 128));
  RELU_Layer *relu = init_RELU_layer(NULL, NULL);
  Layer *l2 = init_layer(NULL, NULL, init_random_weights(129, 11), allocate_empty(129, 10));
  Squish_Layer *squish = init_squish_layer(NULL, NULL);

  in->output = flattened_imgs[0];

  // calc_mean_squared_loss(double output, double correct)
  forward_pass(in, l1, relu, l2, squish);
  backward_pass(in, l1, relu, l2, squish, 1);
  // print_layer(l2, "layer2");
  // print_squish_layer(squish, "squish");

  double correct = 1;
  double loss = 0;
  for (unsigned int i = 0; i < 10; i++) {
    if (i + 1 == correct) {
      loss += calc_mean_squared_loss(squish->output->values[0][i], 1);
    } else {
      loss += calc_mean_squared_loss(squish->output->values[0][i], 0);
    }
  }
  printf("LOSS: %f\n", loss);

  // print_matrix(l1->weight_grads);
  // print_matrix(l2->weight_grads);
  gradient_descent(l1, l2);
  forward_pass(in, l1, relu, l2, squish);

  loss = 0;
  for (unsigned int i = 0; i < 10; i++) {
    if (i + 1 == correct) {
      loss += calc_mean_squared_loss(squish->output->values[0][i], 1);
    } else {
      loss += calc_mean_squared_loss(squish->output->values[0][i], 0);
    }
  }
  printf("LOSS: %f\n", loss);

  free(labels);
  free_matrix_array(flattened_imgs, NUM_IMGS);
  free(in); // output matrix for this layer has already been freed
  return 0;
}