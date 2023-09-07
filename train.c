#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double LEARNING_RATE = 0.002;

void gradient_descent(Layer *layer1, Layer *layer2) {
  gradient_descent_on_layer(layer2, LEARNING_RATE);
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

  // unsigned int current_img = 0;

  for (unsigned int epoch = 0; epoch < 20; epoch++) {
    double epoch_loss = 0;

    for (unsigned int current_img = 0; current_img < 60000; current_img++) {
      in->output = flattened_imgs[current_img];
      double correct = labels[current_img];
      // printf("correct: %f\n", correct);

      forward_pass(in, l1, relu, l2, squish);

      // calculates loss for data point
      for (unsigned int i = 0; i < 10; i++) {
        if (i + 1 == correct) {
          epoch_loss += calc_mean_squared_loss(squish->output->values[0][i], 1);
        } else {
          epoch_loss += calc_mean_squared_loss(squish->output->values[0][i], 0);
        }
      }

      backward_pass(in, l1, relu, l2, squish, correct);
    }
    epoch_loss /= 60000;
    printf("EPOCH LOSS: %f\n", epoch_loss);
    gradient_descent(l1, l2);
    zero_gradients(l1->weight_grads);
    zero_gradients(l2->weight_grads);
  }

  unsigned int NUM_TEST_IMGS = 10000;
  Matrix **test_imgs = load_MNIST_images("TRAINING_DATA/x_test.txt", NUM_TEST_IMGS);
  Matrix **flattened_test_imgs = (Matrix **)calloc(NUM_TEST_IMGS, sizeof(Matrix *));
  for (unsigned int i = 0; i < NUM_TEST_IMGS; i++) {
    flattened_test_imgs[i] = flatten_matrix_and_append_one(test_imgs[i]);
  }

  double *test_labels = load_MNIST_lables("TRAINING_DATA/y_test.txt", NUM_TEST_IMGS);

  // original images are no longer needed because model uses the flattened form
  free_matrix_array(test_imgs, NUM_TEST_IMGS);

  double accuracy = 0;
  for (unsigned int current_img = 0; current_img < 10000; current_img++) {
    in->output = flattened_test_imgs[current_img];
    double correct = test_labels[current_img];
    // printf("correct: %f\n", correct);
    // print_matrix_verbose(in->output);

    forward_pass(in, l1, relu, l2, squish);
    // printf("correct: %f\n", correct);
    double max_output = 0;
    unsigned int max_output_idx = 0;
    for (unsigned int i = 0; i < 10; i++) {
      if (squish->output->values[0][i] > max_output) {
        max_output = squish->output->values[0][i];
        max_output_idx = i;
      }
    }
    // printf("guess: %d correct%f\n", max_output_idx + 1, correct);
    if (max_output_idx + 1 == correct) {
      accuracy += 1;
    }
  }
  accuracy /= 10000;
  printf("accuracy: %f\n", accuracy);

  free(test_labels);
  free_matrix_array(flattened_test_imgs, NUM_TEST_IMGS);

  free(labels);
  free_matrix_array(flattened_imgs, NUM_IMGS);
  free(in); // output matrix for this layer has already been freed
  return 0;
}