#include <stdio.h>
#include <stdlib.h>
#include "network.h"

double LEARNING_RATE = 0.0015;
double NUM_EPOCHS = 20;

void save_weights(Matrix *weights, char *filename) {
  FILE *weights_file = fopen(filename, "w");

  for (unsigned int i = 0; i < weights->rows; i++) {
    for (unsigned int j = 0; j < weights->columns - 1; j++) {
      fprintf(weights_file, "%f\n", weights->values[i][j]);
    }
  }
}

int main() {
  printf("LEARNING RATE: %f\n", LEARNING_RATE);
  unsigned int NUM_IMGS = 60000;

  Matrix **imgs = load_MNIST_images("TRAINING_DATA/x_train.txt", NUM_IMGS);
  Matrix **flattened_imgs = (Matrix **)calloc(NUM_IMGS, sizeof(Matrix *));
  for (unsigned int i = 0; i < NUM_IMGS; i++) {
    flattened_imgs[i] = flatten_matrix_and_append_one(imgs[i]);
    normalize_img_matrix(flattened_imgs[i]);
  }

  // original images are no longer needed because model uses the flattened form
  free_matrix_array(imgs, NUM_IMGS);

  double *labels = load_MNIST_lables("TRAINING_DATA/y_train.txt", NUM_IMGS);
  Layer *in = init_layer(NULL, NULL, NULL, NULL);
  Layer *l1 = init_layer(NULL, NULL, init_random_weights(785, 129), allocate_empty(785, 128));
  RELU_Layer *relu = init_RELU_layer(NULL, NULL);
  Layer *l2 = init_layer(NULL, NULL, init_random_weights(129, 11), allocate_empty(129, 10));
  Squish_Layer *squish = init_squish_layer(NULL, NULL);

  for (unsigned int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    double epoch_loss = 0;

    for (unsigned int current_img = 0; current_img < NUM_IMGS; current_img++) {
      in->output = flattened_imgs[current_img];
      double correct = labels[current_img];

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
      gradient_descent(l1, l2, LEARNING_RATE);
      zero_gradients(l1->weight_grads);
      zero_gradients(l2->weight_grads);
    }
    epoch_loss /= NUM_IMGS;
    // epoch_loss = 60000;
    printf("#%d. EPOCH LOSS: %f\n", epoch + 1, epoch_loss);
  }

  unsigned int NUM_TEST_IMGS = 10000;
  Matrix **test_imgs = load_MNIST_images("TEST_DATA/x_test.txt", NUM_TEST_IMGS);
  Matrix **flattened_test_imgs = (Matrix **)calloc(NUM_TEST_IMGS, sizeof(Matrix *));
  for (unsigned int i = 0; i < NUM_TEST_IMGS; i++) {
    flattened_test_imgs[i] = flatten_matrix_and_append_one(test_imgs[i]);
    normalize_img_matrix(flattened_test_imgs[i]);
  }

  double *test_labels = load_MNIST_lables("TEST_DATA/y_test.txt", NUM_TEST_IMGS);

  // original images are no longer needed because model uses the flattened form
  free_matrix_array(test_imgs, NUM_TEST_IMGS);

  double accuracy = 0;
  for (unsigned int current_img = 0; current_img < NUM_TEST_IMGS; current_img++) {
    in->output = flattened_test_imgs[current_img];
    double correct = test_labels[current_img];

    forward_pass(in, l1, relu, l2, squish);
    double max_output = 0;
    unsigned int max_output_idx = 0;
    for (unsigned int i = 0; i < 10; i++) {
      if (squish->output->values[0][i] > max_output) {
        max_output = squish->output->values[0][i];
        max_output_idx = i;
      }
    }
    if (max_output_idx + 1 == correct) {
      accuracy += 1;
    }
  }
  accuracy /= NUM_TEST_IMGS;
  printf("accuracy: %f\n", accuracy);

  save_weights(l1->weights, "layer1_weights.txt");
  save_weights(l2->weights, "layer2_weights.txt");

  free(test_labels);
  free_matrix_array(flattened_test_imgs, NUM_TEST_IMGS);

  free(labels);
  free_matrix_array(flattened_imgs, NUM_IMGS);
  free(in); // output matrix for this layer has already been freed
  return 0;
}