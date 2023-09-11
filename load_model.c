#include <stdio.h>
#include <stdlib.h>
#include "network.h"

void load_weights(char *filename, Matrix *weights) {
  FILE* file = fopen(filename, "r");
  char line[100];

  unsigned int weights_row = 0;
  unsigned int weights_col = 0;
  while (fgets(line, sizeof(line), file)) {
    unsigned int line_idx = 0;

    // replaces newline character with terminating character
    while (line[line_idx++] != '\n') {}
    line[line_idx - 1] = '\0';

    char *endptr;
    double weight = strtod(line, &endptr);

    if (*endptr != '\0') {
      printf("ERROR: unable to read weights.\n");
      printf("End pointer: %s\n", endptr);
      exit(1);
    }

    weights->values[weights_row][weights_col++] = weight;
    if (weights_col >= weights->columns - 1) {
      weights_col = 0;
      weights_row++;
    }
  }
}

int main() {
  Layer *in = init_layer(NULL, NULL, NULL, NULL);
  Layer *l1 = init_layer(NULL, NULL, init_random_weights(785, 129), allocate_empty(785, 128));
  RELU_Layer *relu = init_RELU_layer(NULL, NULL);
  Layer *l2 = init_layer(NULL, NULL, init_random_weights(129, 11), allocate_empty(129, 10));
  Squish_Layer *squish = init_squish_layer(NULL, NULL);

  char *LAYER_1_FILE = "layer1_weights.txt";
  char *LAYER_2_FILE = "layer2_weights.txt";
  load_weights(LAYER_1_FILE, l1->weights);
  load_weights(LAYER_2_FILE, l2->weights);
  printf("Loaded layer #1 weights from \"%s\"\n", LAYER_1_FILE);
  printf("Loaded layer #2 weights from \"%s\"\n", LAYER_2_FILE);
  printf("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");

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
    if (max_output_idx == correct) {
      accuracy += 1;
    }
  }

  printf("\nImages tested: %d\n", NUM_TEST_IMGS);
  printf("Correctly classified: %d\n", (int)accuracy);
  printf("Incorrectly classified: %d\n", NUM_TEST_IMGS - (int)accuracy);
  accuracy /= NUM_TEST_IMGS;
  printf("Accuracy: %.1f%%\n\n", accuracy * 100);

  free(test_labels);
  free_matrix_array(flattened_test_imgs, NUM_TEST_IMGS);
  free(in); // output matrix for this layer has already been freed
  
  return 0;
}