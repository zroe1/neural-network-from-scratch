typedef struct {
  unsigned int rows;
  unsigned int columns;
  double **values;
} Matrix;

typedef struct {
  Matrix *output;
  Matrix *output_grads;

  Matrix *weights;
  Matrix *weight_grads;
} Layer;

typedef struct {
  Matrix *output;
  Matrix *output_grads;
} RELU_Layer;

/* 
 * This may be swapped out for softmax or sigmoid in a future version.
 * Currently it adds by a constant to make all values positive and divides by 
 * the sum of the ouputs of the previous layer to put all values between 0 & 1.
 */
typedef struct {
  double output_sum;
  Matrix *output;
  Matrix *output_grads;
} Squish_Layer;

/**
 * Prints a matrix to standard output.
 * 
 * @param matrix The matrix to be printed.
 */
void print_matrix(Matrix *matrix);

/**
 * Prints a matrix and it's dimensions to standard output.
 * 
 * @param matrix The matrix to be printed.
 */
void print_matrix_verbose(Matrix *matrix);

/**
 * Prints one row of a matrix. If row passed in is -1 it prints the top padding
 * row. If row passed in is equal to the number of rows in the matrix, it prints
 * the bottom padding row. This function is designed to help do complex printing
 * operations.
 * 
 * @param matrix The matrix to be printed.
 * @param row The row of the matrix to be printed.
 */
void print_matrix_row(Matrix *matrix, int row);

/**
 * Allocates a matrix and it's values (initialized to 0) to the heap.
 * 
 * @param rows The number of rows in the matrix.
 * @param columns The number of columns in the matrix.
 * @return The allocated Matrix struct
 */
Matrix *allocate_empty(unsigned int rows, unsigned int columns);

/**
 * Allocates a matrix struct to the heap using the values of the 2D array 
 * passed to the function.
 * 
 * @param rows The number of rows in the matrix.
 * @param columns The number of columns in the matrix.
 * @param arr[rows][columns] The 2D array where the matrix gets it's values
 * @return The allocated Matrix struct
 */
Matrix *allocate_from_2D_arr(unsigned int rows, unsigned int columns,
                             double arr[rows][columns]);

/**
 * Frees an allocated matrix and the values associated with it.
 * 
 * @param matrix The matrix to be freed
 */
void free_matrix(Matrix *matrix);

/**
 * Preforms matrix multiplication on 2 matrices and returns the result as a
 * freshly allocated matrix.
 * 
 * @param A The first matrix to be multiplied.
 * @param B The second matrix to be multiplied.
 * @return The result of the matrix multiplication.
 */
Matrix *matmul(Matrix *A, Matrix *B);

/**
 * Prints the matrix multiplication of 2 matrices.
 * 
 * @param A The first matrix to be multiplied.
 * @param B The second matrix to be multiplied.
 */
void print_matmul(Matrix *A, Matrix *B);

/**
 * Initializes a layer with the output, output gradients, wieghts, and weight
 * gradients passed into the function.
 * 
 * @param output Matrix representing layer output
 * @param output_grads Matrix representing layer output gradients
 * @param weights Matrix representing layer weights
 * @param weight_grads Matrix representing layer weight gradients
 * @return A freshly allocated layer.
 */
Layer *init_layer(Matrix *output, 
                  Matrix *output_grads, 
                  Matrix *weights, 
                  Matrix *weight_grads);

/**
 * Frees all associated memory with a RELU layer.
 * 
 * @param l an allocated RELU layer
 */
void free_RELU_layer(RELU_Layer *l);

/**
 * Frees all associated memory with a layer.
 * 
 * @param l an allocated layer
 */
void free_layer(Layer *l);

RELU_Layer *init_RELU_layer(Matrix *output, Matrix *output_grads);

/**
 * Changes a Matrix struct representing a set of gradients so each value in the
 * matrix is 0.
 * 
 * @param grads A matrix representing a set of gradients.
 */
void zero_gradients(Matrix *grads);

/**
 * Initializes a Matrix with random weights between -1 and 1. The final column 
 * is all zeros except with a one at the bottom. 
 * 
 * @param rows The number of rows in the matrix to be returned
 * @param cols The number of columns in the matrix to be returned
 * @return The random weights
 */
Matrix *init_random_weights(unsigned int rows, unsigned int cols);

/**
 * Calculates the output of a layer in a neural network assuming it's weights
 * have been initialized.
 * 
 * @param layer A layer with initalized weights
 * @param input A size 1 x n sized matrix
 * @return The output of the layer
 */
void calc_layer_output(Layer *layer, Matrix *input);

/**
 * Calculates the input gradients to a layer. If the gradients have been
 * initialized they will be overwritten by new gradients. Otherwise, new 
 * gradients will be allocated.
 * 
 * @param above_layer layer above the input gradients calculated
 * @param rv the input gradients
 * @return the input gradients
 */
Matrix *calc_layer_input_gradients(Layer *above_layer, Matrix *rv);

/**
 * Calculates the output of a RELU layer. 
 * 
 * @param relu an allocated RELU layer
 * @param input the input to a RELU layer
 */
void calc_RELU_layer(RELU_Layer *relu, Matrix *input);

/**
 * Calculates the output gradients for a given layer, given that there is a
 * RELU layer directly above it. 
 * 
 * @param input_layer an allocated layer
 * @param RELU_grads a matrix representing the gradients of the RELU output
 */
void calc_layer_gradients_from_RELU(Layer *input_layer, Matrix *RELU_grads);

/**
 * Prints a layer given it's output, output gradients, weights, and weight 
 * gradients. The output also includes the "layer_name" passed into the
 * function
 * 
 * @param layer an allocated layer
 * @param layer_name the name of the layer to be printed
 */
void print_layer(Layer *layer, char *layer_name);

/**
 * Prints a RELU layer given it's output and output gradients. The output also 
 * includes the "layer_name" passed into the function.
 * 
 * @param layer an allocated RELU layer
 * @param layer_name the name of the layer to be printed
 */
void print_RELU_layer(RELU_Layer *layer, char *layer_name);

/**
 * Prints a "squish" layer given it's output and output gradients. The output 
 * also includes the "layer_name" passed into the function.
 * 
 * @param layer an allocated "squish" layer
 * @param layer_name the name of the layer to be printed
 */
void print_squish_layer(Squish_Layer *layer, char *layer_name);

/**
 * Calculates the weight gradients of a layer given the layer and the layer
 * inputs.
 * 
 * @param layer an allocated layer.
 * @param layer_inputs A matrix representing the inputs to a layer.
 */
void calc_weight_gradients(Layer *layer, Matrix *layer_inputs);

/**
 * Updates the layer weights based on the weight gradients and learning rate.
 * 
 * @param layer an allocated layer.
 * @param learning_rate a small number the gradient is multiplied by to 
 *                      determine the size of change for a weight.
 */
void gradient_descent_on_layer(Layer *layer, double learning_rate);

/**
 * Calculates the output of a neural network with the architecture described in
 * the README.md file.
 * 
 * @param input_layer The input layer to the network
 * @param layer1 The first layer in the network (fully connected)
 * @param layer1_RELU RELU layer to be applied to first layer
 * @param layer2 The second layer in the network (fully connected)
 * @param squish The "squish" layer described in README.md
 */
void forward_pass(Layer *input_layer,
                  Layer *layer1,
                  RELU_Layer *layer1_RELU,
                  Layer *layer2,
                  Squish_Layer *squish);

/**
 * Calculates the gradients of the weights and outputs in the netwrok with the 
 * architecture described in the README.md file.
 * 
 * @param input_layer The input layer to the network
 * @param layer1 The first layer in the network (fully connected)
 * @param layer1_RELU RELU layer to be applied to first layer
 * @param layer2 The second layer in the network (fully connected)
 * @param squish The "squish" layer described in README.md
 * @param correct The correct label for the previous forward pass
 */
void backward_pass(Layer *input_layer,
                   Layer *layer1,
                   RELU_Layer *layer1_RELU,
                   Layer *layer2,
                   Squish_Layer *squish,
                   double correct);

/**
 * Calculates the mean squared loss of a single output.
 * 
 * @param output A single output.
 * @param correct The correct value for a single ouput.
 * @return The mean squared loss.
 */
double calc_mean_squared_loss(double output, double correct);

/**
 * Calculates the gradient of a single output assuming the use of mean squared 
 * loss.
 * 
 * @param output A single output.
 * @param correct The correct value for a single ouput.
 * @return The gradient of the output.
 */
double calc_grad_of_input_to_loss(double ouput, double correct);

/**
 * Allocates a squish layer with the output and output gradients that are passed
 * in through the function.
 * 
 * @param output Matrix representing layer output
 * @param output_grads Matrix representing layer output gradients
 * @return An allocated squish layer
 */
Squish_Layer *init_squish_layer(Matrix *output, Matrix *output_grads);

/**
 * Frees all associated memory with a squish layer.
 * 
 * @param l an allocated squish layer
 */
void free_squish_layer(Squish_Layer *l);

/**
 * Calculates the output of a "squish" layer. More information about this
 * calculation can be found in README.md.
 * 
 * @param layer an allocated squish layer
 * @param inputs A matrix representing the input to the squish layer
 */
void calc_squish_layer(Squish_Layer *layer, Matrix *inputs);

/**
 * Calculates the output gradients for a given layer, given that there is a
 * "squish" layer directly above it. The new gradients are stored in 
 * "input_layer"
 * 
 * @param input_layer an allocated layer
 * @param squish an allocated squish layer
 */
void calc_layer_gradients_from_squish(Layer *input_layer, Squish_Layer *squish);

/**
 * Loads the MNIST lables from a formated text file.
 * 
 * @param filename Name of the file to be read
 * @param num_lines The number of expected lines
 * @return A list of doubles representing image labels
 */
double *load_MNIST_lables(char *filename, unsigned int num_lines);

/**
 * Loads the MNIST images from a formated text file.
 * 
 * @param filename Name of the file to be read
 * @param num_imgs The number of expected images
 * @return A list of Matrix structs representing MNIST images
 */
Matrix **load_MNIST_images(char *filename, unsigned int num_imgs);

/**
 * Frees all the data assocated with an array of Matrix structs.
 * 
 * @param matrix_arr Matrix array to be freed
 * @param len Number of matrices in the array
 */
void free_matrix_array(Matrix **matrix_arr, unsigned int len);

/**
 * Flattens a matrix so it's shape is 1 * n and appends a one to the end.
 * 
 * @param matrix Matrix to be flattened
 * @return The flattened matrix
 */
Matrix *flatten_matrix_and_append_one(Matrix *matrix);