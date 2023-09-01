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

void print_matrix_row(Matrix *matrix, int row);

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
 * Calculates the dot product of two vectors and returns the answer as a
 * scalar.
 * 
 * @param vec1 A size 1 x n or n x 1 matrix to represent a vector
 * @param vec2 A size 1 x n or n x 1 matrix to represent a vector
 * @return The dot product of the input vectors
 */
double dot_product(Matrix *vec1, Matrix *vec2);

void print_dot(Matrix *A, Matrix *B);

Layer *init_layer(Matrix *output, 
                  Matrix *output_grads, 
                  Matrix *weights, 
                  Matrix *weight_grads);

RELU_Layer *init_RELU_layer(Matrix *output, Matrix *output_grads);

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

Matrix *weights_matmul_gradients_subtotal(Layer *layer);

Matrix *calc_layer_input_gradients(Layer *above_layer);

void calc_RELU_layer(RELU_Layer *relu, Matrix *input);

void calc_layer_gradients_from_RELU(Layer *input_layer, Matrix *RELU_grads);

void print_layer(Layer *layer, char *layer_name);

void print_RELU_layer(RELU_Layer *layer, char *layer_name);

void calc_weight_gradients(Layer *layer, Matrix *layer_inputs);
