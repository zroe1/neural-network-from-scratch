# Neural network with only stdio.h and stdlib.h in C  
<img src="https://img.shields.io/badge/Data-MNIST-black">  <img src="https://img.shields.io/badge/Accuracy-70.7-d6fc2d">  <img src="https://img.shields.io/badge/tests-passing-brightgreen">

This repository includes code for a fully functional neural network using only stdio.h and stdlib.h in C. Preforming matrix multiplication, calculating derivatives, and updating gradients is all done without any outside libraries.

The code is then tested on the famous MNIST handwritten digit dataset. The accuracy of the model is 70.7% on images not present in the training data, but I suspect this number would be higher if it were not for the contraints of only using stdio.h and stdlib.h in C, as I discuss more below.

## Model architecture

The model architecture is almost identical to the [MNIST model](https://www.tensorflow.org/datasets/keras_example) in the TensorFlow documentation. Both models take in a flattened image, have one fully connected layer with 128 outputs (activation RELU), followed by another layer with 10 outputs. 

The main difference is between the loss. The TensorFlow model specifies "SparseCategoricalCrossentropy" as the loss which applies the softmax function to the model outputs (shown below). 

<b>Softmax layer:</b>  
<img width="535" alt="Screenshot 2023-09-07 at 5 19 35 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/ceddfeaf-4476-4df7-8af1-d97967f691c2">  

This sadly cannot easily be calculated without importing the math.h library in C. I wanted to stick to only using stdio.h and stdlib.h to challenge myself as an ML engineer so I chose a substitution that works without any extra imports. I don't know if there is a technical term for this calculation but I call it the "squish" layer (shown below).

<b>"Squish" layer:</b>  
<img width="517" alt="Screenshot 2023-09-07 at 5 19 26 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/2657755b-be4f-4c05-919b-dda94c4612ab">

## Optimizer

The model uses Stochastic Gradient Descent (SGD) to update gradients. The learning rate I have found works best is 10<sup>-3</sup>.

## Running the code

```
clang train.c network_funcs.c linalg.c
```
Run the command above in terminal. The output will display the loss for each epoch and the overall accuracy of the model.

## Next steps

I am going to take a break on this project for a while after I complete the steps below.

<ol>
  <li>Use valgrind to confirm that all allocated memory is freed</li>
  <li>Write docstrings for each function in the header file</li>
  <li>Remove hardcoded values in train.c and replace them with constants</li>
</ol>

Thanks for reading. Made with ❤️ and C.
<p align="center"><img width="100%" alt="Screenshot 2023-09-05 at 2 00 32 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/e7ec83eb-38ca-4cde-9f18-950832b5bcee"></p>  
