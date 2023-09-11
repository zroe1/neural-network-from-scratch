# Neural network with only stdio.h and stdlib.h in C  
<img src="https://img.shields.io/badge/Data-MNIST-black">  <img src="https://img.shields.io/badge/Accuracy-70.7-d6fc2d">  <img src="https://img.shields.io/badge/tests-passing-brightgreen">

This repository includes code for a fully functional neural network using only stdio.h and stdlib.h in C. Performing matrix multiplication, calculating derivatives, and updating gradients is all done without any outside libraries.

The code is then tested on the famous MNIST handwritten digit dataset. The accuracy of the model is 70.7% on images not present in the training data, but I suspect this number would be higher if it weren't for the constraints of only using stdio.h and stdlib.h in C, as I discuss more below.

## Model architecture

The model architecture is almost identical to the [MNIST model](https://www.tensorflow.org/datasets/keras_example) in the TensorFlow documentation. Both models take in a flattened image, have one fully connected layer with 128 outputs (activation RELU), followed by another layer with 10 outputs. 

The main difference between the models is the loss. The TensorFlow model specifies "SparseCategoricalCrossentropy" as the loss which applies the softmax function to the model outputs (shown below). 

<b>Softmax layer:</b>  
<img width="535" alt="Screenshot 2023-09-07 at 5 19 35 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/ceddfeaf-4476-4df7-8af1-d97967f691c2">  

This sadly cannot easily be calculated without importing the math.h library in C. I wanted to stick to only using stdio.h and stdlib.h to challenge myself as an ML engineer so I chose a substitution that works without any extra imports. I don't know if there is a technical term for this calculation but I call it the "squish" layer (shown below).

<b>"Squish" layer:</b>  
<img width="517" alt="Screenshot 2023-09-07 at 5 19 26 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/2657755b-be4f-4c05-919b-dda94c4612ab">

## Notes on performance

I have a few guesses why the performance of the network is only ~70%, while other networks of similar architecture typically score well above 90%.

<ol>
  <li>The differences in model architecture (described above), made necessary by the constraints of the project, could be causing bad gradient updates.</li>
  <li>I could be choosing a bad loss function (I have not experimented with loss functions other than mean squared error).</li>
  <li>There could be an error somewhere in the code that I am missing.</li>
</ol>

If I had to guess, it probably is some combination of 1 & 2. The problem with fixing either of these problems is it would be very difficult to do well only using stdlib.h and stdio.h (this is less true for #2 but it still applies).

## Optimizer and loss

The model uses stochastic gradient descent (SGD) and a mean squared error loss to update gradients. The network is also designed to work with regular gradient descent, where the gradient is calculated by factoring in each image in the training data. The learning rate I have found works best is 0.0015.

## Running the code

To load the model I trained run:
```
make load
./load
```
Because the Makefile is configured to my computer this command may fail. To troubleshoot this, you can compile the necessary files manually using your favorite C compiler. Mine is Clang:
```
clang -o load load_model.c linalg.c network_funcs.c
./load
```

Running either of these commands will give detailed information about the model's performance on a testing dataset.

## Next steps

As of writing this (9/10/23) the code works as intended and runs Valgrind-clean:  
  
<img width="822" alt="Screenshot 2023-09-10 at 2 23 59 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/f67ceb33-5a7a-4c90-b532-3016c3cf0d64">  

I may return to this project at some point to improve model performance, but for the time being, I am taking a break to work on other things.  

Thanks for reading. Made with ❤️ and C.
<p align="center"><img width="100%" alt="Screenshot 2023-09-05 at 2 00 32 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/e7ec83eb-38ca-4cde-9f18-950832b5bcee"></p>  
