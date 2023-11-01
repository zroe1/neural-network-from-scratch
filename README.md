# Neural network with only stdio.h and stdlib.h in C  
<img src="https://img.shields.io/badge/Data-MNIST-black">  <img src="https://img.shields.io/badge/Accuracy-79.4-d6fc2d">  <img src="https://img.shields.io/badge/tests-passing-brightgreen">

This repository includes code for a fully functional neural network using only stdio.h and stdlib.h in C. Performing matrix multiplication, calculating derivatives, and updating gradients is all done without any outside libraries.

The code is then tested on the famous MNIST handwritten digit dataset. The accuracy of the model is 79.4% on images not present in the training data, but I suspect I could improve accuracy with a few changes as I discuss below.

## Model architecture

The model architecture is almost identical to the [MNIST model](https://www.tensorflow.org/datasets/keras_example) in the TensorFlow documentation. Both models take in a flattened image, have one fully connected layer with 128 outputs (activation ReLU), followed by another layer with 10 outputs. 

The main difference between the models is the loss. The TensorFlow model specifies "SparseCategoricalCrossentropy" as the loss which applies the softmax function to the model outputs (shown below). 

<b>Softmax layer:</b>  
<img width="535" alt="Screenshot 2023-09-07 at 5 19 35 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/ceddfeaf-4476-4df7-8af1-d97967f691c2">  

This sadly cannot easily be calculated without importing  math.h to calculate e<sup>z</sup> where "e" is Euler's number and "z" is a C double. I wanted to stick to only using stdio.h and stdlib.h to challenge myself as an ML engineer so I chose a substitution that works without any extra imports. I don't know if there is a technical term for this calculation but I call it the "squish" layer (shown below).

<b>"Squish" layer:</b>  
<img width="517" alt="Screenshot 2023-09-07 at 5 19 26 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/2657755b-be4f-4c05-919b-dda94c4612ab">

## Notes on performance

I have a few ideas why the performance of the network is only ~79%, while other networks of similar architecture typically score well above 90%. My best guess, however, is it probably has something to do with the fact that I used a nontraditional way of calculating the probability distribution of the data using a "squish" layer, discussed in the above section. Softmax in the original model probably played a role as a non-linearity in addition to merely converting outputs to probabilities. As a result, I think adding an extra layer with another ReLU would probably increase performance. The model is currently simply too linear.

## Optimizer and loss

The model uses stochastic gradient descent (SGD) and a mean squared error loss to update gradients. The network is also designed to work with regular gradient descent, where the gradient is calculated by factoring in each image in the training data. The learning rate I have found works best is 10<sup>-3</sup>.

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

For future changes I plan to make see the "Notes on performance" section. I will likely return to this project in the next few months but for the time being, I am taking a break to work on other things.  

Thanks for reading. Made with ❤️ and C.
<p align="center"><img width="100%" alt="Screenshot 2023-09-05 at 2 00 32 PM" src="https://github.com/zroe1/neural-network-from-scratch/assets/114773939/e7ec83eb-38ca-4cde-9f18-950832b5bcee"></p>  
