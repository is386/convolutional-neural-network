# Convolutional Neural Network

This is an convolutional neural network built from scratch. It uses a softmax activation function on all the layers. The gradient is computed by using the cross entropy function. It is used on two black images, one has a horizontal line while the other has a vertical line. The output is a picture of the kernel that shows the two lines from the original images.

## Usage

`python3 cnn.py`

The kernel images will be saved in the same directory.

## Dependencies

- `python 3.8+`

### Python Dependencies

- `numpy`
- `matplotlib`

## Hyper Parameters

- Learning Rate: `0.01`

- Termination Criteria: `100 iterations`

- L2 Regularization Term: `0.1`

## Results

The following network was used for this test:

- 40x40 convolution layer with a single kernel

- 1x1 max pooling layer with a stride of 1

- Flattening layer

- Shallow input layer

The following synthetic data was used for this test:

![](https://github.com/is386/CNN/blob/master/line1.png?raw=true)
![](https://github.com/is386/CNN/blob/master/line2.png?raw=true)

### Initial and Final Kernels

![](https://github.com/is386/CNN/blob/master/initial.png?raw=true)
![](https://github.com/is386/CNN/blob/master/final.png?raw=true)

### Cross Entropy:

![](https://github.com/is386/CNN/blob/master/cross_entropy.png?raw=true)
