# NobleNeuron


<img src="https://github.com/StxGuy/NobleNeuron/blob/main/NobleNeuron.png" width="256" height="256">

## About

A library in C++/Armadillo and Fortran for simulating neural networks. Currently the C++ version supports multilayer feedforward and convolutional networks.
    
[![Generic badge](https://img.shields.io/badge/GitHub-StxGuy/NobleNeuron-<COLOR>.svg)](https://github.com/StxGuy/NobleNeuron)

### File Structure
* FeedForward.cpp (Create a feedforward multilayer network: add layer, feedforward, backpropagate)
  + ff_layer.cpp (Execute the math of each layer of the feedforward network)
    - activ.cpp (Activation ande deactivation functions)
* Convolutional.cpp (Create a convolutional network: add layer, feedforward, backpropagate)
  + conv_layer.cpp (Execute de math of a convolutional layer)
    - activ.cpp
  + pooling_layer.cpp (Execute the math of a pooling layer)
* loss.coo (Loss functions)

### Compile Noble Neuron
make \
make install \
make clean

### Compile an Example
g++ main.cpp -o executable -lNobleNeural -larmadillo

### Use Noble Neuron
#### FeedForward F(n), n is the size of the input vector
+  F.add_layer(n,s), 'n' is the size of layer output and 's' is the activation;
+  F.feedforward(M), 'M' is an input column vector;
+  F.backpropagate(M), 'M' is the gradient of the loss function;
+  F.applygrads(eta), Apply gradients, 'eta' is the learning rate;
+  F.output(), Gives the output of the network;
+  F.unflatten(rows,cols), Produces the gradient of the network as a rows x cols of a slice of a cube to be used in a convolutional network;

#### Convolutional C(height,width,depth), create a convolutional network with this input
+ C.add_conv(fh,fw,channels,activation), Add a convolution layer with 'channels' fh x fw filters;
+ C.add_down(fh,fw,sh,sw,kind), Add a down sampling layer. sh and sw are the stride, kind is the pooling = max or average;
+ C.feedforward(D), 'D' is an input cube;
+ C.backpropagate(D), 'D' is the gradient of the loss function;
+ C.applygrads(eta)
+ C.output()
+ C.flatten(), Produces a column matrix to be used in a feedforward network;

## Credits


    @misc{daCunha2021,
        author       = {C. R. da Cunha},
        title        = {{NobleNeuron, a library for simulating neural networks.}},
        month        = apr,
        year         = 2021,
        version      = {1.0},
        publisher    = {GitHub},
        url          = {https://github.com/StxGuy/NobleNeuron}
        }
        
## License

Copyright (c) 2021-2022 - Carlo R. da Cunha, "NobleNeuron" \
<carlo.requiao@gmail.com>
