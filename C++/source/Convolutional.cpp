#include "NobleNeuron.hpp"

/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTORS */
/*---------------------------*/
Convolutional::Convolutional(int width, int height, int depth) {
    head = nullptr;
    tail = nullptr;
    
    this->input_width = width;
    this->input_height = height;
    this->input_channels = depth;
}

Convolutional::~Convolutional(void) {
    conv_layer *current,*next;
    
    current = this->head;
    while(current) {
        next = current->next;
        delete current;
        current = next;
    }
}

/*---------------------------*/
/*          METHODS          */
/*---------------------------*/

/**************************************************************************************************/
/*                             ADD CONVOLUTION OR POOLING LAYERS                                  */
/**************************************************************************************************/

/*--- Add convolutional Layer ---*/
void Convolutional::add_conv(int filter_width, int filter_height, int channels, string activation) {
    conv_layer   *n;
    
    this->add_layer(channels);
    n = this->tail;

    n->kind = "convolution";
    n->activation = activation;
    
    n->output_width = n->input_width;
    n->output_height = n->input_height;
    
    n->filter_width = filter_width;
    n->filter_height = filter_height;

    // Filter
    n->F.set_size(channels,1);
    for(int i = 0; i < channels; i ++) {
        n->F(i,0).set_size(filter_height,filter_width,n->depth);
        n->F(i,0) = randu(filter_height,filter_width,n->depth);
    }
    
    // Bias & Output
    n->B.set_size(n->output_height, n->output_width, channels);
    n->Y.set_size(n->output_height, n->output_width, channels);
    n->Z.set_size(n->output_height, n->output_width, channels);
    
    // Derivatives
    n->dB.set_size(n->output_width, n->output_height, channels);
    n->dB.zeros();
    n->dL.set_size(n->output_width, n->output_height, n->depth);
    n->dF.set_size(channels,1);    
    for(int i = 0; i < channels; i ++) {
        n->dF(i,0).set_size(filter_width, filter_height, n->depth);
        n->dF(i,0).zeros();
    }
}

/*--- Add pooling layer ---*/

void Convolutional::add_down(int filter_width, int filter_height, int stride_width, int stride_height, string pooling) {
    conv_layer *n;
    
    this->add_layer(0);
    n = this->tail;
    
    n->kind = "downsampling";
    n->stride_width = stride_width;
    n->stride_height = stride_height;
    
    transform(pooling.begin(),pooling.end(),pooling.begin(),:: tolower);
    
    n->pooling = pooling;
    n->filter_width = filter_width;
    n->filter_height = filter_height;
        
    n->output_width = n->input_width/stride_width;
    n->output_height = n->input_height/stride_height;
    
    n->Z.set_size(n->output_height, n->output_width, n->depth);
    n->map.set_size(n->input_height, n->input_width, n->depth);
    n->dL.set_size(n->input_height, n->input_width, n->depth);
}

/*--- Add a layer ---*/
void Convolutional::add_layer(int channels)
{
    conv_layer *n;
    
    n = new conv_layer();
    n->next = nullptr;
        
    if (!head) {
        n->prev = nullptr;
        
        n->depth = this->input_channels;
        n->input_width = this->input_width;
        n->input_height = this->input_height;
        
        if (channels > 0) {
            n->channels = channels;
        }
        else {
            n->channels = this->input_channels;
        }
        
        this->head = n;
        this->tail = n;
    }
    else {
        n->prev = this->tail;
        
        n->depth = this->tail->channels;
        n->input_width = this->tail->output_width;
        n->input_height = this->tail->output_height;
        
        if (channels > 0) {
            n->channels = channels;
        }
        else {
            n->channels = this->tail->channels;
        }
        
        this->tail->next = n;
        this->tail = n;
    }
}

/**************************************************************************************************/
/*                                    PROPAGATION: BACK & FORTH                                   */
/**************************************************************************************************/

/*--- FeedForward ---*/
void Convolutional::feedforward(cube X) {
    conv_layer *n;
    
    // Save a copy of the input matrix.
    input_X = X;
    
    // It always begins at the head, but if it is not set, then return.
    n = head;
    if (!n) return;
    
    // Feed
    if (n->kind == "convolution") {
        n->feed_convolution(X);
    }
    if (n->kind == "downsampling") {
        n->feed_down(X);
    }
    
    // Next
    n = n->next;  
    
    // Remaining elements: feed and move to next
    while(n) {
        if (n->kind == "convolution") {
            n->feed_convolution(n->prev->Z);
        }
        if (n->kind == "downsampling") {
            n->feed_down(n->prev->Z);
        }
        n = n->next;
    }
}

/*--- BackPropagate ---*/
void Convolutional::backpropagate(cube dL) {
    conv_layer *n;
    
    // It always begins at the tail, but if it is not set, then return.
    n = tail;
    if (!n) return;
    
    // If tail is the head, then backpropagate to the input matrix,
    // else backpropagate to the previous layer.
    if (n == head) {
        if (n->kind == "convolution") {
            n->back_conv(input_X,dL);
        }
        if (n->kind == "downsampling") {
            n->back_down(dL);
        }
    }
    else {
        if (n->kind == "convolution") {
            n->back_conv(n->prev->Z,dL);
        }
        if (n->kind == "downsampling") {
            n->back_down(dL);
        }
    }
    
    n = n->prev;    // previous
    while(n) {
        // If it has reached the head, then propagate to the input matrix
        if (n == head) {
            if (n->kind == "convolution") {
                n->back_conv(input_X,n->next->dL);
            }
            if (n->kind == "downsampling") {
                n->back_down(n->next->dL);
            }
        }
        // otherwise, propagate to the previous layer.
        else {
            if (n->kind == "convolution") {
                n->back_conv(n->prev->Z, n->next->dL);
            }
            if (n->kind == "downsampling") {
                n->back_down(n->next->dL);
            }
        }
        n = n->prev;    // previous
    }
}

/**************************************************************************************************/
/*                                          GRADIENTS                                             */
/**************************************************************************************************/

/*--- Apply Gradients ---*/
void Convolutional::applygrads(float eta) {
    conv_layer *n;
    
    n = head;
    while(n) {
        if (n->kind == "convolution")
            n->gradient_descent(eta);
        n = n->next;
    }
}

/**************************************************************************************************/
/*                                        MISCELLANEOUS                                           */
/**************************************************************************************************/

/*--- Output of Neural Networks ---*/
cube Convolutional::output(void) {
    return tail->Z;
}

mat Convolutional::flatten(void) {
    mat M = tail->Z.slice(0);
    
    return reshape(M,M.n_cols*M.n_rows,1);
}
