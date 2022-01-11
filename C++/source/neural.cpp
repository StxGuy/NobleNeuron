
#include "neural.hpp"


/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTORS */
/*---------------------------*/
NeuralNetwork::NeuralNetwork(int isize) {
    head = nullptr;
    tail = nullptr;
    
    input_size = isize;
}

NeuralNetwork::~NeuralNetwork(void) {
    layer *current,*next;
    
    current = head;
    while(current) {
        next = current->next;
        delete current;
        current = next;
    }
}

/*---------------------------*/
/*          METHODS          */
/*---------------------------*/

/*--- Add Layer ---*/
void NeuralNetwork::add_layer(int output_size, string activation) {
    layer   *n;
    
    n = new layer();
    n->next = nullptr;
    n->activation = activation;
        
    if (!head) {
        n->prev = nullptr;
        n->input_size = input_size;
        n->output_size = output_size;
        head = n;
        tail = n;
    }
    else {
        n->prev = tail;
        n->input_size = n->prev->output_size;
        n->output_size = output_size;
        tail->next = n;
        tail = n;
    }
    
    n->W = randu(n->output_size, n->input_size);
    n->B = randu(n->output_size, 1);
    n->dW = zeros(n->input_size, n->output_size);
    n->dB = zeros(1, n->output_size);
}

/*--- FeedForward ---*/
void NeuralNetwork::feedforward(mat X) {
    layer *n;
    
    // Save a copy of the input matrix.
    input_X = X;
    
    // It always begins at the head, but if it is not set, then return.
    n = head;
    if (!n) return;
         
    n->feed(X);     // Feed
    n = n->next;    // next
    
    // Remaining elements: feed and move to next
    while(n) {
        n->feed(n->prev->Z);
        n = n->next;
    }
}

/*--- BackPropagate ---*/
void NeuralNetwork::backpropagate(mat dL) {
    layer *n;
    
    // It always begins at the tail, but if it is not set, then return.
    n = tail;
    if (!n) return;
    
    // If tail is the head, then backpropagate to the input matrix,
    // else backpropagate to the previous layer.
    if (n == head) 
        n->back(input_X,dL);
    else
        n->back(n->prev->Z,dL);
    
    n = n->prev;    // previous
    while(n) {
        // If it has reached the head, then propagate to the input matrix
        if (n == head)
            n->back(input_X,n->next->dL);
        // otherwise, propagate to the previous layer.
        else
            n->back(n->prev->Z,n->next->dL);
        n = n->prev;    // previous
    }
}

/*--- Apply Gradients ---*/
void NeuralNetwork::applygrads(float eta) {
    layer *n;
    
    n = head;
    while(n) {
        n->gradient_descent(eta);
        n = n->next;
    }
}

/*--- Output of Neural Networks ---*/
mat NeuralNetwork::output(void) {
    return tail->Z;
}

