
#include "NobleNeuron.hpp"


/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTORS */
/*---------------------------*/
FeedForward::FeedForward(int isize) {
    head = nullptr;
    tail = nullptr;
    
    input_size = isize;
}

FeedForward::~FeedForward(void) {
    ff_layer *current,*next;
    
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
void FeedForward::add_layer(int output_size, string activation) {
    ff_layer   *n;
    
    n = new ff_layer();
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
void FeedForward::feedforward(mat X) {
    ff_layer *n;
    
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
void FeedForward::backpropagate(mat dL) {
    ff_layer *n;
    
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
void FeedForward::applygrads(float eta) {
    ff_layer *n;
    
    n = head;
    while(n) {
        n->gradient_descent(eta);
        n = n->next;
    }
}

/*--- Output of Neural Networks ---*/
mat FeedForward::output(void) {
    return this->tail->Z;
}

/*--- Propagating Gradient ---*/
// dL = matrix(n x 1)
// reshape to rows x cols = n
// include as a slice of a cube
cube FeedForward::unflatten(int rows, int cols) {
    mat M = this->head->dL;
    cube G(rows,cols,1);
    
    
    G.slice(0) = reshape(M,rows,cols);
    
    return G;
}
