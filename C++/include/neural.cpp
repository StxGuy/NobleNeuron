#ifndef NN_Body
#define NN_Body

#include "layer.hpp"


class NeuralNetwork {
    private:
        /*--- Linked List ---*/
        layer *head;
        layer *tail;
        
        /*--- General Parameters ---*/
        int     input_size;
        mat     input_X;
        
    public:
        void    add_layer(int,string);
        void    feedforward(mat);
        void    backpropagate(mat);
        void    applygrads(float);
        mat     output(void);
        
        NeuralNetwork(int);
        ~NeuralNetwork(); 
};

#endif
