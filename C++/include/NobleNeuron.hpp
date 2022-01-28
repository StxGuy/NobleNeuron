#ifndef NobleNeuron
#define NobleNeuron

#include "ff_layer.hpp"
#include "conv_layer.hpp"

/*************************************************************************/
/*                       FEEDFORWARD NEURAL NETWORK                      */
/*************************************************************************/

class FeedForward {
    private:
        /*--- Linked List ---*/
        ff_layer *head;
        ff_layer *tail;
        
        /*--- General Parameters ---*/
        int     input_size;
        mat     input_X;
        
    public:
        void    add_layer(int,string);
        void    feedforward(mat);
        void    backpropagate(mat);
        void    applygrads(float);
        mat     output(void);
        cube    unflatten(int,int);
        
        FeedForward(int);
        ~FeedForward(); 
};

/*************************************************************************/
/*                    CONVOLUTIONAL NEURAL NETWORK                       */
/*************************************************************************/

class Convolutional {
    private:
        /*--- Linked List ---*/
        conv_layer *head;
        conv_layer *tail;
        
        /*--- General Parameters ---*/
        int     input_width, input_height, input_channels;
        cube    input_X;
        
        void    add_layer(int);
        
    public:
        void    add_conv(int, int, int, string);
        void    add_down(int, int, int, int, string);
        void    feedforward(cube);
        void    backpropagate(cube);
        void    applygrads(float);
        cube    output(void);
        mat     flatten(void);
        
        Convolutional(int,int,int);
        ~Convolutional(); 
};

#endif

