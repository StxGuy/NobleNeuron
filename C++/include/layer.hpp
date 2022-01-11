#ifndef LAYER_NN
#define LAYER_NN

#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

class layer {
    public:
        /*---- Linked List ---*/
        layer   *next;
        layer   *prev;
        
        /*--- General Parameters ---*/
        string  activation;
        int     layerNumber;
        int     input_size, output_size;
        
        /*--- Specific Parameters ---*/
        mat     W, dW;
        mat     B, dB;
        
        /*--- Input & Output ---*/
        mat     Z, Y, dL;
        
        /* METHODS */
        layer();
        
        void    feed(mat);
        void    back(mat,mat);
        void    gradient_descent(float);

};

#endif
