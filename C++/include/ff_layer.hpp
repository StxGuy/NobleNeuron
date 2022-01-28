#ifndef FF_LAYER_NN
#define FF_LAYER_NN

#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

class ff_layer {
    public:
        /*---- Linked List ---*/
        ff_layer   *next;
        ff_layer   *prev;
        
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
        ff_layer();
        
        void    feed(mat);
        void    back(mat,mat);
        void    gradient_descent(float);
};

#endif
