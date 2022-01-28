#ifndef CONV_LAYER_NN
#define CONV_LAYER_NN

#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

class conv_layer {
    public:
        /*--------------------*/
        /*---- Linked List ---*/
        /*--------------------*/
        conv_layer   *next;
        conv_layer   *prev;
        
        /*-------------------------*/
        /*--- Common Parameters ---*/
        /*-------------------------*/
        // Layer
        int     layerNumber;
        string  kind;
        
        // Image
        int     input_width, input_height;
        int     output_width, output_height;
        
        // Filter
        int     filter_width, filter_height;
        
        // Common
        int     depth, channels;
        
        // Output
        cube    Z;
        cube    dL;             // dLdX
                
        /*-------------------*/
        /*--- Convolution ---*/
        /*-------------------*/
        string  activation;
                
        field<cube>     F;      // Filter
        cube            B;      // Bias
        cube            Y;      // Output without activation
        cube            dB;     // dLdB
        field<cube>     dF;     // dLdF
        
        /*---------------*/
        /*--- Pooling ---*/
        /*---------------*/
        string  pooling;
        int     stride_width, stride_height;
        cube    map;    // Map for backpropagate through maxpooling
                        
        /* METHODS */
        conv_layer();
        
        void    feed_down(cube);
        void    back_down(cube);
        void    feed_convolution(cube);
        void    back_conv(cube, cube);
        void    gradient_descent(float);
};

#endif
