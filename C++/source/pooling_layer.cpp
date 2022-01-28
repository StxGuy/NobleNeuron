#include "conv_layer.hpp"

/*---------------------------*/
/*          METHODS          */
/*---------------------------*/

// Index of global maximum element of a matrix
vec gindmax(mat X) {
    urowvec v;
    uword   i,j;
    vec     t(2);
    
    v = index_max(X);
    j = index_max(max(X));
    i = v(j);
    
    t(0) = i;
    t(1) = j;
    
    return t;
}    

// Feedforward downsample layer
void conv_layer::feed_down(cube X) {
    int j_from, j_to;
    int i_from, i_to;
    int a,b;
    mat t;
    vec v;
    
    this->map.zeros();
    
    
    for(int k = 0; k < this->depth; k ++) {
        for(int j = 0; j < this->output_height; j ++) {
            j_from = j*this->stride_height;
            j_to = j_from + this->filter_height - 1;
            if (j_to > this->input_height)
                j_to = this->input_height;
            for(int i = 0; i < this->output_width; i ++) {
                i_from = i*this->stride_width;
                i_to = i_from + this->filter_width - 1;
                if (i_to > this->input_width)
                    i_to = this->input_width;
                
                t = X.slice(k).submat(i_from,j_from,i_to,j_to);
                
                if (this->pooling == "max") {
                    this->Z(i,j,k) = max(max(t));
                    v = gindmax(t);
                    a = i_from + v(0);
                    b = j_from + v(1);
                    this->map(a,b,k) = 1.0;
                }
                else if (this->pooling == "average")
                    this->Z(i,j,k) = mean(mean(t));
            }
        }
    }
}
                
// Backpropagate downsample layer
void conv_layer::back_down(cube dLdZ) {
    int j_from, j_to;
    int i_from, i_to;
    float e2;
    
    e2 = 1.0/(this->filter_width*this->filter_height);
    
    for(int k = 0; k < this->depth; k ++) {
        for(int j = 0; j < this->output_height; j ++) {
            j_from = j*this->stride_height;
            j_to = j_from + this->filter_height;
            if (j_to > this->input_height)
                j_to = this->input_height;
            for(int i = 0; i < this->output_width; i ++) {
                i_from = i*this->stride_width;
                i_to = i_from + this->filter_width;
                if (i_to > this->input_width)
                    i_to = this->input_width;
                                
                for(int q = j_from; q < j_to; q ++) {
                    for(int p = i_from; p < i_to; p ++) {
                        if (this->pooling == "average") {
                            this->dL(p,q,k) = dLdZ(i,j,k)*e2;
                        }
                        if (this->pooling == "max") {
                            this->dL(p,q,k) = dLdZ(i,j,k)*map(p,q,k);
                            
                        }
                    }
                }
            }
        }
    }
}
