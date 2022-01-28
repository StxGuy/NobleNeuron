#include "conv_layer.hpp"
#include "activ.hpp"

/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTOR  */
/*---------------------------*/

conv_layer::conv_layer(void) {
    next = nullptr;
    prev = nullptr;
}

/*********************************************************************************/
/*                               MISCELLANEOUS                                   */
/*********************************************************************************/

// Transpose of a cube
cube ctransp(cube X) {
    cube Y(X.n_cols,X.n_rows,X.n_slices);
    
    for(uword i = 0; i < X.n_slices; i ++)
        Y.slice(i) = X.slice(i).t();
    
    return Y;
}

// Function to mirror reflect a matrix
// F_xyz -> F_-y,-x,z
mat mirror_reflect(mat X) {
    return trans(fliplr(flipud(X)));
}


// row, column, slice, stack

/*********************************************************************************/
/*                                   METHODS                                     */
/*********************************************************************************/

// Feedforward convolution layer
void conv_layer::feed_convolution(cube X) {
    // For loop over out channels
    for(int k = 0; k < this->channels; k ++) {
        this->Y.slice(k) = this->B.slice(k);
        // For loop over in channels
        for(int p = 0; p < this->depth; p ++) {
            this->Y.slice(k) += conv2(X.slice(p),this->F(k,0).slice(p),"same");
        }
    
        this->Z.slice(k) = activate(this->Y.slice(k), this->activation);
    }
}

// Backpropagate convolution layer
void conv_layer::back_conv(cube X, cube dLdZ) {
    cube    dLdY(this->output_width,this->output_height,this->channels);
    mat     W;
    mat     t;
    int     lix,lax,liy,lay;
    
        
    dLdY = dLdZ%deactivate(ctransp(dLdZ),this->activation);
    
    // dLdB
    this->dB += dLdY;
    
    // dLdX
    // For loop over out channels
    for(int c = 0; c < this->depth; c ++) {
        // For loop over in channels
        this->dL.slice(c).zeros();
        for(int k = 0; k < this->channels; k ++) {
            W = mirror_reflect(this->F(k,0).slice(c));
            this->dL.slice(c) += conv2(dLdY.slice(k), W, "same");
        }
    }
    
    // dLdF
    lix = (this->input_width - (this->filter_width-1)) >> 1;
    lax = lix + this->filter_width - 1;
    liy = (this->input_height - (this->filter_height-1)) >> 1;
    lay = liy + this->filter_height - 1;
       
    for(int d = 0; d < this->channels; d ++) {
        for(int c = 0; c < this->depth; c ++) {
            t = conv2(dLdY.slice(d), mirror_reflect(X.slice(c)), "same");
            this->dF(d,0).slice(c) += t.submat(liy,lix,lay,lax);
        }
    }
}
    
// Gradient Descent
void conv_layer::gradient_descent(float eta) {
    for(int k = 0; k < this->channels; k ++) {
        this->F(k,0) -= eta*ctransp(this->dF(k,0));
        this->dF(k,0).zeros();
    }
    
    this->B -= eta*ctransp(dB);
    this->dB.zeros();
}

