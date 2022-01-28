
#include "ff_layer.hpp"
#include "activ.hpp"

/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTORS */
/*---------------------------*/

ff_layer::ff_layer(void) {
    next = nullptr;
    prev = nullptr;
}

/*---------------------------*/
/*          METHODS          */
/*---------------------------*/

// Feedforward
void ff_layer::feed(mat X) {
    this->Y = this->W*X + this->B;
    Z = activate(this->Y,activation);
}

void shape(mat M,string s) {
    cout << s << ": " << M.n_rows << "x" << M.n_cols << endl;
}

// Backpropagate
void ff_layer::back(mat X, mat dLdZ) {
    mat dLdY;
        
    dLdY = dLdZ*deactivate(this->Z,activation);
    this->dB += dLdY;
    this->dW += X*dLdY;
    this->dL = dLdY*W;
}

// Gradient Descent
void ff_layer::gradient_descent(float eta) {
    this->W = this->W - eta*this->dW.t();
    this->B = this->B - eta*this->dB.t();
    
    this->dW.zeros();
    this->dB.zeros();
}

