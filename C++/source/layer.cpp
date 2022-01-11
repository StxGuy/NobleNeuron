
#include "layer.hpp"
#include "activ.hpp"

/*---------------------------*/
/* CONSTRUCTOR & DESTRUCTORS */
/*---------------------------*/

layer::layer(void) {
    next = nullptr;
    prev = nullptr;
}

/*---------------------------*/
/*          METHODS          */
/*---------------------------*/

// Feedforward
void layer::feed(mat X) {
    Y = W*X + B;
    Z = activate(Y,activation);
}

void shape(mat M,string s) {
    cout << s << ": " << M.n_rows << "x" << M.n_cols << endl;
}

// Backpropagate
void layer::back(mat X, mat dLdZ) {
    mat dLdY;
        
    dLdY = dLdZ*deactivate(Z,activation);
    dB = dB + dLdY;
    dW = dW + X*dLdY;
    dL = dLdY*W;
}

// Gradient Descent
void layer::gradient_descent(float eta) {
    W = W - eta*dW.t();
    B = B - eta*dB.t();
    
    dW.zeros();
    dB.zeros();
}

