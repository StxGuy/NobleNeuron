#include "activ.hpp"

/*--- Activate function: Z = sigma(Y) ---*/
mat activate(mat Y,string activation) {
    mat Z;

    transform(activation.begin(),activation.end(),activation.begin(),:: tolower);


    if (activation == "relu")
        Z = clamp(Y,0,Y.max());
    else if (activation == "sigmoid")
        Z = 1.0/(1.0+trunc_exp(-Y));
    else if (activation == "tanh")
        Z = tanh(Y);
    else
        Z = Y;
    
    
    return Z;
}

/* Derivative of the activation function
 * dZdY = (d/dY)sigma(Y) */
mat deactivate(mat Z, string activation) {
    mat dZdY;
    
    transform(activation.begin(),activation.end(),activation.begin(),:: tolower);
    
    
    if (activation == "relu")
        dZdY = Z.transform([](float val) {return val > 0 ? 1:0;});
    else if (activation == "sigmoid")
        dZdY = Z%(1.0 - Z); 
    else if (activation == "tanh")
        dZdY = 1.0 - Z%Z;
    else
        dZdY = Z;
    
    return diagmat(dZdY.col(0));
}
