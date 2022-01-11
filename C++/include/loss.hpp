#ifndef LOSS_NN
#define LOSS_NN

#include <armadillo>

using namespace arma;

float L2(mat, mat);
mat dL2(mat, mat);

#endif
