#include "loss.hpp"

float L2(mat output, mat expected) {
    return accu(square(output-expected));
}

mat dL2(mat output, mat expected) {
    return (output-expected).t();
}
