#include "neural.hpp"
#include "loss.hpp"

int main(void) {
    NeuralNetwork N(2);
    mat X(2,1);
    mat Y(1,1);
    mat dL(1,1);
    float Loss;
    
    cube input_vector(2,1,4);
    cube output_vector(1,1,4);
    
    input_vector(0,0,0) = 0;
    input_vector(1,0,0) = 0;
    input_vector(0,0,1) = 0;
    input_vector(1,0,1) = 1;
    input_vector(0,0,2) = 1;
    input_vector(1,0,2) = 0;
    input_vector(0,0,3) = 1;
    input_vector(1,0,3) = 1;
    
    output_vector(0,0,0) = 0;
    output_vector(0,0,1) = 1;
    output_vector(0,0,2) = 1;
    output_vector(0,0,3) = 0;
    
    
    N.add_layer(3,"Sigmoid");
    N.add_layer(1,"Sigmoid");
    
    do {
        Loss = 0;
        for(int i = 0; i < 4; i ++) {
            X = input_vector.slice(i);
            Y = output_vector.slice(i);
        
            N.feedforward(X);
            Loss += L2(N.output(),Y);
            dL = dL2(N.output(),Y);
            N.backpropagate(dL);
        }
        N.applygrads(0.001);
        cout << "Loss: " << Loss/4 << endl;
    } while(Loss > 0.01);
    
    cout << endl << "Test" << endl;
    cout << "---------------" << endl;
    for(int i = 0; i < 4; i ++) {
        X = input_vector.slice(i);
        N.feedforward(X);
        cout << X(0) << X(1) << ": " << N.output() << endl;
    }    
    
    return 0;
}
    
    
