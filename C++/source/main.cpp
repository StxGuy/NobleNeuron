#include "NobleNeuron.hpp"
#include "loss.hpp"

using namespace std;

int main(void) {
    Convolutional C(4,4,1);
    FeedForward F(4);
    cube X(4,4,1);
    mat Y(1,1);
    mat dL(1,1);
   
    int r1,r2;
    float L;
    
    
    // Create neural network
    C.add_conv(2,2,1,"ReLU");
    C.add_conv(2,2,4,"ReLU");
    C.add_conv(2,2,1,"ReLU");
    C.add_down(2,2,2,2,"max");
    F.add_layer(1,"sigmoid");
    
    do {
        L = 0;
        for(int epoch = 0; epoch < 100; epoch ++) {
            // Create vertical or horizontal lines
            r1 = rand()%2;
            r2 = rand()%4;
                
            X.zeros();
            for(int i = 0; i < 4; i ++) {
                if (r1 == 0) {
                    X(r2,i,0) = 1.0;
                    Y(0,0) = 1.0;
                }
                else {
                    X(i,r2,0) = 1.0;
                    Y(0,0) = 0.0;
                }
            }
        
            // Feedforward
            C.feedforward(X);
            F.feedforward(C.flatten());

            // Loss
            L += L2(F.output(),Y);
            dL = dL2(F.output(),Y);
                        
            // Backpropagation
            F.backpropagate(dL);
            C.backpropagate(F.unflatten(2,2));
        }
        L = L/100;
        cout << L << endl;
    
        F.applygrads(0.001);
        C.applygrads(0.001);
    } while(L > 0.1);      
    
        
    X.zeros();
    for(int i = 0; i < 4; i ++) {
        X(2,i,0) = 1.0;
    }

    // Feedforward
    C.feedforward(X);
    F.feedforward(C.flatten());
    cout << "ver: " << F.output() << endl;

    X.zeros();
    for(int i = 0; i < 4; i ++) {
        X(i,2,0) = 1.0;
    }

    // Feedforward
    C.feedforward(X);
    F.feedforward(C.flatten());
    cout << "hor: " << F.output() << endl;

    
    
    return 0;
}
    
    
