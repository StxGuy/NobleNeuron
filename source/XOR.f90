
program teste
    use NobleNeuron
    
    implicit none
    
    integer,parameter   :: dp = kind(1.d0)    
    type(NeuralNetwork) :: XOR
    integer             :: i,j
    real                :: r,r1,r2
    real(dp)            :: sample(2,1)
    real(dp)            :: expected(1,1)
    double precision    :: L
    real(dp)            :: gL(1,1)
    
    !--------------------------------------------------!
    !              Create Neural Network               !
    !--------------------------------------------------!
    XOR = NeuralNetwork(2,"L2","SGD")
    i = XOR%addLayer(3,"Sigmoid")
    i = XOR%addLayer(1,"Sigmoid")
    
    !--------------------------------------------------!
    !                 Train Network                    !
    !--------------------------------------------------!
    L = 1.0
    do while(L > 0.01)
        ! Epoch = 4 runs
        L = 0.0
        do j = 1,4
            ! Input vector: 00,01,10,11 + noise
            call random_number(r)
            call random_number(r1)
            call random_number(r2)
            r1 = (r1-0.5)/5
            r2 = (r2-0.5)/5
            i = 1 + floor(4*r)
                
            select case(i)
                case(1)
                    sample = reshape([real(dp) :: r1,r2], shape(sample))
                    expected = reshape([real(dp) :: 0], shape(expected))
                case(2)
                    sample = reshape([real(dp) :: r1,1+r2], shape(sample))
                    expected = reshape([real(dp) :: 1], shape(expected))
                case(3)
                    sample = reshape([real(dp) :: 1+r1,r2], shape(sample))
                    expected = reshape([real(dp) :: 1], shape(expected))
                case(4)
                    sample = reshape([real(dp) :: 1+r1,1+r2], shape(sample))
                    expected = reshape([real(dp) :: 0], shape(expected))
            end select
            
            ! Feedforward / Loss / Backpropagation
            call XOR%feedforward(sample)
            L = L + XOR%Loss(expected)
            gL = XOR%gradLoss(expected)
            call XOR%backpropagate(sample,gL)
        end do
        
        ! Plot loss and apply gradients
        write(*,*) L/4
        call XOR%applyGrads(0.001)
    end do
    
    !--------------------------------------------------!
    !                  Show Results                    !
    !--------------------------------------------------!
    write(*,*)
    write(*,*) "=========================================="
    write(*,*) "                 RESULTS                  "
    write(*,*) "=========================================="
    write(*,*)
    
    sample = reshape([real(dp) :: 0,0],shape(sample))
    call XOR%feedforward(sample)
    write(*,*) "00 -> ",XOR%output()
    
    sample = reshape([real(dp) :: 0,1],shape(sample))
    call XOR%feedforward(sample)
    write(*,*) "01 -> ",XOR%output()
    
    sample = reshape([real(dp) :: 1,0],shape(sample))
    call XOR%feedforward(sample)
    write(*,*) "10 -> ",XOR%output()
    
    sample = reshape([real(dp) :: 1,1],shape(sample))
    call XOR%feedforward(sample)
    write(*,*) "11 -> ",XOR%output()
    write(*,*)
    
end program
