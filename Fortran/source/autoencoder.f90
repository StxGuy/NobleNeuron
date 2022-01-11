
program DenoisingAutoencoder
    use NobleNeuron
    
    implicit none
    
    integer,parameter   :: dp = kind(1.d0)    
    type(NeuralNetwork) :: DAE
    integer             :: i,j
    real                :: r,r1,r2
    real(dp)            :: sample(20,1)
    real(dp)            :: expected(20,1)
    double precision    :: L
    real(dp)            :: gL(1,20)
    
    logical             :: existent
    character(len=50)   :: fname
    
    real(dp),parameter  :: pi = 3.1415926535897932384
    
    !--------------------------------------------------!
    !              Create Neural Network               !
    !--------------------------------------------------!
    DAE = NeuralNetwork(20,"L2","SGD")
    i = DAE%addLayer(5,"tanh")
    i = DAE%addLayer(20,"tanh")
    
    !--------------------------------------------------!
    !                 Train Network                    !
    !--------------------------------------------------!
    L = 1.0
    do while(L > 5E-3)
           
        ! Epoch = 360 runs
        L = 0.0
        do j = 1,360
            ! Input vector = A.sin(w.t + theta)
            call random_number(r1)
            r1 = -pi + r1*2*pi
            do i = 1,20
                call random_number(r)
                r2 = sin((i-1)*2*pi/20)
                sample(i,1) = r2 + r/5  ! Noisy input
                expected(i,1) = r2      ! Clear output
            end do    
                        
            ! Feedforward / Loss / Backpropagation
            call DAE%feedforward(sample)
            L = L + DAE%Loss(expected)
            gL = DAE%gradLoss(expected)
            call DAE%backpropagate(sample,gL)
        end do
        L = L/360
        
        ! Plot loss and apply gradients
        write(*,*) L
        call DAE%applyGrads(0.001)
    end do
    
    !--------------------------------------------------!
    !      Save Results for plotting in Python         !
    !--------------------------------------------------!
    ! Open file
    fname = "python/out.dat"
    inquire(file=fname,exist=existent)
    
    if (existent) then
        open(1,file=fname,status="old")
    else
        open(1,file=fname,status="new")
    end if
    
    ! Create input vector
    do i = 1,20
        r2 = sin((i-1)*2*pi/5)
        sample(i,1) = r2
    end do

    ! Feedforward
    call DAE%feedforward(sample)
    write(1,*) DAE%output()
    close(1)
    
end program
