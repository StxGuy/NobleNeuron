using PyPlot

#---------------------------#
#           MODEL           #
#---------------------------#
mutable struct DenseLayer
    x           :: Matrix
    W           :: Matrix
    b           :: Matrix
    z           :: Matrix
    activation  :: String
    ∂L∂W        :: Matrix
    ∂L∂b        :: Matrix
end

#-------------------------#
# Dense Layer Initializer #
#-------------------------#
function Dense(input_size,output_size,act)
    x = zeros(input_size,1)
    W = (rand(output_size,input_size) .- 0.5) ./ output_size
    b = (rand(output_size,1) .- 0.5) 
    z = zeros(output_size,1)
    dW = zeros(input_size,output_size)
    db = zeros(1,output_size)
        
    return DenseLayer(x,W,b,z,act,dW,db)
end

#----------------------#
# Activation Functions #
#----------------------#
function sigmoid(x)
    return 1.0/(1.0 + exp(-x))
end    

#-------------------------#
# FeedForward Dense Layer #
#-------------------------#
function forwardDense!(x,layer :: DenseLayer)
    setfield!(layer,:x,x)
    y = layer.W*x + layer.b
    
    if (layer.activation == "tanh")
        z = tanh.(y)
    end
    if (layer.activation == "sig")
        z = sigmoid.(y)
    end
    
    setfield!(layer,:z,z)
    
    return z
end

#---------------------------#
# BackPropagate Dense Layer #
#---------------------------#
function backDense!(∂L∂z,layer :: DenseLayer)
    if (layer.activation == "tanh")
        D = 1.0 .- layer.z'.^2
    end
    if (layer.activation == "sig")
        D = (layer.z .* (1.0 .- layer.z))'
    end
    
    ∂L∂y = ∂L∂z.*D
    setfield!(layer,:∂L∂b,layer.∂L∂b + ∂L∂y)
    setfield!(layer,:∂L∂W,layer.∂L∂W + layer.x*∂L∂y)
    ∂L∂x = ∂L∂y*layer.W
    
    return ∂L∂x
end

#------------------#
# Gradient Descend #
#------------------#
function Gradient!(layer :: DenseLayer, η)
    setfield!(layer,:W, layer.W - η*layer.∂L∂W')
    setfield!(layer,:b, layer.b - η*layer.∂L∂b')
    
    i,j = size(layer.∂L∂W)
    z = zeros(i,j)
    setfield!(layer,:∂L∂W,z)
    
    i,j = size(layer.∂L∂b)
    z = zeros(i,j)
    setfield!(layer,:∂L∂b,z)
     
    return nothing
end

#====================================================#
#                       MAIN                         #
#====================================================#
Layer1 = Dense(2,3,"sig")
Layer2 = Dense(3,1,"sig")

X = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
X = [0.0 0.0 1.0 1.0
     0.0 1.0 0.0 1.0]
Y = [0.0; 1.0; 1.0; 0.0]

loss_y = []
loss = 1.0
while(loss > 0.001)
    global loss = 0
    for it in 1:4
        # Generate data
        r = rand(1:4)
        x = reshape(X[:,r],2,1) + 0.01*(rand(2,1) .- 0.5)
        y = reshape([Y[r]],1,1)
                
        # Feedforward
        z1 = forwardDense!(x,Layer1)
        z2 = forwardDense!(z1,Layer2)

        # Compute loss and its gradient
        D = z2 - y
        loss += D[1,1]^2
            
        # Backpropagate
        D2 = backDense!(D,Layer2)
        D1 = backDense!(D2,Layer1)
    end
    loss /= 4
    push!(loss_y,loss)
    println(loss)

    # Apply gradient descend
    Gradient!(Layer1,0.01)
    Gradient!(Layer2,0.01)
end

# Print results
for i in 1:4
    x = reshape(X[:,i],2,1) + 0.01*(rand(2,1) .- 0.5)
    z1 = forwardDense!(x,Layer1)
    ŷ = forwardDense!(z1,Layer2)
    println(round(x[1,1]),",",round(x[2,1]),": ",ŷ)
end

plot(loss_y)
show()
