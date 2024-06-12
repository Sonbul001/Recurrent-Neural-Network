include("Structure.jl")

rnn(xs::GraphNode, h_init::GraphNode, W_x::GraphNode, W_h::GraphNode, b::GraphNode, hs::GraphNode) = BroadcastedOperator(rnn, xs, h_init, W_h, W_x, b, hs)

# forward(::BroadcastedOperator{typeof(rnn)}, xs, h_init, W_h, W_x, b) = let
#     println(size(xs))
#     println(size(h_init))
#     println(size(W_h))
#     println(size(W_x))
#     println(size(b))
#     h = h_init
#     hs = zeros(size(h, 1), size(xs, 2))
#     for t in 1:size(xs, 2)  # Loop over columns (time steps)
#         x_t = xs[:, t]  # Each input vector of size 784
#         h = tanh.(W_h * h .+ W_x * x_t .+ b)
#         hs[:, t] = h[:]  # Store the hidden state in the corresponding column
#     end
#     return hs
# end
forward(::BroadcastedOperator{typeof(rnn)}, xs, h_init, W_h, W_x, b, hs) = let
    h = h_init  # Initialize the hidden state correctly
    # hs = zeros(size(h, 1), size(xs, 2))  # Hidden states for all time steps
    for t in 1:size(xs, 2)  # Loop over columns (time steps)
        x_t = xs[:, t]  # Each input vector of size 784
        h = tanh.(W_h * h .+ W_x * x_t .+ b)
        hs[:, t] = h  # Store the hidden state in the corresponding column
    end
    return hs
end

backward(::BroadcastedOperator{typeof(rnn)}, xs, h_init, W_h, W_x, b, hs, grads ) = let
    # println(size(grads))
    T = size(xs, 2)  # Number of time steps
    dW_h = zeros(size(W_h))
    dW_x = zeros(size(W_x))
    db = zeros(size(b))
    dh_next = zeros(size(hs, 1))  # Initialize with zeros
    dxs = zeros(size(xs))  # To store the gradients with respect to the inputs

    for t in T:-1:1
        # Retrieve the input and hidden state for this time step
        x_t = xs[:, t]
        h_t = hs[:, t]
        
        # Compute the gradient of the hidden state with respect to the previous hidden state
        grad_h = grads[t] .+ dh_next  # Include gradient from the next time step
        
        # Compute the gradient of the tanh activation function
        tanh_prime = 1 .- h_t.^2  # derivative of tanh output
        
        # Compute the gradients for this time step
        delta = grad_h .* tanh_prime
        
        dh_prev = W_h' * delta
        dx = W_x' * delta
        dW_ht = delta * h_t'
        dW_xt = delta * x_t'
        dbt = delta
        
        # Update the gradients
        dW_h += dW_ht
        dW_x += dW_xt
        db += dbt
        
        # Store the gradient with respect to the input
        dxs[:, t] = dx
        
        # Prepare the gradient for the previous time step
        dh_next = dh_prev
    end

    return tuple(dW_h, dW_x, db, dxs)
end
