include("Structure.jl")

rnn(xs::GraphNode, W_x::GraphNode, W_h::GraphNode, b::GraphNode) = BroadcastedOperator(rnn, xs, W_x, W_h, b)

forward(::BroadcastedOperator{typeof(rnn)}, xs, W_x, W_h, b) = let
    h = 0.1 * ones(64, 1)
    hs = zeros(64, size(xs, 2))
    h_cache = similar(h)
    for t in 1:size(xs, 2)
        x_t = xs[:, t]
        mul!(h_cache, W_h, h)
        h_cache .+= W_x * x_t .+ b
        @. h = tanh(h_cache)
        hs[:, t] = h
    end
    return hs
end

backward(::BroadcastedOperator{typeof(rnn)}, xs, W_x, W_h, b, grads) = let
    T = size(xs, 2)
    dW_h = zeros(size(W_h))
    dW_x = zeros(size(W_x))
    db = zeros(size(b))
    dh_next = zeros(64, 1)
    dxs = zeros(size(xs))
    h = 0.1 * ones(64, 1)
    hs = zeros(64, size(xs, 2))
    h_cache = similar(h)
    
    for t in 1:T
        x_t = xs[:, t]
        mul!(h_cache, W_h, h)
        h_cache .+= W_x * x_t .+ b
        @. h = tanh(h_cache)
        hs[:, t] = h
    end
    
    tanh_prime = zeros(64, 1)
    delta = zeros(64, 1)
    
    for t in T:-1:1
        x_t = xs[:, t]
        h_t = hs[:, t]

        grad_h = grads[:, t] .+ dh_next

        @. tanh_prime = 1 - h_t^2
        @. delta = grad_h * tanh_prime

        dW_h .+= delta * h_t'
        dW_x .+= delta * x_t'
        db .+= delta

        dxs[:, t] = W_x' * delta

        dh_next .= W_h' * delta
    end
    
    return dxs, dW_x, dW_h, db
end


