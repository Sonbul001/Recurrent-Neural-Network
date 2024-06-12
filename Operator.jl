include("Structure.jl")

import Base: *
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g) #gradient x, gradient y

Base.Broadcast.broadcasted(-, x::GraphNode) = BroadcastedOperator(-, x)
forward(::BroadcastedOperator{typeof(-)}, x::Matrix{Float64}) = return .-x
backward(::BroadcastedOperator{typeof(-)}, x::Matrix{Float64}, g) = tuple(-g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = let
    grad = exp.(x) .* g
    tuple(grad)
end

Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
backward(node::BroadcastedOperator{typeof(tanh)}, x, g) = let
    grad = g .* (1 .- tanh.(x).^2)
    tuple(grad)
end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = y .* (x .^ (y .- 𝟏))
    Jy = (x .^ y) .* log.(abs.(x))
    tuple(Jx * g, Jy * g)
end

linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x) = return x
backward(::BroadcastedOperator{typeof(linear)}, x, g) = tuple(g)
select(x::GraphNode, index) = BroadcastedOperator(select, x, index)
forward(::BroadcastedOperator{typeof(select)}, x::Vector{Float32}, index::Int) = return x[index]
backward(::BroadcastedOperator{typeof(select)}, x::Vector{Float32}, index::Int, g::Float32) =
let
    result = zeros(Float32, size(x))
    result[index] = g
    tuple(result')
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x::Matrix{Float64}) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x::Matrix{Float64}, g::AbstractMatrix{Float64}) = tuple(((1 ./ x)' .* g)')

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        # y_hat = y_hat .- maximum(y_hat)
        # y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end

mse(ŷ::GraphNode, y::GraphNode) = BroadcastedOperator(mse, ŷ, y)
forward(::BroadcastedOperator{typeof(mse)}, ŷ, y) = let 
    return mean((y .- ŷ) .^ 2)
end
backward(::BroadcastedOperator{typeof(mse)}, ŷ, y, g) = let 
    return tuple(g .* (ŷ - y))
end
# σ(x::GraphNode) = BroadcastedOperator(σ, x)
# forward(::BroadcastedOperator{typeof(σ)}, x) = return 1 ./ (1 .+ exp.(-x))
# backward(node::BroadcastedOperator{typeof(σ)}, x, g) = let
#     y = node.output
#     dx = g .* y .* (1 .- y)
#     tuple(dx)
# end