using LinearAlgebra
using MLDatasets, Flux
import Statistics: mean

include("Structure.jl")
include("Backward.jl")
include("Forward.jl")
include("Rnn.jl")
include("Operator.jl")

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :) # reshape 28×28 pixels into a vector of pixels
    y  = Flux.onehotbatch(data.targets, 0:9) # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x1dim, y); batchsize, shuffle=true)
end

(trainX, trainY) = only(loader(train_data; batchsize=length(train_data)))
(testX, testY) = only(loader(test_data; batchsize=length(test_data)))

#warstwy fully conected
function dense(w, b, x, activation) return activation(w * x .+ b) end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

function initialize_uniform_bias(in_features::Int64, out_features::Int64)
    k = √(1 / in_features)
    return rand(-k:k, out_features)
end

function xavier_init(in_dim::Int, out_dim::Int)
    scale = sqrt(2 / (in_dim + out_dim))
    return randn(out_dim, in_dim) * scale
end

function he_init(fan_in, fan_out)
    std_dev = sqrt(2 / fan_in)
    weights = randn(Float32, fan_in, fan_out) * std_dev
    return weights
end

# input weights, hidden weights, output weights
W_i = Variable(xavier_init(784, 64), name="wi")
W_h = Variable(xavier_init(64, 64), name="wh")
b_h = Variable(initialize_uniform_bias(1, 64), name="bh")
W_o = Variable(xavier_init(64,10), name="wo")
b_o = Variable(initialize_uniform_bias(10, 1), name="bo")

# W_i = Variable(he_init(64, 784), name="wi")
# size(W_i.output)
# W_h = Variable(he_init(64, 64), name="wh")
# b_h = Variable(initialize_uniform_bias(1, 64), name="bh")
# W_o = Variable(he_init(10,64), name="wo")
# b_o = Variable(initialize_uniform_bias(10, 1), name="bo")

function mean_squared_loss(y, ŷ)
    return Constant(0.5) .* (y .- ŷ) .^ Constant(2)
end

Variable(0.1 * ones(64, 1)).output[:,1]

function net(x, wx, wh, bh, wo, bo, y) #2x2 macierz obrazu, wagi warstw, label
    â = rnn(x, Variable(0.1 * ones(64, 1)), wx, wh, bh, Variable(zeros(64, size(x.output, 2)))) #stały filtr
    â.name = "â"
    ŷ = dense(wo, bo, â, linear)
    ŷ.name = "ŷ"
    E = cross_entropy_loss(ŷ, y)
    # E = mean_squared_loss(y, ŷ)
    # E = mse(ŷ, y)
    E.name = "loss"
    return topological_sort(E), ŷ
end


x = Variable(trainX, name="x")
y = Variable(trainY, name="y")

graph, out = net(x, W_i, W_h, b_h, W_o, b_o, y)
forward!(graph)
# backward!(graph)



epochs=4
training_set_size = 60000

#inicjacja tablicy na funkjcę strat
losses = Float64[] 

for i=1:epochs
    total = 0
    correct = 0
    for j=1:training_set_size
        x = Variable(trainX[:,j], name="x")
        y = Variable(trainY[:,j], name="y")
        graph, result = net(x, W_i, W_h, b_h, W_o, b_o, y)
        currentloss = forward!(graph)

        total += 1
        if (argmax(result.output[:]) == argmax(y.output))
            correct += 1
        end

        backward!(graph)
        lr = 0.0
        i < 5 ? lr = 0.001 : lr = 0.01
        for i in 1:size(W_i.output, 1)
            W_i.output[i, :] .-= lr * W_i.gradient
        end
        for i in 1:size(W_h.output, 1)
            W_h.output[i, :] .-= lr * W_h.gradient
        end
        W_o.output -= lr * W_o.gradient
        push!(losses, first(currentloss))
    end
    current_loss = mean(losses[training_set_size*(i-1)+1:training_set_size*i])
    current_accuracy = total == 0 ? 0.0 : correct / total
    println("Epoch: ", i)
    println("Loss: ", current_loss)
    println("Accuracy: ", round(current_accuracy * 100, digits=1), "%")
end

####################################################################################################

# Assuming you have a separate test dataset testX and testY

test_set_size = size(testX, 2)
total = 0
correct = 0

for j in 1:test_set_size
    x = Variable(testX[:, j], name="x")
    y = Variable(testY[:, j], name="y")
    graph, result = net(x, W_i, W_h, b_h, W_o, b_o, y)
    forward!(graph)
    
    total += 1
    if argmax(result.output[:]) == argmax(y.output)
        correct += 1
    end
end

test_accuracy = correct / total
println("Test Accuracy: ", round(test_accuracy * 100, digits=1), "%")



####################################################################################################

# function predict(x, wx, wh, bh, wo, bo)
#     â = rnn(x, Variable(0.1 * ones(64, 1)), wx, wh, bh, Variable(0.1 * ones(64, size(x.output, 2)))) #stały filtr
#     â.name = "â"
#     ŷ = dense(wo, bo, â, linear)
#     ŷ.name = "ŷ"
#     pred = return_prediction(ŷ)
#     pred.name = "pred"
#     return topological_sort(pred) #kom
# end
# W_i.output
# W_i.gradient
# W_h.output
# W_o.output

# x = Variable(trainX[:,3000], name="x")
# y = Variable(trainY[:,3000], name="y")
# trainY[:,3000]
# trainX[:,3000]
# graph, result = net(x, W_i, W_h, b_h, W_o, b_o, y)
# result.output
# argmax(result.output[:])
# argmax(y.output)
# forward!(graph)
# argmax(result.output) == argmax(y.output)

# argmax(result.output)
# total = 0
# correct = 0
# for b=1:60000
#     x = Variable(trainX[:,b], name="x")
#     y = Variable(trainY[:,b], name="y")
#     graph, result = net(x, W_i, W_h, b_h, W_o, b_o, y)
#     forward!(graph)
#     total += 1
#     if (argmax(result.output[:]) == argmax(y.output))
#         correct += 1
#     end
#     # println(forward!(result)) #predict ma to samo co architektura bez funkcji strat, obliczana jest tutaj wartość bo fwd ma compute
#     # println(trainY[b])
# end

# function getAccuracy(total, correct)
#     acc = total == 0 ? 0.0 : correct / total
#     println(round(acc * 100, digits=1), "%")
# end
# getAccuracy(total, correct)

# b=1
# a=trainX[:,b]