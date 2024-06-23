using LinearAlgebra;
using MLDatasets, Flux;
import Statistics: mean;

include("Structure.jl");
include("Backward.jl");
include("Forward.jl");
include("Rnn.jl");
include("Operator.jl");


####################################################################################################
# Data preparation

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :)
    y  = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x1dim, y); batchsize, shuffle=true)
end

(trainX, trainY) = only(loader(train_data; batchsize=length(train_data)));
(testX, testY) = only(loader(test_data; batchsize=length(test_data)));

####################################################################################################
# Network preparation

function dense(w, b, x, activation) return activation(w * x .+ b) end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

function initialize_uniform_bias(in_features::Int64, out_features::Int64)
    k = √(1 / in_features)
    return rand(-k:k, out_features)
end

# function xavier_init(in_dim::Int, out_dim::Int)
#     scale = sqrt(2 / (in_dim + out_dim))
#     return randn(out_dim, in_dim) * scale
# end

# function he_init(fan_in, fan_out)
#     std_dev = sqrt(2 / fan_in)
#     weights = randn(Float32, fan_in, fan_out) * std_dev
#     return weights
# end

W_i = Variable(Flux.glorot_normal(64, 784), name="wi")
W_h = Variable(Flux.glorot_normal(64, 64), name="wh")
b_h = Variable(initialize_uniform_bias(1,64), name="bh")
W_o = Variable(Flux.glorot_normal(10,64), name="wo")
b_o = Variable(initialize_uniform_bias(1,10), name="bo")

function net(x, wx, wh, bh, wo, bo, y) #2x2 macierz obrazu, wagi warstw, label
    â = rnn(x, wx, wh, bh)
    â.name = "â"
    ŷ = dense(wo, bo, â, linear)
    ŷ.name = "ŷ"
    # E = cross_entropy_loss(ŷ, y)
    E = mse(ŷ, y)
    E.name = "loss"
    return topological_sort(E), ŷ
end

####################################################################################################
# initialization

x = Variable(trainX, name="x")
y = Variable(trainY, name="y")

graph, out = net(x, W_i, W_h, b_h, W_o, b_o, y)
forward!(graph)
backward!(graph)

####################################################################################################
# Training

epochs=4
training_set_size = 60000

#inicjacja tablicy na funkjcę strat
losses = Float64[] 

@time @allocated for i=1:epochs
    @time @allocated begin
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
            W_i.output -= lr * W_i.gradient
            W_h.output -= lr * W_h.gradient
            W_o.output -= lr * W_o.gradient
            b_h.output -= lr * b_h.gradient

            push!(losses, first(currentloss))
        end
        current_loss = mean(losses[training_set_size*(i-1)+1:training_set_size*i])
        current_accuracy = total == 0 ? 0.0 : correct / total
        println("Epoch: ", i)
        println("Loss: ", current_loss)
        println("Train accuracy: ", round(current_accuracy * 100, digits=1), "%")
    end
end

####################################################################################################
# Testing

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
println("Test accuracy: ", round(test_accuracy * 100, digits=1), "%")