include("Structure.jl")

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = begin
    if isnothing(node.gradient)
        node.gradient = gradient
    else 
        node.gradient .+= gradient
    end
    node.gradient = max.(-0.5, min.(0.5, node.gradient))
end
#aktualizacja wartości gradientu dla danego węzła lub jego inicjalizacja

function backward!(order::Vector; seed=1.0) #wejscie to vector posortowanych węzłów
    result = last(order)
    result.gradient = seed #zapoczątkowanie obliczeń
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order) #odwrócenie kolejności przetrzymywanej w order i wykonanie backward pass od tyłu
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient) #zastosowanie funkcji backward dla zdefiniowanej operacji
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end