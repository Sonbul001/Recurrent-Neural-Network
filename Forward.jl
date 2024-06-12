include("Structure.jl")

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing
#wyzerowanie pochodnej dla różnych typów

compute!(node::Constant) = nothing #wartości są już wiadome
compute!(node::Variable) = nothing #wartości są już obliczone
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)
#wywołanie różnych funkcji forward określonych dla różnych operatorów

function forward!(order::Vector)
    for node in order
        compute!(node) #obliczenie wartości dla konkretnego węzła
        reset!(node) #wyzerowanie pochodnych
    end
    return last(order).output #ostatni element w liście, koniec graphu
end