#Definiowanie strukur do wykorzystania przy tworzeniu grafu
abstract type GraphNode end
abstract type Operator <: GraphNode end #dziedziczenie po grafie, wykorzystywany do różnych operacji

#stała w grafie, output przechowywuje jej wartość, parametryczna stuktura
struct Constant{T} <: GraphNode
    output :: T
end

#zmienna, gradient przechowuje pochodną po pewnej wartości
mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

#skalar input jest zamieniany na skalar output
mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

#przyjmuje input o dowolnym wymiarze, przydatny potem do sledzenia różnych wersji forward i backward
mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end


#drukowanie graphu, io - input output stream
import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end


function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node) #dodanie do listy z już odwiedzonymi węzłami
        push!(order, node) #utworzenie listy ze wszystkimi węzłami
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode) #sortowanie elemntów przechowane w 
    visited = Set()
    order = Vector()
    visit(head, visited, order) #head to początek graphu
    return order #zwraca posortowane elementy tablicy w postaci array
end