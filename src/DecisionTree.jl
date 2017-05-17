__precompile__()

module DecisionTree

using Compat, StatsBase

import Base: length, convert, promote_rule, show, start, next, done

export Leaf, Node, Ensemble, print_tree, depth, build_stump, build_tree,
       prune_tree, apply_tree, apply_tree_proba, nfoldCV_tree, build_forest,
       apply_forest, apply_forest_proba, nfoldCV_forest, build_adaboost_stumps,
       apply_adaboost_stumps, apply_adaboost_stumps_proba, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix, mean_squared_error,
       R2, _int, variable_importance

# ScikitLearn API
export DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
       RandomForestRegressor, AdaBoostStumpClassifier,
       # Should we export these functions? They have a conflict with
       # DataFrames/RDataset over fit!, and users can always
       # `using ScikitLearnBase`.
       predict, predict_proba, fit!, get_classes

#####################################
##### Compatilibity Corrections #####

const Range1 = Range
_int(x) = map(y->round(Integer, y), x)
float(x) = map(Float64, x)

_squeeze(m::Matrix, i::Int) = squeeze(m, i)
_squeeze(v::Vector, i::Int) = v
# `.!arr` is invalid in 0.5, and `!arr` triggers a warning in 0.6.
neg(arr) = map(!, arr)


###########################
########## Types ##########

const NO_BEST=(0,0)

immutable Leaf
    majority::Any
    values::Vector
end

immutable Node
    featid::Integer
    featval::Any
    left::Union{Leaf,Node}
    right::Union{Leaf,Node}
    samples::Integer
end

const LeafOrNode = Union{Leaf,Node}

immutable Ensemble
    trees::Vector{Node}
    seeds::Vector{UInt32}
    labels::Int
end

convert(::Type{Node}, x::Leaf) = Node(0, nothing, 0., x, Leaf(nothing,[nothing]))
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

### Iterators ###
# Iterator interface to trees: iterates through nodes in depth-first order.
start(tree::Node)::Vector{LeafOrNode} = [tree]
done(tree::Node, state::Vector{LeafOrNode}) = isempty(state)
function next(tree::Node, state::Vector{LeafOrNode})
  n = pop!(state)
  _push_children!(state, n)
  return n, state
end
function _push_children!(state::Vector{LeafOrNode}, n::Node)
  push!(state, n.right)
  push!(state, n.left)
end
function _push_children!(state::Vector{LeafOrNode}, n::Leaf)
end

# Iterator interface over unique ranges of values in a vector.
immutable UniqueRanges
    v::AbstractVector
end

start(u::UniqueRanges) = 1
done(u::UniqueRanges, s) = done(u.v, s)
next(u::UniqueRanges, s) = (val = u.v[s];
                            t = searchsortedlast(u.v, val, s, length(u.v), Base.Order.Forward);
                            ((val, s:t), t+1))

##############################
########## Includes ##########

include("measures.jl")
include("classification.jl")
include("regression.jl")
include("scikitlearnAPI.jl")


#############################
########## Methods ##########

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)

depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))

function print_tree(leaf::Leaf, depth=-1, indent=0)
    matches = find(leaf.values .== leaf.majority)
    ratio = string(length(matches)) * "/" * string(length(leaf.values))
    println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::Node, depth=-1, indent=0)
    if depth == indent
        println()
        return
    end
    println("Feature $(tree.featid), Threshold $(tree.featval)")
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, depth, indent + 1)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, depth, indent + 1)
end

function show(io::IO, leaf::Leaf)
    println(io, "Decision Leaf")
    println(io, "Majority: $(leaf.majority)")
    print(io,   "Samples:  $(length(leaf.values))")
end

function show(io::IO, tree::Node)
    println(io, "Decision Tree")
    println(io, "Split:    [$(tree.featid)] < $(tree.featval)")
    println(io, "Leaves:   $(length(tree))")
    print(io,   "Depth:    $(depth(tree))")
end

function show(io::IO, ensemble::Ensemble)
    println(io, "Ensemble of Decision Trees")
    println(io, "Trees:      $(length(ensemble))")
    println(io, "Avg Leaves: $(mean([length(tree) for tree in ensemble.trees]))")
    print(io,   "Avg Depth:  $(mean([depth(tree) for tree in ensemble.trees]))")
end


function sample_weights(ens::Ensemble, itree::Integer)
    rng = MersenneTwister(ens.seeds[itree])
    nsamples = samples(ens)
    nlabels = ens.labels
    inds = rand(rng, 1:nlabels, nsamples)
    return mapslices(x->counts(x, nlabels), inds, 1)
end

ib_sample(ens::Ensemble, itree::Integer)  = sample_weights(ens, itree) .>  0
oob_sample(ens::Ensemble, itree::Integer) = sample_weights(ens, itree) .== 0

samples(ens::Ensemble) = isempty(ens.trees) ? 0 : samples(ens.trees[1])
samples(node::Node) = node.samples
samples(leaf::Leaf) = length(leaf.values)
variance(node::Node) = variance(node.left) + variance(node.right)
variance(leaf::Leaf) = var(leaf.values)

function variable_importance(tree::Node; impurity::Function = variance)
    importance = Dict{Integer, Float64}()
    for node in tree
        importance[node.featid] = get(importance, node.featid, 0)
            + samples(node) * impurity(node)
            - samples(node.left) * impurity(node.left)
            - samples(node.right) * impurity(node.right)
    end
    importance /= samples(node)
    return importance
end

end # module
