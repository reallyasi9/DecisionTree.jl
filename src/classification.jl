# Utilities

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict([Pair(v => k) for (k, v) in enumerate(labels)])

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::Vector, votes::Vector, weights=1.0)
    label2ind = label_index(labels)
    counts = zeros(Float64, length(label2ind))
    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::Vector` to each row in X
# and returns a Matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::Matrix)
    N = size(X, 1)
    N_cols = length(row_fun(_squeeze(X[1,:],1))) # gets the number of columns
    out = Array{Float64}(N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(_squeeze(X[i,:],1))
    end
    return out
end

################################################################################

function _split(labels::AbstractVector, features::AbstractMatrix, nsubfeatures::Int, weights::Vector, rng::AbstractRNG)
    if weights == [0]
        _split_info_gain(labels, features, nsubfeatures, rng)
    else
        _split_neg_z1_loss(labels, features, weights)
    end
end

function _split_info_gain(labels::AbstractVector, features::AbstractMatrix, nsubfeatures::Int,
                          rng::AbstractRNG)
    nf = size(features, 2)
    N = length(labels)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(rng, nf)
        inds = r[1:nsubfeatures]
    else
        inds = 1:nf
    end

    for i in inds
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]

        hist1 = _hist(labels_i, 1:0)
        hist2 = _hist(labels_i)
        N1 = 0
        N2 = N

        for (d, range) in UniqueRanges(features_i)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = (i, d)
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels_i, range)
            N1 += deltaN
            N2 -= deltaN
        end
    end
    return best
end

function _split_neg_z1_loss(labels::AbstractVector, features::AbstractMatrix, weights::AbstractVector)
    best = NO_BEST
    best_val = -Inf
    for i in 1:size(features,2)
        domain_i = sort(unique(features[:,i]))
        for thresh in domain_i[2:end]
            cur_split = features[:,i] .< thresh
            value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[neg(cur_split)], weights[neg(cur_split)])
            if value > best_val
                best_val = value
                best = (i, thresh)
            end
        end
    end
    return best
end

function build_stump(labels::AbstractVector, features::AbstractMatrix, weights::AbstractVector=[0];
                     rng=Base.GLOBAL_RNG)
    S = _split(labels, features, 0, weights, rng)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    left_split = view(labels, split)
    right_split = view(labels, neg(split))
    return Node(id, thresh,
                Leaf(majority_vote(left_split), left_split),
                Leaf(majority_vote(right_split), right_split),
                length(labels))
end

function build_tree(labels::AbstractVector, features::AbstractMatrix, nsubfeatures=0, maxdepth=-1; rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    if maxdepth < -1
        error("Unexpected value for maxdepth: $(maxdepth) (expected: maxdepth >= 0, or maxdepth = -1 for infinite depth)")
    elseif maxdepth==0
        return Leaf(majority_vote(labels), labels)
    end
    S = _split(labels, features, nsubfeatures, [0], rng)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    labels_left = view(labels, split)
    labels_right = view(labels, neg(split))
    pure_left = all(labels_left .== labels_left[1])
    pure_right = all(labels_right .== labels_right[1])
    if pure_right && pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    Leaf(labels_right[1], labels_right),
                    length(labels))
    elseif pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    build_tree(labels_right, view(features, neg(split), :), nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                               length(labels))
    elseif pure_right
        return Node(id, thresh,
                    build_tree(labels_left, view(features, split, :), nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    Leaf(labels_right[1], labels_right),
                    length(labels))
    else
        return Node(id, thresh,
                    build_tree(labels_left, view(features, split, :), nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    build_tree(labels_right, view(features, neg(split), :), nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                               length(labels))
    end
end

function prune_tree(tree::LeafOrNode, purity_thresh=1.0)
    function _prune_run(tree::LeafOrNode, purity_thresh::Real)
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values; tree.right.values]
            majority = majority_vote(all_labels)
            matches = find(all_labels .== majority)
            purity = length(matches) / length(all_labels)
            if purity >= purity_thresh
                return Leaf(majority, all_labels)
            else
                return tree
            end
        else
            return Node(tree.featid, tree.featval,
                        _prune_run(tree.left, purity_thresh),
                        _prune_run(tree.right, purity_thresh),
                        tree.samples)
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

apply_tree(leaf::Leaf, feature::Vector) = leaf.majority

function apply_tree(tree::Node, features::Vector)
    if tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_tree(tree, _squeeze(features[i,:],1))
    end
    if typeof(predictions[1]) <: Float64
        return float(predictions)
    else
        return predictions
    end
end

"""    apply_tree_proba(::Node, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::Leaf, features::Vector, labels) =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node, features::Vector, labels)
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::Node, features::Matrix, labels) =
    stack_function_results(row->apply_tree_proba(tree, row, labels), features)

"""
    build_forest(labels::Vector, features::Matrix, nsubfeatures::Integer, ntrees::Integer, partialsampling=0.7, maxdepth=-1; rng=Base.GLOBAL_RNG)

Trains a forest of regression tress.  Parameters are:
    * `labels`: A `Vector` of outcomes.
    * `features`: A 2-D `Matrix` of observation features.  The first dimension of `features` must be the same length as `labels`.
    * `nsubfeatures`: How many subfeatures to use when splitting data at each node.  Must be less than or equal to the length of the second dimension of `features`.
    * `ntrees`: The number of estimators to train.
    * `partialsampling`: The fraction of samples to use when bootstrapping samples for training trees.  Must be between 0 (exclusive) and 1 (inclusive).
    * `maxdepth`: The maximum depth of each tree.  If there is a conflict, `maxdepth` overrides `maxlabels`.
    * `rng`: A random number generator or integer seed for initializing a random number generator.
"""
function build_forest(labels::Vector, features::Matrix, nsubfeatures::Integer, ntrees::Integer, partialsampling=0.7, maxdepth=-1; rng=Base.GLOBAL_RNG)
    @assert 0 < partialsampling <= 1
    rng = mk_rng(rng)::AbstractRNG
    Nlabels = length(labels)
    Nsamples = _int(partialsampling * Nlabels)
    # Keep these outside the parallel loop to ensure reproducibility of OOB sample.
    seeds = rand(rng, UInt32, ntrees)
    forest = @parallel (vcat) for i in 1:ntrees
        irng = MersenneTwister(seeds[i])
        ix = rand(rng, 1:Nlabels, Nsamples)
        build_tree(view(labels, ix), view(features, ix, :), nsubfeatures, maxdepth;
                   rng=irng)
    end
    return Ensemble([forest;], seeds, Nlabels)
end

function apply_forest(forest::Ensemble, features::Vector)
    ntrees = length(forest)
    votes = Array{Any}(ntrees)
    for i in 1:ntrees
        votes[i] = apply_tree(forest.trees[i],features)
    end
    if typeof(votes[1]) <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end

function apply_forest(forest::Ensemble, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_forest(forest, _squeeze(features[i,:],1))
    end
    if typeof(predictions[1]) <: Float64
        return float(predictions)
    else
        return predictions
    end
end

"""    apply_forest_proba(forest::Ensemble, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_forest_proba(forest::Ensemble, features::Vector, labels)
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(forest::Ensemble, features::Matrix, labels) =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)

function build_adaboost_stumps(labels::AbstractVector, features::AbstractMatrix, niterations::Integer; rng=Base.GLOBAL_RNG)
    N = length(labels)
    weights = ones(N) / N
    stumps = Node[]
    coeffs = Float64[]
    seeds = rand(rng, UInt32, niterations)
    i = 1
    for i in 1:niterations
        irng = MersenneTwister(seeds[i])
        new_stump = build_stump(labels, features, weights; rng=irng)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        matches = labels .== predictions
        weights[neg(matches)] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (Ensemble(stumps, seeds[1:i], N), coeffs)
end

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Vector)
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps.trees[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = stumps.trees[1].left.majority
    top_count = -Inf
    for (k,v) in counts
        if v > top_count
            top_prediction = k
            top_count = v
        end
    end
    return top_prediction
end

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, _squeeze(features[i,:],1))
    end
    return predictions
end

"""    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                     features::Vector, labels::Vector)
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                    features::Matrix, labels::Vector)
    stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row,
                                                           labels),
                           features)
end
