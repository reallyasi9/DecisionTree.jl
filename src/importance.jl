type MatrixColumnShuffleView{T} <: AbstractMatrix{T}
    data::AbstractMatrix{T}
    index::Int
    perm::Vector{Int}
    function MatrixColumnShuffleView{T}(a::AbstractMatrix{T}, idx::Int; rng=Base.GLOBAL_RNG)
        rng = mk_rng(rng)
        new(a, idx, randperm(rng, size(a, 1)))
    end
end
Base.size(s::MatrixColumnShuffleView) = Base.size(s.data)
function Base.getindex{T}(s::MatrixColumnShuffleView{T}, idx::Vararg{Int,2})
    idxa = [idx...]
    if idxa[2] == s.index
        idxa[1] = s.perm[idxa[1]]
    end
    return Base.getindex(s.data, idxa...)
end
function Base.setindex!{T}(s::MatrixColumnShuffleView{T}, v, idx::Vararg{Int,2})
    idxa = [idx...]
    if idxa[2] == index
        idxa[1] = s.perm[idxa[1]]
    end
    Base.setindex!(s.data, v, idxa...)
end

function oob_variable_importance{T,U}(ensemble::Ensemble, labels::AbstractVector{T}, features::AbstractMatrix{U}; rng=Base.GLOBAL_RNG, iterations=1)

    rng = mk_rng(rng)::AbstractRNG

    baseline_prediction = oob_predict(ensemble, features, T)
    baseline_error = misclassification_rate(baseline_prediction, labels)
    denom = baseline_error
    if baseline_error == 0
        warn("Baseline OOB error rate is zero:  variable importance will be reported in terms of absolute errors rather than relative errors.")
        denom = 1.
    end

    nlabels, nfeatures = size(features)

    feature_errors = zeros(Float64, nfeatures, iterations)
    for itr in 1:iterations
        for ifeature in 1:nfeatures
            noised_features = MatrixColumnShuffleView{U}(features, ifeature; rng=rng)
            prediction = oob_predict(ensemble, noised_features, T)
            feature_errors[ifeature, itr] = misclassification_rate(prediction, labels)
        end
    end
    feature_error = mean(feature_errors, 2)

    return (feature_error - baseline_error) ./ denom
end

function oob_predict(forest::Ensemble, features::AbstractMatrix, T::Type)
    @assert forest.labels == size(features, 1)

    oobs = oob_samples(forest) # dimensions: labels x trees
    predictions = zeros(T, forest.labels)

    for i in 1:forest.labels
        oob_trees = oobs[i, :] # trees that did not use this label
        n_oob_trees = sum(oob_trees)
        if n_oob_trees == 0
            warn("All trees use sample $i (OOB prediction will not be useful):  consider increasing the number of estimators.")
            continue
        end
        predictions[i] = apply_forest(view(forest.trees, oob_trees), features[i, :])
    end
    # println("predictions: $predictions")
    return predictions
end

misclassification_rate(pred::AbstractVector, truth::AbstractVector) = mean(truth .!= pred)
misclassification_rate(pred::AbstractVector{Float64}, truth::AbstractVector{Float64}) = mean((truth - pred).^2)
