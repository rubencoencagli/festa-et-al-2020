module DataAnalysisRCC
export @nm, SpikingData

using HDF5, Serialization, MAT
using DataFrames, DataFramesMeta
using Statistics , StatsBase, LinearAlgebra
using SmoothingSplines # smoothen PSTH for threshold
using HypothesisTests , Bootstrap
using Formatting
using EponymTuples

function  nm(df::DataFrame)
  println( "number of rows : $(nrow(df)) ; number of columns: $(ncol(df))" )
  [print(String(n)," ","| ") for n in names(df)]; println("\n")
  return nothing
end
macro nm(df)
  println("DataFrame :  ", string(df))
  :(nm($(esc(df))))
end

"""parameters = params_vect2struct(model_fulltrained.p_prc.p);
    struct SpikingData
Container for unabridged data that includes spiketimes
"""
mutable struct SpikingData
  spikes::DataFrame
  views::DataFrame
  timestim::Float64
  timebins::Vector{Float64}
  times::Vector{Float64}
end

# general utility functions

spitp(a,b) = pvalue(OneSampleTTest(a,b); tail=:right)
spitp(a) = spitp(a,0.0)
spitdeltaerror(x, down,up) = (x-down, up-x)

spitp_type2(a,b) = pvalue(ApproximateMannWhitneyUTest(a,b))
spitp_type3(a,b) = pvalue(ApproximateSignedRankTest(a,b))

score_straight(a,b) = a-b
score_perc(a,b) = 200(a-b)/(a+b)
p_straight = spitp # alias
p_perc(a,b) = spitp(score_perc.(a,b))

round2(x)=round(x;sigdigits=2)
round3(x)=round(x;sigdigits=3)

function ntup2df(ntup)
  df=DataFrame()
  for (k,v) in pairs(ntup)
    df[!,k] = [v]
  end
  return df
end

function matrix_to_dataframe(matrix, cols_tuple , spikenames=:spikes ;
     verbose=true)
    cols = values(cols_tuple)
    @assert length(cols) == ndims(matrix) "number of parameter values wrong!"
    szmat = size(matrix)
    ntrials=szmat[end]
    @assert begin
        a = [ length(c) == d for (c,d) in zip(cols,szmat[1:end-1]) ]
        all(a)
    end "wrong column sizes!"
    dfret = DataFrame()
    ktot = length(cols_tuple)
    for (k,(key,col)) in enumerate(pairs(cols_tuple))
        if verbose
            @info "building data column $k of $ktot ... "
        end
        _sz = [szmat...]
        _szre = copy(_sz)
        _szre[Not(k)] .= 1
        _c = reshape(col,Tuple(_szre))
        _sz[k]=1
        _mat  = repeat(_c; outer=_sz)
        dfret[!,key] = _mat[:]
    end
    if verbose ; @info "Adding the data !" ; end
    dfret[!,spikenames] = matrix[:]
    return dfret
end

function matrix_bin_vects(mat)
    @assert all((mat .== 0) .| (mat .== 1) .| isnan.(mat) )
    alldims = size(mat)
    retdims = size(mat)[1:end-1]
    ret = Array{Union{BitArray{1},Missing},ndims(mat)-1}(missing,retdims)
    for ijk in CartesianIndices(retdims)
        ijkt = Tuple(ijk)
        vret = mat[ijkt... ,:]
        if any( isnan.(vret))
            @assert all(isnan.(vret))
        else
            ret[ijk] = (vret .== 1.0)
        end
    end
    return ret
end

# vectorized version, data should be 1,0 and NaN only !
function matrix_to_dataframe_binvect(matrix, cols_tuple , spikenames=:spk ;
     verbose=true)
    cols = values(cols_tuple)
    @assert length(cols) == ndims(matrix) - 1 "number of parameter values wrong!"
    if verbose
        @info "Binarizing input matrix"
    end
    matbin = matrix_bin_vects(matrix)
    verbose && println("Done!")
    szmat = size(matbin)
    @assert all( [length(c) == d for (c,d) in zip(cols,szmat)]) "wrong column sizes!"
    dfret = DataFrame()
    ktot = length(cols_tuple)
    for (k,(key,col)) in enumerate(pairs(cols_tuple))
        if verbose
            @info "building data column $k of $ktot ... "
        end
        # from parameter vector, to multidimensional matrix
        _szre = [szmat...]
        _szre[Not(k)] .= 1
        _c = reshape(col,Tuple(_szre))
        # now repeat the matrix in all other dimensions
        _szrep = [szmat...]
        _szrep[k]=1
        _mat  = repeat(_c; outer=_szrep)
        dfret[!,key] = _mat[:]
    end
    if verbose ; @info "Adding the data !" ; end
    dfret[!,spikenames] = matbin[:]
    return dfret
end

function relative_orientation(ori::Union{Missing,R},
      ref_ori::Real, all_oris::AbstractVector) where R<:Real
  ismissing(ori) && return missing
  @assert (ori in all_oris) "orientation not present in the list"
  @assert (ref_ori in all_oris) "reference orientation non present in the list"
  @assert 0 in all_oris "orientation list should include 0"
  _idx = findfirst(ori .== all_oris)
  _idxz = findfirst(all_oris .== 0 )
  _idxsh = findfirst(ref_ori .== all_oris)
  new_oris = circshift(all_oris,_idxsh - _idxz)
  new_oris[_idx]
end

"""
        matrix_for_pgfplots(x,y,mat::AbstractMatrix)

Creates as string with data can can be plotted in matrix form by pgfplots (and gnuplot)

`x` and `y` are the entries on the respective axes, corresponding to columsn and rows of matrix `mat`.
"""
function matrix_for_pgfplots(x,y,mat::AbstractMatrix)
  ny,nx=size(mat)
  @assert length(x) == nx && length(y) == ny
  ret = ""
  xs,ys,mats = (string.(el) for el in (x,y,mat))
  for (j,x) in enumerate(xs)
    for (i,y) in enumerate(ys)
      ret *= x *"\t" * y *"\t" * mats[i,j] * "\n"
    end
     ret *= "\n"
  end
  return ret
end

include("./natureneuroscience2015postprocess.jl")
include("./surround_orientation_anesthetized.jl")
include("./surround_orientation_awake.jl")
include("./natural_areasumm_awake.jl")


end # module

# testing the function above
# matref = zeros(10,33,22,2,4)
# cols = (
#     a=collect(1:size(matref,1)),
#     b=collect(1:size(matref,2)),
#     c=collect(1:size(matref,3)),
#     d=collect(1:size(matref,4)),
#     e=collect(1:size(matref,5)) )
#
# mymat = map( iii -> Tuple(iii), CartesianIndices(matref))
# trythis = D.matrix_to_dataframe(mymat,cols)
# @show trythis[rand(1:nrow(trythis)),:];
