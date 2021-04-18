abstract type DataFilter end

function filter_data(df_series, args...)
  df = deepcopy(df_series)
  nneusstart = nneus(df)
  nsersstart = nseries(df)
  for a in args
    df = _filter_data(df,a)
  end
  nneusend = nneus(df)
  nsersend = nseries(df)
  @info """Application of selection criteria
  Before selection : $nsersstart series, $nneusstart, neurons
  After selection  : $nsersend series, $nneusend, neurons
  """
  return sort!(df,serselector)
end

function df_to_keep!(df::DataFrame)
  delete!(df, .! df.to_keep)
  return select!(df,Not(:to_keep))
end

struct NoFilter <: DataFilter
end
_filter_data(dfs,f::NoFilter) = dfs


# no natural images
struct OnlyNat <: DataFilter end
function _filter_data(dfs,n::OnlyNat)
  filter!(:natimg => (i -> !ismissing(i)) , dfs)
  return dfs
end

# no natural images
struct NoNat <: DataFilter end
function _filter_data(dfs,n::NoNat)
  filter!(:natimg=>ismissing, dfs)
  return dfs
end

# reponse score
struct ResponseScore <: DataFilter
  kth::Real
end
function _filter_data(df,p::ResponseScore)
  transform!(groupbyseries(df),
    :resp_score => (scors -> any(scors .>= p.kth) ) => :to_keep)
  return df_to_keep!(df)
end

# minimum mean spk count (absolute) above rmin
struct MinMeanCount <: DataFilter
  rmin::Float64
end
function _filter_data(df,p::MinMeanCount)
  transform!(groupbyseries(df),
      :spk_count => (spks -> minimum(mean.(spks)) >= p.rmin)   => :to_keep)
  return  df_to_keep!(df)
end

struct AverageFFLower <: DataFilter
  ff_min::Real
end
function _filter_data(df,p::AverageFFLower)
  transform!(groupbyseries(df),
      :spk_ff => ( ffs -> geomean(collect(skipmissing(ffs))) <= p.ff_min )  => :to_keep)
  return  df_to_keep!(df)
end

struct NSeriesMin <: DataFilter
  nmin
end
function _filter_data(df,p::NSeriesMin)
  transform!(groupby(df,neuselector),
    :series => (s -> length(Set(s)) >= p.nmin ) => :to_keep)
  return  df_to_keep!(df)
end

struct ByRFSizes <: DataFilter
  idx_rf_good
end
function _filter_data(df,p::ByRFSizes)
  function f_keep(spikecounts,sizes)
    (_,is_rf,_)  = get_idx_rf_and_large(spikecounts,sizes)
    return (findfirst(is_rf) in p.idx_rf_good)
  end
  sort!(df,vcat(serselector,:size))
  transform!(groupbyseries(df),
    [:spk_mean,:size] => f_keep => :to_keep)
  return  df_to_keep!(df)
end

struct NoNatGap <: DataFilter end
function _filter_data(df,p::NoNatGap)
  filter!(:hasgap => (g -> !g) , df)
  return df
end

struct SurroundSuppression <: DataFilter
  suppr::Float64
end
function _filter_data(df,p::SurroundSuppression)
  function f_keep(mus,sizes)
    (_,is_rf,is_lg)  = get_idx_rf_and_large(mus,sizes)
    rf = mus[findfirst(is_rf)]
    lg = mus[findfirst(is_lg)]
    return 2(rf-lg)/(rf+lg) >= p.suppr
  end
  sort!(df,vcat(serselector,:size))
  transform!(groupbyseries(df),
    [:spk_mean,:size] => f_keep => :to_keep)
  return  df_to_keep!(df)
end


# filter for surroind orientation experiments

struct BestOri <: DataFilter
end

function _filter_data(dfs,p::BestOri)
  serselector = vcat(neuselector,:series)
  dffilt = combine(groupby(dfs, neuselector)) do df
    if :contrast in names(df)
      dfrf = @where(df, ismissing.(:oriS),:contrast .== 1)
    else
      dfrf = @where(df, ismissing.(:oriS))
    end
    sers = (df.session[1],df.electrode[1],df.neuron[1])
    if isempty(dfrf)
      @warn "contrast 1 not found, cannot estimate best orientation"
      @warn " session $(sers[1]) , electrode $(sers[2]), neuron $(sers[3])"
      keep = false
    else
      oriCkeep = dfrf.oriC[argmax(dfrf.spk_mean)]
      keep =  df.oriC .== oriCkeep
    end
    return DataFrame(to_keep = keep,series=df.series)
  end
  # dffilt = dffilt[dffilt.keep,:]
  # return semijoin(dfs,dffilt ; on=serselector)
  return df_to_keep!(dffilt)
end

struct HighestContrast <: DataFilter
  c::Float64
end


function _filter_data(dfs,p::HighestContrast)
    serselector = vcat(neuselector,:series)
    dffilt = combine(groupby(dfs,serselector),
        :contrast => ( _c-> (unwrap(_c[1]) > p.c) ) => :to_keep)
    # dffilt = dffilt[dffilt.keep,:]
    # return semijoin(dfs,dffilt ; on=serselector)
    return df_to_keep!(dffilt)
end

# function filter_data(dfs, p::SurroundSuppression)
#   serselector = vcat(neuselector,:series)
#   dffilt = combine(groupby(dfs,serselector)) do df
#     (_, is_rf,is_large) = _rf_and_large_sizes(df.spk_mean,df.size,false)
#     DataFrame(keep=keep)
#   end
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=serselector, kind=:semi)
# end

struct Mean4Views <: DataFilter
  m
  idx_views
end

# struct NotRF <: DataFilter
#   noidx
# end
struct SmallImageVsBaseline <: DataFilter
  baseline_data
  spk_deltat
  thresh
end

#
# function filter_data(dfs,p::Mean4Views)
#   dfs_views = @where(dfs, in.(:view,Ref(p.idx_views)))
#   dffilt = combine(groupby(dfs,serselector)) do df
#     muu = mean(mean.(df.spk_count))
#     keep = muu >= p.m
#     DataFrame(keep=keep)
#   end
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=neuselector, kind=:semi)
# end
#
#
# function filter_data(dfs, p::NotRF)
#   serselector = vcat(neuselector,:series)
#   sizes = sort(unique(dfs.size))
#   nonosizes = getindex(sizes,p.noidx)
#   dffilt = combine(groupby(dfs,serselector)) do df
#     (_, is_rf,is_large) = _rf_and_large_sizes(df.spk_mean,df.size,false)
#     keep = !(df.size[is_rf][1] in nonosizes)
#     DataFrame(keep=keep)
#   end
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=serselector, kind=:semi)
# end
#
# function filter_data(dfs, p::ByRFSizes)
#   serselector = vcat(neuselector,:series)
#   sizes = sort(unique(dfs.size))
#   if any(sizes .== 0 )
#     @warn "Size 0 will not be counted as an index! Index 1 is the smallest non-zero size"
#   end
#   filter!(x->x>0,sizes)
#   yesyessizes = getindex(sizes,p.yesidx)
#   dffilt = combine(groupby(dfs,serselector)) do df
#     (_, is_rf,_) = _rf_and_large_sizes(df.spk_mean,df.size,false)
#     keep = df.size[is_rf][1] in yesyessizes
#     DataFrame(keep=keep)
#   end
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=serselector, kind=:semi)
# end
#
# # only neurons with enough series are selected
# function filter_data(dfs, p::NSeriesMin)
#   dffilt = combine(groupby(dfs,neuselector)) do df
#     nseries = length(Set(df.series))
#     DataFrame(keep = nseries >= p.nmin)
#   end
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=neuselector, kind=:semi)
# end
#
# function filter_data(dfs , p::SmallImageVsBaseline)
#   dfb,dt,th = p.baseline_data, p.spk_deltat, p.thresh
#   df = @where(dfs, .! ismissing.(:natimg), :size .< 1.1)
#   serselector = vcat(neuselector,:series)
#   df = select!(df,vcat(serselector,:spk_mean))
#   df = @transform(df, spk_mean_hz = :spk_mean ./ dt)
#   df = join(df,dfb ; on = neuselector)
#   dffilt = by(df , serselector, [:blank_mean,:blank_var,:spk_mean_hz] =>
#       x->(keep=x.spk_mean_hz .> 1000*(x.blank_mean .+ (th .* sqrt.(x.blank_var))),);
#       sort=true)
#   dffilt = dffilt[dffilt.keep,:]
#   return join(dfs,dffilt ; on=neuselector, kind=:semi)
# end
#
# function filter_data(df, args...)
#   serselector = vcat(neuselector,:series)
#   _df = df
#   (nneus, nsers) = n_neus_series(_df)
#   @info "Starting with $nsers series ($nneus neurons)"
#   for a in args
#     _df = filter_data(_df,a)
#   end
#   (nneus, nsers) = n_neus_series(_df)
#   @info "Ending with $nsers series ($nneus neurons)"
#   return sort!(_df,serselector)
# end
#
# neurons that are in 1 but ARE NOT in 2
# therefore filter 2 should be narrower!
function filter_data_between(df, filter¹::S,filter²::S) where S<:DataFilter
  df¹ = filter_data(df,filter¹)
  if isempty(df¹)
    @warn "The first element is empty!!"
  end
  df² = filter_data(df,filter²)
  if isempty(df²)
    @warn "The second element is empty!"
  end
  ret = join(df¹,df² ; on=vcat(neuselector,:series),kind=:anti)
  if isempty(ret)
    @warn "The selection by exclusion results in an empty filter!"
  end
  return ret
end

#
# # data is by series
# function get_rf_size_byseries(df)
#   serselector = vcat(neuselector,:series)
#   return combine(groupby(df,serselector)) do df
#     (_, is_rf,_) = _rf_and_large_sizes(df.spk_mean,df.size,false)
#     DataFrame(rfsize=df.size[is_rf])
#   end
# end
