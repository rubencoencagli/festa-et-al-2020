


function  nm(df::DataFrame)
  println( "number of rows : $(nrow(df)) ; number of columns: $(ncol(df))" )
  [print(String(n)," ","| ") for n in names(df)]; println("\n")
  return nothing
end

"""
Macro to quickly print the column names of a DataFrame object
"""
macro nm(df)
  println("DataFrame :  ", string(df))
  :(nm($(esc(df))))
end

"""
    struct SpikingData
Container for data that includes spiketimes
"""
mutable struct SpikingData
  spikes::DataFrame
  views::DataFrame
  timestim::Float64
  timebins::Vector{Float64}
  times::Vector{Float64}
end


const standard_sizes = collect(2 .^(-1.56:.7:2.64))
const standard_sizes_rel = standard_sizes ./ standard_sizes[2]
function standardize_size(s::R,reference::AbstractVector{R}=standard_sizes ;
        atol=0.1) where R<:Real
    idx = findfirst(isapprox.(s,reference;atol=atol))
    isnothing(idx) && return s
    return reference[idx]
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


"""
        matrix_bin_vects(mat::AbstractArray)
Takes a multi-dimensional array of 1 , 0 and NaN of dimensions `dims`
Returns an array of dimension `dims-1`, that contains either binary vectors
corresponding to 1 and 0, or `missing` in case of NaNs .
Throws an error in case 1 or 0 are mixed with NaNs.
"""
function matrix_bin_vects(mat::AbstractArray)
    @assert all((mat .== 0) .| (mat .== 1) .| isnan.(mat) )
    @assert ndims(mat) > 1
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



const neuselector = [:session,:electrode,:neuron]
const serselector = vcat(neuselector,:series)

dfneus(df::AbstractDataFrame) = unique(select(df,neuselector))
dfneus(dat::SpikingData) = dfneus(dat.spikes)

get_sizes(df) = sort(unique(df[!,:size]))
get_sizes(dat::SpikingData) = get_sizes(dat.views)

groupbyseries(df::AbstractDataFrame) = groupby(df,serselector;sort=true)

nneus(df) = length(groupby(df,neuselector))
nseries(df) = length(groupbyseries(df))
n_neus_series(df) = (nneus(df), nseries(df))

# reading spike train

function spikes_mean_and_var(spk::AbstractVector{B},idxs::B;
       use_khz::Bool=false) where B<:BitArray{1}
    counts = [count( v .& idxs ) for v in spk]
    if use_khz
        nbins = sum(idxs)
        counts = counts ./ nbins
    end
    return(mean_and_var(counts)..., counts)
end


function addemptycounts(dfdat)
  # take all trials for a certain view and session
  ret = combine(groupby(dfdat,:session)) do dfs
    # pick all units in that session
    dfneus = unique(select(dfs,[:electrode,:neuron]))
    # for each trial in that session....
    combine(groupby(dfs,[:view,:trial])) do df
      df = select(df,Not([:session,:view,:trial]))
      # merge all neurons with that trial
      return rightjoin(df,dfneus; on=names(dfneus))
    end
  end
  # create empty spike count
  emptycount = fill!(similar(dfdat.spk[1]),false)
  # add empty counts in all recorded trials
  ret[!,:spk] = coalesce.(ret[!,:spk],Ref(emptycount))
  return ret
end

# convert exact spike times to a binary vector where each element corresponds to a
# small time interval . Having more than one spike is not allowed.
function binarybinformat(dfdat,bins)
  ret_counts = combine(groupby(dfdat, vcat(neuselector,[:view,:trial]) ; sort=true) ) do df
    spk_t = df.spiketime
    h=fit(Histogram,spk_t,bins).weights
    @assert !any( h .> 1 )
    ret = DataFrame( spk = [h .== 1] )
  end
  return addemptycounts(ret_counts)
end


"""
    get_spontaneus_rates(sd::SpikingData ; window) -> dfspont::DataFrame

Computes the spontaneous rates using ALL trials, and the window indicated.
The inpuits are the data object, and the time window , `window=(t1,t2)` where `t1` is expected to be negative and `t2` positive.

If there are negative times (before signal onset) the interval `(t1,t2)` is used, otherwise the window goes from `0` to `t2` , and to `t_end-t1` to `t_end`.

"""
function get_spontaneus_rates(sd::SpikingData , window::Tuple)
  tn,tp=window
  @assert tn<0 && tp>0 "bad interval selected ($tn,$tp)"
  times = sd.times
  if any(times .< tn )
    idxs = (tn .<= times .<= tp)
  else
    tend = times[end]
    idxs = (0.0 .<= times .<= tp) .|
     ( (tend+tn) .< sd.times )
  end
  function mean_etc(spk)
      _mu,_var,_ = spikes_mean_and_var(spk , idxs ; use_khz=true)
      _ff = _mu == 0 ? missing : _var/_mu
      return DataFrame(blank_mean=_mu,blank_var=_var,blank_ff=_ff)
  end
  return combine(:spk => mean_etc , groupby(sd.spikes,neuselector))
end


function get_responses(spikes_selected::DataFrame, idx_sum::BitArray{1}; byview = true)
  bynames = byview ? [:session,:electrode,:neuron,:view] : [:session,:electrode,:neuron]
  return combine(groupby(spikes_selected,bynames)) do df
    mus, vars,_ = spikes_mean_and_var(df.spk, idx_sum; use_khz = true)
    (spk_mean = mus, spk_var = vars)
  end
end

"""
 function get_responses_window( sd::SpikingData ; window = (50E-5,150E-3) )

Average response per stimulus per neuron!
Counted in the specified window, referred to the time vector of spiking data
"""
function get_responses_window( sd::SpikingData,window)
  idx_sum = window[1].< sd.times .< window[2]
  return get_responses(sd.spikes, idx_sum ; byview = true)
end


function get_blank_and_window(sd,window_blank,window_stimulus)
  a = get_responses_window(sd,window_stimulus)
  b = get_spontaneus_rates(sd,window_blank)
  return innerjoin(a,b ; on=intersect(names.([a,b])...))
end


function test_views_included(spk_mean::Real,blank_mean::Real,blank_var::Real,k::Real)
    return  spk_mean >= blank_mean + k *sqrt(blank_var)
end


"""
    get_idx_rf_and_large(spikecounts,sizes,isnatural;largerlarge=false)
Returns:  (spk_rate_rel, is_rf,is_large)
"""
function get_idx_rf_and_large(spikecounts::AbstractVector{<:Real},
            sizes::AbstractVector{<:Real})
  @assert issorted(sizes) "Please sort by size"
  @assert length(spikecounts) == length(sizes) "inputs should have the same size"
  spk_max = maximum(spikecounts)
  spk_rate_rel = spikecounts ./ spk_max
  idx_rfsize,idx_lgsize = ( falses(length(sizes)) for _ in 1:2)
  idx_rfsize[findfirst(spikecounts .>= (0.95spk_max))] = true
  idx_lgsize[argmin( abs.(sizes .- 2sizes[idx_rfsize]) )] = true
  return spk_rate_rel,idx_rfsize,idx_lgsize
end


"""
    get_views_included(sd::SpikingData ;
        kthresh = 1.0,  secondary_features=[:phase,:ori],
        window_stim = (50E-5,150E-3) , window_blank = (25E-3,75E-3) )

Views are included only when their best response is above baseline activity as
specified by `kthresh` . Baseline mean and std is computed using `window_blank`
Response is computed using `window_stim`
"""
function get_views_included(sd::SpikingData ;
      kthresh = 1.0,  secondary_features=[:phase,:ori],
      window_stim = (50E-5,150E-3) , window_blank = (25E-3,75E-3) )
  resps = get_blank_and_window(sd,window_blank,window_stim)
  # size tuning
  if !(:natimg in names(sd.views)) # if not defined, define it as missing
      sd.views[!,:natimg] .= missing
  end
  viewsgrat = filter(:natimg=>ismissing, sd.views)
  grats_views = get_views_included_sizetuning(resps,viewsgrat ;
        kthresh=kthresh, secondary_features=secondary_features)
  # natural img
  viewsnats = filter(:natimg=> i-> !ismissing(i), sd.views)
  nats_views = get_views_included_natural(resps,viewsnats ; kthresh=kthresh)

  # cols = intersect(names.([grats_views,nats_views])...) # meh
  # ret = vcat(select(grats_views,cols),select(nats_views,cols))
  # return  semijoin(resps,ret ; on=intersect(names.([ret,resps])...))
  return vcat(nats_views,grats_views)
end


function get_views_included_natural(resps::DataFrame,views::DataFrame; kthresh=1.0)
  isempty(views) && return DataFrame([])
  nneuspre =nneus(resps)
  @assert nneuspre > 0
  nnats = length(groupby(views,:natimg))
  natsresps=innerjoin(resps,views;on=:view)
  # select only images that pass the test
  imgselector = vcat(neuselector,:natimg)
  resps_filt = filter(:size => s->s<1.3, natsresps)
  transform!(resps_filt,
    AsTable([:spk_mean,:blank_mean,:blank_var]) =>
    ByRow(t-> test_views_included(t.spk_mean,t.blank_mean,t.blank_var,kthresh))
    => :to_keep)
  delete!(resps_filt, .!resps_filt.to_keep)
  nats_select = semijoin(natsresps,resps_filt ; on=imgselector )
  nnatspost = length(groupby(nats_select,imgselector))
  npost=nneus(nats_select)
  @info """ Selection of views for natural images
  Starting with:  $nneuspre neurons, $nnats images each.
  After selection: $npost neurons, $nnatspost combinations of neuron/image
  """
  return nats_select
end


function test_views_included_with_rf(spk_mean,blank_mean,blank_var,k,sizes)
    (_, is_rf, _) = get_idx_rf_and_large(spk_mean, sizes)
    return test_views_included(spk_mean[is_rf][1],
                blank_mean[is_rf][1],blank_var[is_rf][1],k)
end

function get_views_included_sizetuning(resps::DataFrame, views::DataFrame; kthresh = 1.0,
     secondary_features = [:phase,:ori])
    isempty(views) && return DataFrame()
    nneuspre = nneus(resps)
    @assert nneuspre > 0
    serselector = vcat(neuselector, secondary_features)
    gratsresps=innerjoin(resps,views;on=:view)
    sort!(gratsresps,vcat(serselector,:size) )
    # select rf size only
    dffilt=combine(groupby(gratsresps, serselector),
       AsTable([:spk_mean,:blank_mean,:blank_var,:size]) =>
        (t -> test_views_included_with_rf(t.spk_mean,
            t.blank_mean,t.blank_var,kthresh,t.size) )   => :to_keep)
    delete!(dffilt, .! dffilt.to_keep)
    grats_select = semijoin(gratsresps, dffilt; on = serselector)
    nneuspost = nneus(grats_select)
    @info """ Selection of views for gratings
    Starting with:  $nneuspre neurons
    After selection: $nneuspost neurons
    """
    return grats_select
end



"""
        define_series(data_spikecounts::DataFrame ;
                secondary_features = [:natimg,:phase,:ori])
Secondary features define series, a series is kept if all responses are > 0
"""
function define_series(df_spk::DataFrame ; secondary_features = [:natimg,:phase,:ori],
    kthresh=1.0 )
  nneus_initial =nneus(df_spk)
  serselector = vcat(neuselector,secondary_features)
  df_goodseries = combine(groupby(df_spk,serselector) ,
   :spk_mean => (mus -> all(mus .> 0 ) ) => :to_keep)
  delete!(df_goodseries, .! df_goodseries.to_keep)
  select!(df_goodseries,Not(:to_keep))
  # add series idx
  df_goodseries[!,:series] .= 0
  for df in groupby(df_goodseries,neuselector)
      s_idx=0
      for ddf in groupby(df,serselector)
          s_idx+=1
          ddf.series .= s_idx
  end end
  ret =  innerjoin(df_spk, df_goodseries ; on = serselector)
  nneus_final = nneus(ret)
  @info """ Selection of series
  After removing series that include null mean responses
  we have $nneus_final neurons (out of the previous $nneus_initial)
  """
  return sort!(ret,vcat(neuselector,:series))
end


function relativize_sizes(df;atol=0.1)
  function f_sizerel(rates,sizes)
    (_,is_rf,_)  = get_idx_rf_and_large(rates,sizes) # this also tests the vectors
    size_rf = sizes[findfirst(is_rf)]
    return map(
        s-> standardize_size(s/size_rf,standard_sizes_rel;atol=atol), sizes)
  end
  dfret = transform(groupbyseries(df) ,
     [:spk_mean, :size] => f_sizerel => :size_rel )
  select!(dfret,Not(:size))
  rename!(dfret , :size_rel=>:size)
  return dfret
end

function average_over_series(dfdata, primary_feature)
  df_byfeature = groupby(dfdata,vcat(neuselector,primary_feature))
  # average by series done here
  dfret =  combine(df_byfeature,
    :spk_mean => mean  => :spk_mean ,
    :spk_mean => var  => :spk_var , # variance across means!
    :spk_ff => geomean =>:spk_ff )
  # now add relative rates
  # add the max rate for each neuron across primary feat
  transform!(groupby(dfret,neuselector),
    :spk_mean => maximum => :spk_mean_max)
  transform!(dfret, [:spk_mean,:spk_mean_max]=>
    ((a,b)-> a ./ b) =>:spk_mean_rel)
  select!(dfret,Not(:spk_mean_max))
  return dfret
end

function average_over_series_rel_old(data_spikecounts_series ; sizesnat = 2,sizesgrat=7)
  dat = @transform(data_spikecounts_series, isnatural = .!ismissing.(:natimg))
  _to_join = vcat(neuselector,[:size,:isnatural])
  sizes = get_sizes(dat)
  # average by series done here
  ret = combine(groupby(dat,_to_join)) do df
    _mu,_ss = mean_and_var(df.spk_mean)
    _ff = geomean(df.spk_ff)
    DataFrame(spk_mean=_mu, spk_var = _ss , spk_ff = _ff)
  end
  # adding relative rates
  retrels = combine(groupby(ret,vcat(neuselector,:isnatural))) do df
    # assert it has the right number of elements, corresponding to sizes
    isnat = df.isnatural[1]
    @assert ( isnat && (nrow(df) == sizesnat) )  ||
      ( (!isnat) && (nrow(df) == sizesgrat) ) "natural ? $isnat, unexpected number of rows: $(nrow(df))"
    spk_max = maximum(df.spk_mean)
    spk_rate_rel = df.spk_mean ./ spk_max
    _idx_good =  df.spk_mean .>= (0.95spk_max)
    _rf_size = minimum(df.size[_idx_good])
    is_rf = (df.size .== _rf_size) .& (!isnat)
    _lgsize = sizes[argmin(abs.(sizes .- 2*_rf_size) )]
    is_large = (df.size .== _lgsize) .& (!isnat)
    @assert ( (count(is_large) == 1) && (!isnat) ) || isnat
    DataFrame(size =df.size, spk_rate_rel = spk_rate_rel,is_rf=is_rf,is_large=is_large)
  end
  retboth =innerjoin(ret,retrels ; on =_to_join)
  return retboth
end

function check_signi(x1,x2,y1,y2)
  return (x2 < y1) || (y2 < x1)
end
