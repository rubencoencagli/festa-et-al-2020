
## High level functions to read responses to natural images

# create dataframe with stimuli (a.k.a. views) information
function make_dfviews_pvc8()
  # natural images parameters
  idx_matnat = collect(1:540)
  nimg = div(540,2)
  sizes_deg = standard_sizes # defined in data_analysis_base.jl
  size_nat50, size_nat150  = sizes_deg[[3,6]]
  sizes_nat = repeat([size_nat50,size_nat150] ; outer=nimg )
  natimgs = repeat(1:nimg ; inner=2)
  views_nats = DataFrame(view = idx_matnat, idxmat = idx_matnat,
      size=sizes_nat, natimg=natimgs,
      phase=missing,ori=missing,category=missing,)
  # gratings parameters
  idxs = 540 .+ collect(1:224)
  idx_matgrat = collect( (1:224)) .+ (64 + 128 + 2*9*30)
  oris = [0,45,90,135]
  phase = (1:4)
  category = (1:2)
  allpars =collect(Iterators.product(phase,category,sizes_deg,oris))[:]
  pars_get(k::Integer) =getindex.(allpars,k)
  views_grats = DataFrame(view=idxs, idxmat = idx_matgrat,
      phase = pars_get(1) , category=pars_get(2),size=pars_get(3), ori = pars_get(4),
      natimg=missing)
  return vcat(views_nats, views_grats)
end


# get list of files to read
dir_pvc8() = joinpath(read_dirfile()["dir_exp"],"crcns_pvc8")

# reads a single file, as array
function read_spk_pvc8(file,idx_stims)
  dd=matread(file)
  idxc = dd["INDCENT"][:] .== 1
  resp = dd["resp_train"][idxc,:,:,:]
  resp_blk = dd["resp_train_blk"][idxc,:,:,:]
  resp_both = cat(resp,resp_blk;dims=4)
  return resp_both[:,idx_stims,:,:]
end



# reads all files, saves as SpikingData object
function SpikingData_pvc8()
  filenames = filter(f-> occursin(r"\.mat\b",f),  readdir(dir_pvc8()))
  filenames = joinpath.(dir_pvc8(),filenames)
  views = make_dfviews_pvc8()
  idx_read_keep = views.idxmat
  # now read the files
  # make empty dataframe
  datadfs = map(filenames) do f
    @info "now reading file $f"
    mat  = read_spk_pvc8(f,idx_read_keep)
    nneurons,nviews,ntrials,_ = size(mat)
    datacols = (neuron = UInt8.(1:nneurons), view=UInt16.(1:nviews),
      trial=UInt8.(1:ntrials) )
    dfout = matrix_to_dataframe_binvect(mat,datacols;verbose=false)
    dfout[!,:session] .= basename(f)
    return dfout
  end
  datadf = vcat(datadfs...)
  categorical!(datadf,:session;compress=true)
  datadf[!,:electrode] .= UInt8(1)
  # category is missing or 1 , not 2. Remove unnecessary views
  filter!(:category => (c -> ismissing(c) || c != 2 ),  views)
  # keep only good views in spike train
  filter!(:view => (v-> v in views.view), datadf)
  @assert !any(ismissing.(datadf.spk))
  dropmissing!(datadf,:spk;disallowmissing=true)
  # now the time bins
  _, T = trim_spiketrains!(datadf.spk)
  time_stim = 0.0
  time_bins = collect(0:T) .* 1E-3
  times = midpoints(time_bins)
  # define the object
  return SpikingData(datadf, views, time_stim,time_bins,times)
end
##


"""
Ensures that all spiketrains have the same length, cutting extra bins.
"""
function trim_spiketrains!(spks::AbstractVector{B}) where B<:BitArray{1}
  ls = length.(spks)
  ls_min = minimum(ls)
  for i in findall(ls .> ls_min)
    spks[i] = spks[i][1:ls_min]
  end
  return spks,ls_min
end


function average_ff_over_series_pvc8_natimg(data_spikecounts_series ; ci=0.86)
  dat = filter(:natimg => (i-> !ismissing(i)) , data_spikecounts_series)
  @assert !isempty(dat) "Is this data from natural image stimuli?"
  # error only FF , average by series, for each size
  return  combine(groupby(dat,neuselector;sort=true)) do df
    dfsm = @where(df,:size .< 1.0) # small stim
    dflg = @where(df,:size .> 1.0) # large stim
    ffslg = [dflg.spk_ff...] # FFs for all views
    ffssm = [dfsm.spk_ff...]
    @assert length(ffslg) == length(ffssm) "All images should be in small and lage prentations"
    # gemetric means, check that the difference is significant
    # functions here are defined in data_analysis_base.jl
    gglg = geomean_boot(ffslg ; conf=ci,prefix="ff_lg")
    ggsm = geomean_boot(ffssm ; conf=ci,prefix="ff_sm")
    np2 = (issigni = check_signi(gglg.ff_lg_cidown, gglg.ff_lg_ciup,
              ggsm.ff_sm_cidown, ggsm.ff_sm_ciup) ,)
    # date too
    np3 = (spk_lg = mean(dflg.spk_mean) , spk_sm=mean(dfsm.spk_mean) )
    merge(ggsm,gglg,np2,np3)
  end
end




function _psth_count(vs::Vector{BitArray{1}})
  ret = zeros(Int64,length(vs[1]))
  for v in vs
    ret[v] .+= 1
  end
  return ret
end
function get_psth(spikes_select::Vector{BitArray{1}}, times::Vector{Float64} ;
        smooth_scale::Float64=2E-6)
  counts_sum = _psth_count(spikes_select)
  _nn = length(spikes_select)
  counts = @.  Float64(counts_sum) / _nn
  spl = fit(SmoothingSpline, times, counts , smooth_scale)
  counts_smooth = SmoothingSplines.predict(spl)
  (time=times , psth = counts , psth_smooth = counts_smooth)
end
function get_psth(sd::SpikingData,dffilter; smooth_scale::Float64=1E-6)
  _spikes = join(sd.spikes,dffilter;on=names(dffilter), kind=:semi).spk
  times = sd.times
  get_psth(_spikes,times; smooth_scale=smooth_scale)
end
function show_psth(sd,filt; smooth_scale=1E-6)
  psth = get_psth(sd,filt;smooth_scale=smooth_scale)
  return (psth.time, psth.psth,psth.psth_smooth)
  # Plots.bar(psth.time , psth.psth)
  # Plots.plot!(psth.time , psth.psth_smooth;
  #     leg=false, linewidth=5,xlabel="time(s)")
end

function dfqueryneuron(df,dfquery)
  dfq = dfneus(dfquery)
  @assert nrow(dfq) == 1  "you can query only one neuron at a time"
  dfout = join(df,dfq;on=names(dfq),kind=:semi)
  @assert !isempty(dfout)  "neuron not found (rows =  $(nrow(df)))"
  return dfout
end


function query_spontaneous(dfspont,dfquery)
  dfout = dfqueryneuron(dfspont, dfquery)
  @assert nrow(dfout) == 1 "something wrong here"
  return (dfout.blank_mean, dfout.blank_var)
end

# phase + ori => stimulus series , finds the rf size
function add_rfsize_byseries(spikes_and_stuff::DataFrame,secondary_features)
  bynames = vcat(neuselector,secondary_features)
  @assert  all(in.(bynames,Ref(names(spikes_and_stuff)))) "Secondary features not found!"
  return by(spikes_and_stuff,bynames) do df
    @assert nrow(df) == 7 " not enough sizes! Expected 7 , found $(nrow(df))"
    _idx_good =  df.spk_mean .>= (0.95maximum(df.spk_mean))
    rf_size = minimum(df.size[_idx_good])
    dfout = select(df,Not(bynames))
    dfout[!,:rf_size] .= rf_size
    dfout[!,:is_rf] = (rf_size .== df.size)
    return dfout
  end
end


function _test_latency(neuron_select::DataFrame, sd::SpikingData,included_data)
  _spikes = join(sd.spikes, neuron_select ; on=names(neuron_select), kind=:semi)
  spk_good = join(_spikes, included_data;on=vcat(neuselector,:view) , kind = :semi)
  incl_select = join(included_data,neuron_select; on=neuselector, kind=:semi)
  @show nrow(incl_select)
  @assert all(incl_select.blank_mean .== incl_select.blank_mean[1] )
  @assert all(incl_select.blank_var .== incl_select.blank_var[1] )
  ts = sd.times
  psth_all = get_psth(spk_good.spk,ts)
  if :size in names(sd.views)
    views_good = @where(sd.views, 0.4 .< :size .< 1.2 )
  else
    views_good = sd.views
  end
  dfgoodv = join(spk_good,views_good; on=:view , kind=:semi)
  psth_good = get_psth(dfgoodv.spk,ts)
  nt = length(ts)
  _blankmuline = fill(incl_select.blank_mean[1],nt )
  _spkmuline = fill(mean(incl_select.spk_mean),nt )
  return (ts,psth_all.psth_smooth,  psth_good.psth_smooth, _blankmuline, _spkmuline)
  # plot(ts, [ all_smooth,  good_smooth, _blankmuline, _spkmuline];
  # label =["psth all" "psth good" "blank mean" "spk mean" "thresh"], linewidth=3 )
end


function _cutoff(x,xmin,xmax)
  max(xmin,min(x,xmax))
end

# above the max between threshold and 1/3 peak
# if never above threshold, missing
function _idx_latency(smoothcurve,th)
  all(smoothcurve .<= th ) && return missing
  maxc=maximum(smoothcurve)
  _th = max( maxc*0.333, th )
 return findfirst(smoothcurve .> th)
end

function compute_latency(sd::SpikingData, included_data ;
        min_latency=25E-3 , max_latency=100E-3, k_latency=2.0)
  spk_good = join(sd.spikes, included_data;
      on = vcat(neuselector,:view) , kind = :semi )
  if :size in names(sd.views)
    views_good = @where(sd.views, 0.4 .< :size .< 1.2 ).view
  else
    views_good = sd.views.view
  end
  spk_good = @where(spk_good, in.(:view, Ref(views_good)) )
  ret = combine(groupby(spk_good,neuselector)) do df
    spks = [s for s in df.spk]
    psth_stuff = get_psth(spks,sd.times)
    rr = @where(included_data, :session .== df.session[1] ,
      :electrode .== df.electrode[1] , :neuron .== df.neuron[1])
    th = rr.blank_mean[1] + k_latency*sqrt(rr.blank_var[1])
    # correct the psth, so that times before onset aren't considered
    _psth_smooth = let _sm = copy(psth_stuff.psth_smooth)
      _sm[ psth_stuff.time .<= 0 ] .= -Inf
      _sm
    end
    idx = _idx_latency(_psth_smooth,th)
    lat = ismissing(idx) ? missing : _cutoff(psth_stuff.time[idx],min_latency,max_latency)
    (latency = lat, blank_mean= rr.blank_mean[1], blank_var = rr.blank_var[1])
  end
  npre = nrow(ret)
  dropmissing!(ret, disallowmissing=true )
  npost = nrow(ret)
  @info """ Selection by latency:
  Latencies computed with threshold of $(k_latency) stds above spontaneous rate
  Neurons kept: $npost (out of the prevous $npre)
  """
  return ret
end

function count_spikes(datgood::DataFrame,times::Vector{Float64},latencydata::DataFrame;
  time_count::Float64 = 0.106)
 ret =  combine(groupby(datgood, vcat(neuselector,:view))) do df
   rlat = @where(latencydata, :session .== df.session[1] ,
     :electrode .== df.electrode[1] , :neuron .== df.neuron[1])
   @assert nrow(rlat) == 1 ; rlat=rlat[1,:] ; _latency = rlat.latency
   idx_sum =  _latency .< times .< (_latency+time_count)
   spk = [s for s in df.spk]
   _mu,_var,_counts = spikes_mean_and_var(spk,idx_sum ; use_khz=false )
   _ff = _mu == 0 ? missing : _var/_mu
   _resp_score = (_mu/count(idx_sum) - rlat.blank_mean)/ sqrt(rlat.blank_var)
   DataFrame( spk=[spk], spk_count=[_counts],
        spk_mean = _mu, spk_var = _var, spk_ff = _ff ,
        resp_score=_resp_score)
 end
end


"""
    _rf_and_large_sizes(spikecounts,sizes,isnatural)
Returns:  (spk_rate_rel, is_rf,is_large)
"""
function _rf_and_large_sizes(spikecounts,sizes,isnatural;largerlarge=false)
  spk_max = maximum(spikecounts)
  spk_rate_rel = spikecounts/spk_max
  if ~isnatural
    _idx_good =  spikecounts .>= (0.95spk_max)
    _rf_size = minimum(sizes[_idx_good])
    is_rf = (sizes .== _rf_size)
    _diff = @. sizes - 2*_rf_size
    if largerlarge
      _diff[_diff .<= 0] .= Inf
    else
      @. _diff = abs(_diff)
    end
    _lgsize =  sizes[argmin(_diff)]
    is_large = sizes .== _lgsize
    return (spk_rate_rel, is_rf,is_large)
  else
    fals() =falses(length(sizes))
    return (spk_rate_rel, fals() , fals())
  end
end

function _find_sizes(spk_means,sizes, ratio_small, ratio_large ; by_excess=false)
  @assert issorted(sizes) "Please sort by size first!"
  rounds2(x) = round(x;sigdigits=2)
  nsizes = length(sizes)
  sizesx = rounds2.(sizes)
  spk_max = maximum(spk_means)
  rf_size = minimum(sizesx[spk_means .>= (0.95spk_max)])
  rf_size_idx = findfirst(rf_size .== sizesx)
  lg_size_exact = rf_size*ratio_large |> rounds2
  sm_size_exact = rf_size*ratio_small |> rounds2
  lg_size_idx = if !by_excess
      argmin(abs.(sizesx .- lg_size_exact))
    else
      _idx = findfirst(sizesx .>= lg_size_exact)
      something(_idx,nsizes)
  end
  sm_size_idx = if !by_excess
      argmin(abs.(sizesx .- sm_size_exact))
    else
      _idx = findfirst(sm_size_exact .> sizesx)
      max(1,something(_idx,1)-1)
  end
  ret = @eponymtuple(rf_size_idx,lg_size_idx,sm_size_idx)
  for (k,v) in pairs(ret)
    @assert !isnothing(v) "Oh no! $k is nothing!"
  end
  return ret
end


function rf_gratings_suppr_score(dfdata_rel)
  return by(dfdata_rel,neuselector) do df
    _spk_rf = df.spk_mean[df.is_rf][1]
    _spk_lg = df.spk_mean[df.is_large][1]
    _ff_rf = df.spk_ff[df.is_rf][1]
    _ff_lg = df.spk_ff[df.is_large][1]
    @assert all(df.size.>0) # just in case
    is_small  = argmin(df.size)
    _ff_sm = df.spk_ff[is_small]
    scor(a,b) = 2(a-b)/(a+b)
    DataFrame(
      spk_suppr_score=scor(_spk_rf,_spk_lg) ,
      ff_rf_vs_small = scor(_ff_sm,_ff_rf) ,
      ff_large_vs_rf = scor(_ff_rf,_ff_lg) )
  end
end
###########
# median  of population mean  and ff by size ,
# using bootstrap


# function ffdiff_boot(spkA::AbstractVector{T},spkB::AbstractVector{T} ;
#         nrun = 1_000, conf = 0.95, prefix="ffdiff") where T<:Real
#     is_a = vcat( trues(length(spkA)) , falses(length(spkB)) )
#     spks = collect(zip(is_a,vcat(spkA,spkB)))
#     function ffdiff(spks)
#       is_a = getindex.(spks,1)
#       _spks = getindex.(spks,2)
#       spkA = _spks[is_a]
#       spkB = _spks[.!is_a]
#       FFa,FFb = fanofactor.((spkA,spkB))
#       return 2(FFa-FFb)/(FFa+FFb+eps(FFa))
#     end
#     _bs = bootstrap(ffdiff, spks, BasicSampling(nrun))
#     theprefix, theprefix_cidown, theprefix_ciup = confint(_bs, BCaConfInt(conf))[1]
#     theprefix_ddown,theprefix_dup = spitdeltaerror.(theprefix, theprefix_cidown, theprefix_ciup)
#     retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
#     return (; zip(retnames, [theprefix,theprefix_cidown,theprefix_ciup,
#         theprefix_ddown,theprefix_dup])...)
# end
#
# function ff_symmetrize_boot(spkA::AbstractVector{T},spkB::AbstractVector{T} ;
#         nrun = 1_000, conf = 0.68, prefix="ff_symm") where T<:Real
#   is_a = vcat( trues(length(spkA)) , falses(length(spkB)) )
#   spks = collect(zip(is_a,vcat(spkA,spkB)))
#   function ff_symm(spks)
#     is_a = getindex.(spks,1)
#     _spks = getindex.(spks,2)
#     spkA = _spks[is_a]
#     spkB = _spks[.!is_a]
#     FFa,FFb = fanofactor.((spkA,spkB))
#     return sqrt(FFa*FFb)
#   end
#   _bs = bootstrap(ff_symm, spks, BasicSampling(nrun))
#   theprefix, theprefix_cidown, theprefix_ciup = confint(_bs, BCaConfInt(conf))[1]
#   theprefix_ddown,theprefix_dup = spitdeltaerror.(theprefix, theprefix_cidown, theprefix_ciup)
#   retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
#   return (; zip(retnames, [theprefix,theprefix_cidown,theprefix_ciup,
#       theprefix_ddown,theprefix_dup])...)
#  end
#
#  function mean_symmetrize_boot(spkA::AbstractVector{T},spkB::AbstractVector{T} ;
#          nrun = 1_000, conf = 0.95, prefix="mean_symm") where T<:Real
#    is_a = vcat( trues(length(spkA)) , falses(length(spkB)) )
#    spks = collect(zip(is_a,vcat(spkA,spkB)))
#    function mean_symm(spks)
#      is_a = getindex.(spks,1)
#      _spks = getindex.(spks,2)
#      spkA = _spks[is_a]
#      spkB = _spks[.!is_a]
#      mua,mub = mean.((spkA,spkB))
#      return 0.5(mua+mub)
#    end
#    _bs = bootstrap(mean_symm, spks, BasicSampling(nrun))
#    theprefix, theprefix_cidown, theprefix_ciup = confint(_bs, BCaConfInt(conf))[1]
#    theprefix_ddown,theprefix_dup = spitdeltaerror.(theprefix, theprefix_cidown, theprefix_ciup)
#    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
#    return (; zip(retnames, [theprefix,theprefix_cidown,theprefix_ciup,
#        theprefix_ddown,theprefix_dup])...)
#   end

function population_bysize(dfpop::DataFrame)
  ret = by(dfpop, :size;sort=true) do df
    _spk_means = [s for s in df.spk_rate_rel]
    _spk_ffs = [s for s in df.spk_ff]
    spk_mean, spk_cidown, spk_ciup, spk_ddown, spk_dup = mean_boot(_spk_means)
    ff_geomean, ff_cidown, ff_ciup, ff_ddown,ff_dup = geomean_boot(_spk_ffs)
    @eponymtuple( spk_mean, spk_cidown, spk_ciup , spk_ddown, spk_dup,
      ff_geomean, ff_cidown, ff_ciup , ff_ddown, ff_dup)
  end
  return filter!(r->r.size>0, ret)
end

function population_blank_rel(dfpop,dfblank)
  dfpop = by(dfpop, [:session,:electrode,:neuron]) do df
    rate_norm = df.spk_mean[df.is_rf]
    dfb = join(dfblank, select(df,[:session,:electrode,:neuron])  ; kind=:semi)
    @assert nrow(dfb) == 1
    blank_spk = dfb.spk_blank[1]
    blank_spk_rel = blank_spk ./ rate_norm
    blank_ff = dfb.ff_blank
    @eponymtuple(blank_spk_rel,blank_spk,blank_ff)
  end
  blank_rel_all = mean(dfpop.blank_spk_rel)
  blank_ff_all = geomean(dfpop.blank_ff)
  return (dfpop, blank_rel_all, blank_ff_all)
end

function nanpadding(vecs)
  nrows = maximum(length.(vecs))
  ncols = length(vecs)
  ret = fill(NaN,nrows,ncols)
  for (j,vec) in enumerate(vecs)
    for (i,el) in enumerate(vec)
      ret[i,j] = el
  end end
  return ret
end


function interval_diff_signi(ff_a,ff_a_ciup,ff_a_cidown, ff_b,ff_b_ciup,ff_b_cdown)
    d = 2(ff_a - ff_b)/(ff_a+ff_b)
    issignificant = let b0 = d>=0,
      b1 = ff_a_cidown >= ff_b_ciup ,
      b2 = ff_a_ciup <= ff_b_cdown
      ( b0 && b1 ) || ( (!b0) && b2 )
    end
    d,issignificant
end

function ff_diff_signi(spka::V,spkb::V;nrun=15_000) where V<:AbstractVector{<:Real}
  pa =  ff_boot(spka; conf=0.68,prefix="ff",nrun=nrun)
  pb =  ff_boot(spkb; conf=0.68,prefix="ff",nrun=nrun)
  return interval_diff_signi( pa.ff , pa.ff_ciup , pa.ff_cidown ,
                        pb.ff , pb.ff_ciup , pb.ff_cidown)
end

function rate_diff_signi(spka::V,spkb::V;nrun=15_000) where V<:AbstractVector{<:Real}
  pa =  mean_boot(spka; conf=0.68,prefix="rat",nrun=nrun)
  pb =  mean_boot(spkb; conf=0.68,prefix="rat",nrun=nrun)
  return interval_diff_signi( pa.rat , pa.rat_ciup , pa.rat_cidown ,
                        pb.rat , pb.rat_ciup , pb.rat_cidown)
end


function ff_differences_byseries(dfdata; idx_small = 1, largerlarge=false)
  # no natural images allowed!
  dfd = @where(dfdata, ismissing.(:natimg))
  sizes=get_sizes(dfd)
  isnat = false
  # pick the small size
  small_size = sizes[idx_small]
  serselector = vcat(neuselector,:series)
  return by(dfd,serselector) do df
    is_small = df.size .== small_size
    (spk_rate_rel, is_rf,is_large) = _rf_and_large_sizes(df.spk_mean,df.size,isnat;
        largerlarge=largerlarge)
    d_rf, issigni_rf = ff_diff_signi(df.spk_count[is_rf][1],df.spk_count[is_large][1])
    d_small, issigni_small = ff_diff_signi(df.spk_count[is_small][1],df.spk_count[is_rf][1])
    @eponymtuple(d_rf, issigni_rf, d_small, issigni_small)
  end
end

function ff_differences_byneuron(dfdata; idx_small = 1, largerlarge=false)
  sizes=get_sizes(dfdata)
  isnat = false
  # pick the small size
  small_size = sizes[idx_small]
  _scor(a,b) = 2(a-b)/(a+b)
  return by(dfdata,neuselector) do df
    is_small = df.size .== small_size
    (spk_rate_rel, is_rf,is_large) = _rf_and_large_sizes(df.spk_mean,df.size,isnat ;
      largerlarge=largerlarge)
    # @show count.([is_rf,is_small,is_large])
    @assert all( count.([is_rf,is_small,is_large]).== 1) "Error in indexing!"
    ff_rf = df.spk_ff[is_rf][1]
    ff_large =df.spk_ff[is_large][1]
    ff_small =  df.spk_ff[is_small][1]
    d_rf = _scor(ff_rf,ff_large)
    d_small = _scor(ff_small,ff_rf)
    @eponymtuple(d_rf, d_small)
  end
end

"""
        ff_differences_boot(dfdata ;idx_small = 1)

Difference by series, but confidence interval associated to the difference itself,
using the function `ffdiff_boot`.

Should have only one kind of data, either gratings or natural
"""
function ff_differences_boot(dfdata ;
        idx_small = 1 , nrun=1_000, conf=0.68, largerlarge = false)
  dfd = dfdata
  sizes=get_sizes(dfd)
  isnat = false
  # pick the small size
  small_size = sizes[idx_small]
  serselector = vcat(neuselector,:series)
  return by(dfd,serselector) do df
    is_small = df.size .== small_size

    ( _ , is_rf,is_large) = _rf_and_large_sizes(df.spk_mean,df.size,isnat;
        largerlarge=largerlarge)

   diffsrf = ffdiff_boot(df.spk_count[is_rf][1],df.spk_count[is_large][1];
    prefix="diff_rf", nrun=nrun, conf=conf)

   diffssmall = ffdiff_boot(df.spk_count[is_small][1], df.spk_count[is_rf][1];
     prefix="diff_small", nrun=nrun, conf=conf)
   merge(diffsrf,diffssmall)
  end
end

function ff_keyvalues_byseries(dfdata, sm_ratio, lg_ratio;
        by_excess = false)
  @assert sm_ratio < 1 < lg_ratio "Something wrong in the ratios"
  _selector = vcat(neuselector,:series)
  dfd = deepcopy(dfdata)
  nsizes = length(unique(dfd.size))
  sort!(dfd,vcat(_selector,:size))
  return by(dfd,_selector) do df
    @assert nrow(df) == nsizes
    sizestuff = _find_sizes(df.spk_mean, df.size,sm_ratio,lg_ratio ;
      by_excess=by_excess)
    ff_rf = df.spk_ff[sizestuff.rf_size_idx]
    ff_lg = df.spk_ff[sizestuff.lg_size_idx]
    ff_sm = df.spk_ff[sizestuff.sm_size_idx]
    @eponymtuple(ff_rf,ff_lg,ff_sm)
  end
end

function ff_keyvalues_byneuron(dfdata, sm_ratio, lg_ratio;
        by_excess = false)
  @assert sm_ratio < 1 < lg_ratio "Something wrong in the ratios"
  _selector = neuselector
  dfd = deepcopy(dfdata)
  nsizes = length(unique(dfd.size))
  sort!(dfd,vcat(_selector,:size))
  return combine(groupby(dfd,_selector)) do df
    @assert nrow(df) == nsizes "expected $nsizes, found $(nrow(df))"
    sizestuff = _find_sizes(df.spk_mean, df.size,sm_ratio,lg_ratio ;
      by_excess=by_excess)
    ff_rf = df.spk_ff[sizestuff.rf_size_idx]
    ff_lg = df.spk_ff[sizestuff.lg_size_idx]
    ff_sm = df.spk_ff[sizestuff.sm_size_idx]
    @eponymtuple(ff_rf,ff_lg,ff_sm)
  end
end

function ff_tablevalues_byneuron(dfdata, sm_ratio, lg_ratio;
        by_excess = false)
   k = ff_keyvalues_byneuron(dfdata, sm_ratio, lg_ratio;
                by_excess = by_excess)
return (
    delta_ff_rf_lg = score_straight.(k.ff_rf,k.ff_lg) |> mean |> round3 ,
    delta_ff_sm_rf = score_straight.(k.ff_sm,k.ff_rf) |> mean |> round3 ,
    delta_perc_ff_rf_lg = score_perc.(k.ff_rf,k.ff_lg) |> mean |> round3 ,
    delta_perc_ff_sm_rf = score_perc.(k.ff_sm,k.ff_rf) |> mean |> round3 ,
    p_ff_rf_lg = p_straight(k.ff_rf,k.ff_lg) |> round3 ,
    p_ff_sm_rf = p_straight(k.ff_sm,k.ff_rf) |> round3 ,
    p_perc_ff_rf_lg = p_perc(k.ff_rf,k.ff_lg) |> round3 ,
    p_perc_ff_sm_rf = p_perc(k.ff_sm,k.ff_rf) |> round3)
end

"""
    meanmatch_keyvalues_byseries(dfdata) -> dfdata_labels
Simply adds labels: `[:rf,:left,:right]` to each spike count
"""
function meanmatch_get_leftright(dfdata; byneuron=false)
  _selector = byneuron ? neuselector : vcat(neuselector,:series)
  dfd = deepcopy(dfdata)
  nsizes = length(unique(dfd.size))
  #@show nrow(dfneus(dfd))
  sort!(dfd,vcat(_selector,:size))
  mergh = combine(groupby(dfd,_selector;sort=true)) do df
    @assert nrow(df) == nsizes "unexpected number of size elements $(nrow(df))"
    sz = _find_sizes(df.spk_mean, df.size,0.5,2 ;
      by_excess=false)
    mm_labels = fill(:rf,nsizes)
    for i in 1:sz.rf_size_idx-1
      mm_labels[i] = :left
    end
    for i in sz.rf_size_idx+1:nsizes
      mm_labels[i] = :right
    end
    DataFrame(size=df.size, mm_label=mm_labels)
  end
  return join(dfd,mergh; on=vcat(_selector,:size))
end


function ff_differences_boot_byneu(dfdiffs)
  return by(dfdiffs,neuselector) do df
    _mu = mean(df.diff_rf)
    _sigms = @. (0.5(df.diff_rf_dup + df.diff_rf_ddown))^2
    err =  sqrt(sum(_sigms)) / length(_sigms)
    ret1 = (diff_rf = _mu , diff_rf_err = err)
    _mu = mean(df.diff_small)
    _sigms = @. (0.5(df.diff_small_dup + df.diff_small_ddown))^2
    err =  sqrt(sum(_sigms)) / length(_sigms)
    ret2 = (diff_small = _mu , diff_small_err = err)
    merge(ret1,ret2)
  end
end

function print_mean_and_pval(v , description="this thing")
  _scors = mean_boot(v)
  n = length(v)
  npos = count(v .> 0 )
  pval = spitp(v)
  @info("The mean value for $description is $(_scors.mean) , c.i."*
   "[$(_scors.mean_cidown),$(_scors.mean_ciup)]")
  @info("difference is positive for $npos (out of $n) elements. Ratio $(npos/n) ")
  @info("p value for FF score being positive is... \n p = $pval\n")
end



# data is by series
function get_rf_size_byseries(df)
  serselector = vcat(neuselector,:series)
  return combine(groupby(df,serselector)) do df
    (_, is_rf,_) = _rf_and_large_sizes(df.spk_mean,df.size,false)
    DataFrame(rfsize=df.size[is_rf])
  end
end
