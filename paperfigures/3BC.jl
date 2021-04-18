
using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics
using DataFrames, DataFramesMeta

## load data from .mat files, convert to object

dataspikes_monyet = F.read_surrori_monyet()
dataspikes_cadet = F.read_surrori_cadet()

@info "Data files read!"

## parameters
const time_count = 200E-3
const window_0 = (-20E-3,30E-3)
const window_spk = (50E-3,50E-3+time_count)
const secondary_feat_cadet = [:oriC]
const secondary_feat_monyet = [:contrast,:oriC]
const kthresh= 1.0
const k_latency = 1.0
const min_latency = 40E-3
const max_latency = 90E-3
const sizes = F.get_sizes(dataspikes_cadet)
const window_stim = window_spk
const window_blank = window_0

const data_filters = [
    F.BestOri(),
    F.HighestContrast(0.99),
    F.AverageFFLower(2.0) ]

##
# This is equivalent to the procedure for figure 2F
# but repeated on the two datasets
views_included_cadet = F.get_views_included_surrori(dataspikes_cadet ; kthresh=kthresh ,
        secondary_features=[:oriC],
        window_stim=window_spk, window_blank=window_0)
views_included_monyet = F.get_views_included_surrori(dataspikes_monyet ; kthresh=kthresh ,
        secondary_features=[:oriC, :contrast],
        window_stim=window_spk, window_blank=window_0)
data_latency_cadet = F.compute_latency(dataspikes_cadet,views_included_cadet ;
                min_latency=min_latency , max_latency=max_latency, k_latency=k_latency)
data_latency_monyet = F.compute_latency(dataspikes_monyet,views_included_monyet ;
        min_latency=min_latency , max_latency=max_latency, k_latency=k_latency)

data_spontaneous_cadet = F.get_spontaneus_rates(dataspikes_cadet, window_blank)
data_spontaneous_monyet = F.get_spontaneus_rates(dataspikes_monyet, window_blank)
data_spontaneous = vcat(data_spontaneous_cadet,data_spontaneous_monyet)


data_responses_cadet = F.get_responses_window(dataspikes_cadet, window_spk)
data_responses_monyet = F.get_responses_window(dataspikes_monyet, window_spk)
data_responses = vcat(data_responses_cadet,data_responses_monyet)



##
# now spikecounts for selected stimuli
data_spikecounts_cadet = let dat=dataspikes_cadet , dflat=data_latency_cadet,
  dfw = views_included_cadet,
  # select the neurons included, and the views included
  spikes = semijoin(dat.spikes, dflat ; on=F.neuselector)
  spikes = semijoin(spikes, dfw;on=vcat(F.neuselector,:view))
  times = dat.times
  # add views parameters, latency
  dfcount =F.count_spikes(spikes,times,dflat; time_count=time_count)
  ret = innerjoin( dfcount , dat.views ; on=:view,matchmissing=:equal)
  sort!(ret,vcat(F.neuselector,secondary_feat_cadet))
end


data_spikecounts_monyet = let dat=dataspikes_monyet , dflat=data_latency_monyet,
  dfw = views_included_monyet,
  # select the neurons included, and the views included
  spikes = semijoin(dat.spikes, dflat ; on=F.neuselector)
  spikes = semijoin(spikes, dfw; on=vcat(F.neuselector,:view))
  times = dat.times
  # add views parameters, latency
  dfcount =F.count_spikes(spikes,times,dflat; time_count=time_count)
  ret = innerjoin( dfcount , dat.views ; on=:view,matchmissing=:equal)
  sort!(ret,vcat(F.neuselector,secondary_feat_monyet))
end

data_spikecounts = let dfc=deepcopy(data_spikecounts_cadet),
  dfm = deepcopy(data_spikecounts_monyet)
  # if there is surround, there is a gap too, redundant
  select!(dfc,Not([:oriSAbs,:hasgap,:size]))
  dfc[!,:contrast] .= 1.0
  vcat(dfc,dfm)
end

categorical!(data_spikecounts,:contrast)

data_spikecounts_series = let dat=data_spikecounts,
    df =  F.define_series_surrori(data_spikecounts;
          secondary_features=secondary_feat_monyet)
    sort!(df,vcat(F.neuselector,:series))
end

##

data_series_filt = F.filter_data(data_spikecounts_series, data_filters...)
data_series_neus = F.average_over_series_surrori(data_series_filt)
data_pop_average = F.population_average_surrori(data_series_neus;ci=0.68)


## Histogram of ff differences, considering the averages over series for each neuron
#

function get_keyvals_neus(df_neus)
  if !( "ff_rel" in names(df_neus) )
    @warn "adding relative ff measure to series data"
    df_neus = D.add_relative_ff(df_neus)
  end
  return combine(groupby(df_neus,D.neuselector)) do df
    dfrf = @where(df , ismissing.(:oriS))
    @assert nrow(dfrf) == 1 ; dfrf = dfrf[1,:]
    dflg = @where(df , .!ismissing.(:oriS),:oriS .== 0)
    @assert nrow(dflg) == 1 ; dflg = dflg[1,:]
    dfpe = @where(df , .!ismissing.(:oriS),:oriS .== 90)
    @assert nrow(dfpe) == 1 ; dfpe = dfpe[1,:]
    # add if difference is is significant !
    ff_diff_pelg,ff_diff_pelg_issigni =
       D.ff_diff_signi(dfpe.spk_count,dflg.spk_count)
    rate_diff_pelg,rate_diff_pelg_issigni =
      D.rate_diff_signi(dfpe.spk_count,dflg.spk_count)
    (ff_rf = dfrf.spk_ff ,  ff_lg = dflg.spk_ff ,ff_pe = dfpe.spk_ff,
      ff_rel_rf = dfrf.ff_rel ,  ff_rel_lg = dflg.ff_rel ,ff_rel_pe = dfpe.ff_rel,
      spk_rf = dfrf.spk_mean ,  spk_lg = dflg.spk_mean, spk_pe = dfpe.spk_mean,
      spk_rel_rf = dfrf.spk_mean / dfrf.spk_mean ,
      spk_rel_lg = dflg.spk_mean / dfrf.spk_mean,
      spk_rel_pe = dfpe.spk_mean / dfrf.spk_mean,
      ff_diff_pelg = ff_diff_pelg, ff_diff_pelg_issigni=ff_diff_pelg_issigni,
      rate_diff_pelg = rate_diff_pelg, rate_diff_pelg_issigni=rate_diff_pelg_issigni)
  end
end
function get_keyvals_means(dfkvals)
  #spk first
  spk_rel_lg = D.mean_boot(dfkvals.spk_rel_lg; prefix="spk_rel_lg_mean")
  spk_rel_pe = D.mean_boot(dfkvals.spk_rel_pe; prefix="spk_rel_pe_mean")
  p_spk_rel_pelg = D.spitp(dfkvals.spk_rel_pe,dfkvals.spk_rel_lg)
  #spk , non relative
  spk_lg = D.mean_boot(dfkvals.spk_lg; prefix="spk_lg_mean")
  spk_pe = D.mean_boot(dfkvals.spk_pe; prefix="spk_pe_mean")
  p_spk_pelg = D.spitp(dfkvals.spk_pe,dfkvals.spk_lg)
  # now fanos
  ff_rel_lg = D.geomean_boot(dfkvals.ff_rel_lg; prefix="ff_rel_lg_geomean")
  ff_rel_pe = D.geomean_boot(dfkvals.ff_rel_pe; prefix="ff_rel_pe_geomean")
  #non relative version
  ff_lg = D.geomean_boot(dfkvals.ff_lg; prefix="ff_lg_geomean")
  ff_pe = D.geomean_boot(dfkvals.ff_pe; prefix="ff_pe_geomean")
  p_ff_rel_pelg = D.spitp(dfkvals.ff_rel_pe,dfkvals.ff_rel_lg)
  p_ff_pelg = D.spitp(dfkvals.ff_pe,dfkvals.ff_lg)
  # score for spk and FF , and p value of score
  scorefun(a,b)= 200*(a-b)/(a+b)
  _scors =   scorefun.(dfkvals.spk_pe,dfkvals.spk_lg)
  scor_spk_pelg = D.mean_boot(_scors ;  prefix="spk_score_pelg_mean"  )
  p_spk_score_pelg = D.spitp(_scors)
  _scors =   scorefun.(dfkvals.spk_rf,dfkvals.spk_lg)
  scor_spk_rflg = D.mean_boot(_scors ; prefix="spk_score_rflg_mean"  )
  p_spk_score_rflg = D.spitp(_scors)
  _scors =   scorefun.(dfkvals.ff_rf,dfkvals.ff_lg)
  scor_ff_rflg = D.mean_boot( _scors ; prefix="ff_score_rflg_mean"  )
  p_ff_score_rflg =D.spitp(_scors)
  _scors =   scorefun.(dfkvals.ff_pe,dfkvals.ff_lg)
  scor_ff_pelg = D.mean_boot(_scors ; prefix="ff_score_pelg_mean"  )
  p_ff_score_pelg = D.spitp(_scors)
  return merge(spk_lg,spk_pe,spk_rel_lg,spk_rel_pe,ff_rel_lg,ff_rel_pe,ff_lg,ff_pe,
    scor_spk_pelg,scor_spk_rflg,scor_ff_pelg,scor_ff_rflg,
    @eponymtuple(p_spk_pelg,p_spk_rel_pelg,p_ff_pelg,p_ff_rel_pelg,
    p_spk_score_pelg,p_spk_score_rflg,p_ff_score_pelg,p_ff_score_rflg ))
end
keyvals_neus = get_keyvals_neus(data_series_filt)
keyvals_neus_pop = get_keyvals_means(keyvals_neus)
# histogram plot
_ = if 1==1 let dat = keyvals_neus, nbins=15,
  nneus = nrow(D.dfneus(dat))
  scorefun(a,b)= 200*(a-b)/(a+b)
  dscore = scorefun.(dat.ff_rel_pe, dat.ff_rel_lg)
  D.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  h1 = fit(Histogram,dscore,bins)
  plot(h1 ; leg=false , title = "Cadet and Monyet, N = $nneus neurons")
  h2 = fit(Histogram,dscore[dat.ff_diff_pelg_issigni],bins)
  plot!(h2 ; leg=false)
end end
# And what about spike count score?
_ = if 1==1 let dat = keyvals_neus, nbins=15,
  nneus = nrow(D.dfneus(dat))
  scorefun(a,b)= 200*(a-b)/(a+b)
  dscore = scorefun.(dat.spk_pe, dat.spk_lg)
  D.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  h1 = fit(Histogram,dscore,bins)
  plot(h1 ; leg=false , title = "Cadet and Monyet, N = $nneus neurons")
  h2 = fit(Histogram,dscore[dat.rate_diff_pelg_issigni],bins)
  plot!(h2 ; leg=false)
end end

savefig("/tmp/ff_score_hist.png")

# histogram plot for RATIO
_ = if 1==1 let dat = keyvals_neus, nbins=15,
  nneus = nrow(D.dfneus(dat))
  scorefun(a,b)= (a/b)
  # dscore = scorefun.(dat.ff_rel_pe, dat.ff_rel_lg)
  dscore = scorefun.(dat.ff_pe, dat.ff_lg)
  D.print_mean_and_pval(dscore .- 1 ," perp Vs large , ratio - 1")
  bins = range(0,2;length=nbins)
  h1 = fit(Histogram,dscore,bins)
  plot(h1 ; leg=false , title = "Cadet and Monyet, N = $nneus neurons")
  h2 = fit(Histogram,dscore[dat.ff_diff_pelg_issigni],bins)
  plot!(h2 ; leg=false)
end end

savefig("/tmp/ff_score_rat.png")

# what about just the p value ?
_ = let dat=keyvals_neus
  p1 = D.spitp(dat.ff_pe,dat.ff_lg)
  p2 = D.spitp(dat.ff_rel_pe,dat.ff_rel_lg)
  @info "\n pvalue $p1 , pvalue relative $p2"
end
_ = let dat=keyvals_neus
  p1 = D.spitp_type2(dat.ff_pe,dat.ff_lg)
  p2 = D.spitp_type2(dat.ff_rel_pe,dat.ff_rel_lg)
  @info "\n pvalue $p1 , pvalue relative $p2"
end

_ = let dat=keyvals_neus
  p1 = D.spitp_type3(dat.ff_pe,dat.ff_lg)
  p2 = D.spitp_type3(dat.ff_rel_pe,dat.ff_rel_lg)
  @info "\n pvalue $p1 , pvalue relative $p2"
end



## Just print the average values on screen
println("\n\n")
for k in keys(keyvals_neus_pop)
  println(k)
end
_ = let d = keyvals_neus_pop
  println("\n")
  @info """
  average FF for uniform stimulus (RELATIVE TO RF):
  $(d.ff_rel_lg_geomean) [$(d.ff_rel_lg_geomean_ciup) $(d.ff_rel_lg_geomean_cidown)]
  average FF for orthogonal stimulus (RELATIVE TO RF):
  $(d.ff_rel_pe_geomean) [$(d.ff_rel_pe_geomean_ciup) $(d.ff_rel_pe_geomean_cidown)]
  and the p value is... $(d.p_ff_rel_pelg)
  \n
  average FF for uniform stimulus:
  $(d.ff_lg_geomean) [$(d.ff_lg_geomean_ciup) $(d.ff_lg_geomean_cidown)]
  average FF for orthogonal stimulus:
  $(d.ff_pe_geomean) [$(d.ff_pe_geomean_ciup) $(d.ff_pe_geomean_cidown)]
  and the p value is... $(d.p_ff_pelg)

  Mean score and p value, RF Vs matched:
  $(d.ff_score_rflg_mean) and p = $(d.p_ff_score_rflg)

  Mean score and p value, orthogonal Vs matched:
  $(d.ff_score_pelg_mean) and p = $(d.p_ff_score_pelg)
  """
end

# The non-relative value is expressed in Hz

_ = let d = keyvals_neus_pop
  tohz(x) = x/time_count
  println("\n")
  @info """
  average response for uniform stimulus (Hz):
  $(tohz(d.spk_lg_mean)) [$(tohz(d.spk_lg_mean_ciup)) $(tohz(d.spk_lg_mean_cidown))]
  average reponse for orthogonal stimulus (Hz):
  $(tohz(d.spk_pe_mean)) [$(tohz(d.spk_pe_mean_ciup)) $(tohz(d.spk_pe_mean_cidown))]
  and the p value is... $(d.p_spk_pelg)
  \n
  average response for uniform stimulus (RELATIVE TO RF):
  $(d.spk_rel_lg_mean) [$(d.spk_rel_lg_mean_ciup) $(d.spk_rel_lg_mean_cidown)]
  average reponse for orthogonal stimulus (RELATIVE TO RF):
  $(d.spk_rel_pe_mean) [$(d.spk_rel_pe_mean_ciup) $(d.spk_rel_pe_mean_cidown)]
  and the p value is... $(d.p_spk_rel_pelg)


  Mean score and p value, RF Vs matched:
  $(d.spk_score_rflg_mean) and p = $(d.p_spk_score_rflg)

  Mean score and p value, orthogonal Vs matched:
  $(d.spk_score_pelg_mean) and p = $(d.p_spk_score_pelg)
  """
end

## Select ONLY neurons with a positive orientation tuning score (on rates)
# and pring again all those values

const data_filters_ortuning = [
    D.BestOri(),
    D.HighestContrast(0.99),
    D.AverageFFLower(2.0),
    D.OrientationTuning(0.0) ]


data_series_filt_or = D.filter_series(data_spikecounts_series, data_filters_ortuning...)
keyvals_neus_pop_or = get_keyvals_means(get_keyvals_neus(data_series_filt_or))


_ = let d = keyvals_neus_pop_or
  println("\n")
  @info """
  average FF for uniform stimulus (RELATIVE TO RF):
  $(d.ff_rel_lg_geomean) [$(d.ff_rel_lg_geomean_ciup) $(d.ff_rel_lg_geomean_cidown)]
  average FF for orthogonal stimulus (RELATIVE TO RF):
  $(d.ff_rel_pe_geomean) [$(d.ff_rel_pe_geomean_ciup) $(d.ff_rel_pe_geomean_cidown)]
  and the p value is... $(d.p_ff_rel_pelg)
  \n
  average FF for uniform stimulus:
  $(d.ff_lg_geomean) [$(d.ff_lg_geomean_ciup) $(d.ff_lg_geomean_cidown)]
  average FF for orthogonal stimulus:
  $(d.ff_pe_geomean) [$(d.ff_pe_geomean_ciup) $(d.ff_pe_geomean_cidown)]
  and the p value is... $(d.p_ff_pelg)

  Mean score and p value, RF Vs matched:
  $(d.ff_score_rflg_mean) and p = $(d.p_ff_score_rflg)

  Mean score and p value, orthogonal Vs matched:
  $(d.ff_score_pelg_mean) and p = $(d.p_ff_score_pelg)
  """
end

# The non-relative value is expressed in Hz

_ = let d = keyvals_neus_pop_or
  tohz(x) = x/time_count
  println("\n")
  @info """
  average response for uniform stimulus (Hz):
  $(tohz(d.spk_lg_mean)) [$(tohz(d.spk_lg_mean_ciup)) $(tohz(d.spk_lg_mean_cidown))]
  average reponse for orthogonal stimulus (Hz):
  $(tohz(d.spk_pe_mean)) [$(tohz(d.spk_pe_mean_ciup)) $(tohz(d.spk_pe_mean_cidown))]
  and the p value is... $(d.p_spk_pelg)
  \n
  average response for uniform stimulus (RELATIVE TO RF):
  $(d.spk_rel_lg_mean) [$(d.spk_rel_lg_mean_ciup) $(d.spk_rel_lg_mean_cidown)]
  average reponse for orthogonal stimulus (RELATIVE TO RF):
  $(d.spk_rel_pe_mean) [$(d.spk_rel_pe_mean_ciup) $(d.spk_rel_pe_mean_cidown)]
  and the p value is... $(d.p_spk_rel_pelg)


  Mean score and p value, RF Vs matched:
  $(d.spk_score_rflg_mean) and p = $(d.p_spk_score_rflg)

  Mean score and p value, orthogonal Vs matched:
  $(d.spk_score_pelg_mean) and p = $(d.p_spk_score_pelg)
  """
end


## Save histograms for the figure

function ntup_to_df(tup)
  df = DataFrame()
  for (k,v) in pairs(tup)
    df[!,k] = [v]
  end
  return df
end

function do_the_sav(dat,nbins)
  nneus = nrow(D.dfneus(dat))
  scorefun(a,b)= 200*(a-b)/(a+b)
  dscore = scorefun.(dat.ff_pe, dat.ff_lg)
  D.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  binsc = midpoints(bins)
  h1 = fit(Histogram,dscore,bins)
  h2 = fit(Histogram,dscore[dat.ff_diff_pelg_issigni],bins)
  w_ff=Float64.(h1.weights)
  ws_ff=Float64.(h2.weights)
  w_ff[w_ff .== 0] .= NaN
  ws_ff[ws_ff .== 0] .= NaN
  # same for spike counts
  dscore = scorefun.(dat.spk_pe, dat.spk_lg)
  D.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  binsc = midpoints(bins)
  h1 = fit(Histogram,dscore,bins)
  h2 = fit(Histogram,dscore[dat.rate_diff_pelg_issigni],bins)
  w_spk=Float64.(h1.weights)
  ws_spk=Float64.(h2.weights)
  w_spk[w_spk .== 0] .= NaN
  ws_spk[ws_spk .== 0] .= NaN
  dfout = DataFrame(binsc=binsc, ff_score_all = w_ff , ff_score_signi = ws_ff ,
      spk_score_all = w_spk , spk_score_signi = ws_spk)
end

do_the_sav(keyvals_neus,15)


# save it!
_ = let tosav = do_the_sav(keyvals_neus,15),
  savname = joinpath(this_dir,date2str()*"AwakeBothSurroundModulationHist.csv")
  CSV.write(savname,tosav)
  @info "all saved in $savname , exiting..."
end
exit()
