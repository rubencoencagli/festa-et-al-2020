
using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics,StatsBase
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


keyvals_neus = F.surrori_get_keyvals_neus(data_series_filt)
keyvals_neus_pop = F.surrori_get_keyvals_means(keyvals_neus)
# histogram plot
ff_hist =  let dat = keyvals_neus, nbins=15,
  nneus = nrow(F.dfneus(dat))
  scorefun(a,b)= 200*(a-b)/(a+b)
  dscore = scorefun.(dat.ff_rel_pe, dat.ff_rel_lg)
  F.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  h1 = fit(Histogram,dscore,bins)
  plot(h1 ; leg=false , title = "Two awake monkeys, $nneus neurons")
  h2 = fit(Histogram,dscore[dat.ff_diff_pelg_issigni],bins)
  plot!(h2 ; leg=false, xlabel="FF orthogonal - matched %")
end
# And what about spike count score?
spk_hist = let dat = keyvals_neus, nbins=15,
  nneus = nrow(F.dfneus(dat))
  scorefun(a,b)= 200*(a-b)/(a+b)
  dscore = scorefun.(dat.spk_pe, dat.spk_lg)
  F.print_mean_and_pval(dscore," perp Vs large")
  bins = range(-70,70;length=nbins)
  h1 = fit(Histogram,dscore,bins)
  plot(h1 ; leg=false , title = "Two awake monkeys, $nneus neurons")
  h2 = fit(Histogram,dscore[dat.rate_diff_pelg_issigni],bins)
  plot!(h2 ; leg=false, xlabel="spike count orthogonal - matched %" )
end

plt=plot(spk_hist,ff_hist;layout=(1,2),size=(800,400))

# save it locally
mkpath(F.dir_plots)
figname = date2str()*"_main_3BC.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"


exit()


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
