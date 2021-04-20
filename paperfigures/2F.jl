using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics
using DataFrames, DataFramesMeta
using EponymTuples
## load data from .mat files, convert to object
dataspikes= F.SpikingData_natural_sizetuning()

##

const time_count = 200E-3 # stimulus duration
const window_stim = (50E-3,50E-3+time_count) # window for preliminary spike count
const window_blank = (-20E-3,30E-3) # window for spontaneous activity
const secondary_features = [:image] # features to average over
const kthresh=1. # number of stds above spontaneous activity
const k_latency = 1. # number of stds used to compute latency
# boundaries for latency
const min_latency = 40E-3
const max_latency = 90E-3
const sizes = F.get_sizes(dataspikes)

## Filters used to select the neurons
data_filters =[
         F.NoNatGap(), # exclude stimulus with gap
         F.AverageFFLower(2.0),
         F.ByRFSizes([2,3]),# RF size should only be 0.5 or 0.9 , not smaller or higher
         F.SurroundSuppression(0.15) ] # sufficient rate surround suppression
##
# Include only responsive stimuli
views_included = F.get_views_included(dataspikes ;
  @eponymtuple(kthresh,secondary_features,window_stim,window_blank)...)

# compute latencies for each neuron
data_latency =
      F.compute_latency(dataspikes,views_included ;
  @eponymtuple(min_latency,max_latency,k_latency)...)

# spontaneous rates (for inspection)
data_spontaneous = F.get_spontaneus_rates(dataspikes, window_blank)


# now spikecounts for selected stimuli
data_spikecounts = let dat=dataspikes ,
  # select the neurons included, and the views included
  spikes = semijoin(dat.spikes, data_latency ; on=F.neuselector)
  spikes = semijoin(spikes, views_included; on=vcat(F.neuselector,:view))
  times = dat.times
  # add views parameters, latency
  dfcount =F.count_spikes(spikes,times,data_latency; time_count=time_count)
  ret = innerjoin( dfcount , dat.views ; on=:view)
  sort!(ret,F.neuselector)
end

# define series using secondary features
data_spikecounts_series = F. define_series(data_spikecounts;
      secondary_features=secondary_features)

##
data_series_filt = F.filter_data(data_spikecounts_series, data_filters...)

##

data_series_relsizes=F.relativize_sizes(data_series_filt)

data_neus=F.average_over_series(data_series_relsizes,:size)

@show names(data_neus);

data_pop=F.population_average_sizetuning(data_neus)

plt = let dat=data_pop,
  nneus=F.nneus(data_neus),
  p=plot(title="Data, natural area summation curve, N = $nneus" ,
      xlabel="stimulus  size relative to RF" , ylims=(0,1.6))
  xplot = dat.size
  plot!(p, xplot, dat.mean; ribbon=[dat.mean_ddown,dat.mean_dup],
    linewidth=3, marker=:diamond , label="mean spike count (normalized)",
    ylims=(0,1.2), xlims=(0.3,10),  xscale=:log10,
    color=colorant"DarkGreen")
  plot!(twinx(p), xplot, dat.geomean;
    ribbon= [dat.geomean_ddown,dat.geomean_dup],
    linewidth=3, marker=:diamond , label="geomean FF",legend=:topleft,
    color=:blue , ylims = (0.9,1.7) ,xlims=(0.3,10),  xscale=:log10)
end
##
# save the plot
mkpath(F.dir_plots)
figname = date2str()*"_main_2F.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
