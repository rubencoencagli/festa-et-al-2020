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
         F.MinMeanCount(0.01), # no mean spk count below 0.01
         F.ResponseScore(1.0), # best response should be above baseline + 1 std baseline
         F.AverageFFLower(2.0),
         F.ByRFSizes([2,3]), # RF size should only be 0.5 or 0.9 , not smaller or higher
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
data_spikecounts_series = df =  F. define_series(data_spikecounts;
      secondary_features=secondary_features)
##

data_series_filt = F.filter_data(data_spikecounts_series, data_filters...)

##

function pop_avg_relativesize(data)
  serselector = vcat(D.neuselector,:series)
  sizes_all = sort(unique(data.size))
  sizes_good = sizes_all ./ sizes_all[2]
  data_filt =  D.filter_series(data,
       D.NoNatGap(),
       D.ByRFSizes([2,3]),
       #D.AverageFFLower(2.0),
       D.SurroundSuppression(0.30) )
  data_rf = D.get_rf_size_byseries(data_filt)
  data  = join(data_filt,data_rf ; on=serselector)
  data[!,:size] = data.size ./ data.rfsize
  data[!,:size] = _standardsizes.(data.size, Ref(sizes_good))
  nneus = nrow(D.dfneus(data))
  data_neus = D.average_over_series_natareasumm(data)
  data_pop = D.population_average_natareasumm(data_neus)
  return nneus,data,data_neus,data_pop
end
## single neuron examples!
using Formatting
@nm pop_avg_relativesize(data_spikecounts_series)[2]


whatevs = if true let data=data_spikecounts_series,
  (nneus,dat,_,_) = pop_avg_relativesize(data)
  k = 0
  myfold=mktempdir(this_dir)
  function figname()
    k+=1
    joinpath(myfold,fmt("03d",k)*".svg")
  end
  by(dat,vcat(D.neuselector,:series)) do _df
    df = _df[:,:]
    sort!(df,:size)
    p = plot(
    title="$(df.session[1]) , $(df.electrode[1]) , $(df.neuron[1]), S$(df.series[1])" ,
      xlabel="stimulus  size relative to RF")
    xplot = df.size
    plot!(p, xplot, df.spk_mean;
    linewidth=3, marker=:diamond , label="mean spike count (relative)",
    xlims=(0.3,10),  xscale=:log10,
    color=colorant"Orange Red")
    plot!(twinx(), xplot, df.spk_ff;
    linewidth=3, marker=:diamond , label="geomean FF",legend=:topleft,
    color=:blue ,xlims=(0.3,10),  xscale=:log10)
    savefig(p,figname())
    nothing
  end
end end
