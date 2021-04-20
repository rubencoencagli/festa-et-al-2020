using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics
using DataFrames, DataFramesMeta
using EponymTuples
## load data from .mat files, convert to object

dataspikes= F.SpikingData_pvc8()
##

const time_count = 106E-3 # duration of a trial
const window_stim = (50E-3,50E-3+time_count) # window for preliminary spike count
const window_blank = (-20E-3,30E-3) # window for spontaneous activity
const secondary_features = [:phase,:ori,:natimg] # features to average over
const kthresh=1. # number of stds above spontaneous activity
const k_latency = 1. # number of stds used to compute latency
# boundaries for latency
const min_latency = 40E-3
const max_latency = 90E-3
const sizes = F.get_sizes(dataspikes)

## Filters used to select the neurons
data_filters =[
        F.OnlyNat(), # exclude grating stimuli
        F.MinMeanCount(0.01), # no mean spk count below 0.01
        F.ResponseScore(1.0), # best response should be above baseline + 1 std baseline
        F.AverageFFLower(2.), # FF averaged on all stimuli below 2
        F.NSeriesMin(10)] # the neuron should respond to 10 or more images

##

# Include only responsive stimuli
views_included= F.get_views_included(dataspikes ;
  @eponymtuple(kthresh,secondary_features,window_stim,window_blank)...)


# compute latencies for each neuron
data_latency = F.compute_latency(dataspikes,views_included ;
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

data_neus_scatter = F.average_ff_over_series_pvc8_natimg(data_series_filt ;
  ci=0.68)

## scatter plots
plt = let dat =  data_neus_scatter,
  _scal = (-0.1,0.35),
   nneus = nrow(dat),
   dfno = @where(dat, .! :issigni)
   dfsi = @where(dat,:issigni)
   nsign = count(dat.issigni)
   scatter(dfno.ff_sm,dfno.ff_lg ; ratio=1, color=:gray)
   scatter!(dfsi.ff_sm,dfsi.ff_lg ; ratio=1, color=:blue)
   @info """ statistical significance
   Neurons with significant difference in FF :  $nsign (out of $nneus)
   Of those, the FF is lower for large size in $(count(dfsi.ff_sm .> dfsi.ff_lg)) (out of $(nrow(dfsi)))
   """
   plot!(identity; leg=false, scale=:log10,
    xlim = 10 .^_scal, ylim =10 .^_scal,
    title="N=$nneus", xlabel="Fano factor small size",
    ylabel="Fano factor large size")
end

# save it locally
mkpath(F.dir_plots)
figname = date2str()*"_main_2C.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
