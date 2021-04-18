using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics
using DataFrames, DataFramesMeta
using EponymTuples
## load data from .mat files, convert to object
dataspikes = F.SpikingData_natural_sizetuning()

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
         F.ByRFSizes([2,3]), # RF size should only be 0.5 or 0.9 , not smaller or higher
         ]

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
# size relative to RF size
data_relative_sizes = F.relativize_sizes(data_series_filt)

function get_data_single_neuron(data,session,electrode,neuron,series ; ci=0.68)
  all_sess = data.session |> unique
  dat = @where(data,
        :session .== all_sess[session],:electrode .==electrode,
        :neuron .== neuron , :series.==series)
  @assert !isempty(dat) "empty selection"
  ret = combine(groupby(dat, :size ; sort=true)) do df
   @assert nrow(df) .== 1
   spks=df.spk_count[1] # spike counts for specific view
   r1 = F.mean_boot(spks;conf=ci)
   r2 = F.ff_boot(spks;conf=ci)
   merge(r1,r2)
  end
  # normalize rates
  norm = inv(maximum(ret.mean))
  ret[!,:mean] .*=norm
  ret[!,:mean_ciup] .*=norm
  ret[!,:mean_cidown] .*=norm
  ret[!,:mean_dup] .*=norm
  ret[!,:mean_ddown] .*=norm
  return ret
end


function plot_single_neuron(df_dat)
  ret=df_dat
  # plot mean
  plot(ret.size,ret.mean ;
    color=colorant"forestgreen",
    linewidth=3,
    markerstrokecolor=colorant"forestgreen",
    markerstrokewidth=3,
    yerror=(ret.mean_dup,ret.mean_ddown),
    leg=false,xscale=:log10,
    xlabel="size (relative to RF)" , ylabel="rate (relatitve to RF)")
  # plot FF
  plot!(twinx(),ret.size,ret.ff ;
    color=colorant"blue",
    linewidth=3,
    markerstrokecolor=colorant"blue",
    markerstrokewidth=3,
    yerror=(ret.ff_dup,ret.ff_ddown),
    leg=false,xscale=:log10,
    ylabel="Fano factor")
end
function plot_single_neuron(data,session,electrode,neuron,series ; ci=0.68)
  dat = get_data_single_neuron(data,session,electrode,neuron,series ; ci=0.68)
  plot_single_neuron(dat)
end

## Select 2 neurons and specific stimuli, and plot
# N.B. the two example neurons here are different ones
plt1 = plot_single_neuron(data_relative_sizes, 1,80,1,6)
plt2 = plot_single_neuron(data_relative_sizes, 2,53,2,1)


plt = plot(plt1,plt2 ; layout=(2,1), size=(600,850))

# save the plot
mkpath(F.dir_plots)
figname = date2str()*"_main_2E.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
