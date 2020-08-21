
# get list of files to read
dir_nat_sizetun() = joinpath(read_dirfile()["dir_exp"],"natural_sizetuning")

# needed to link the size to the view parameters
function _phase_to_xy(ph)
  angs = collect(45:90:315)
  idx = [1,3,4,2]
  idxC= CartesianIndices((2,2))
  idxC[idx[angs .== ph][1]]
end
function _get_stim_params(imgID,phase,infoquads)
  xy = _phase_to_xy(phase)
  vals = infoquads[Tuple(xy)..., imgID]
end
# read dataframe with stimuli (a.k.a. views) information
function read_views_natural_sizetuning(file)
  dfview_ret_raw = matread(file)["viewInfo"]
  dfview_ret = DataFrame()
  dfview_ret[!,:view] = Int8.(dfview_ret_raw[:,1])
  dfview_ret[!,:quadID] = Int8.(dfview_ret_raw[:,2])
  dfview_ret[!,:phaseImg] = Int16.(dfview_ret_raw[:,3])
  stim_sizes = standardize_size.([0.33,0.55,0.9,2.3,3.8,3.8];atol=0.15)
  stim_hasgap = [falses(5)...,true]
  infoquads = let sz = repeat(stim_sizes,outer=10),
    sz = reshape(sz,2,2,15),
    gp = repeat(stim_hasgap,outer=10),
    gp = reshape(gp,2,2,15),
    imgid = repeat(Int8.(collect(1:10)), inner=6),
    imgid = reshape(imgid,2,2,15)
    collect(zip(imgid,sz,gp))
  end
  pars =_get_stim_params.(dfview_ret.quadID,dfview_ret.phaseImg,Ref(infoquads))
  dfview_ret[!,:image] = getindex.(pars,1)
  dfview_ret[!,:size] = getindex.(pars,2)
  dfview_ret[!,:hasgap] = getindex.(pars,3)
  return dfview_ret
end


function SpikingData_natural_sizetuning()
  filenames = filter(f-> occursin(r"\.mat\b",f),  readdir(dir_nat_sizetun()))
  filenames = joinpath.(dir_nat_sizetun(),filenames)
  time_bins = collect(range(-0.15,0.349 ; length=500))
  @info "Now reading .mat data files"
  datadfs = map(filenames) do f
    println("reading file "*f)
    d=matread(f)
    mat = d["spikesNstuff"]
    mat = copy( mat[(mat[:,4] .!= 0.0) .& (mat[:,4] .!= 255.0) , : ])
    df_spiketimes = DataFrame(
      session = basename(f),
      electrode = UInt8.(mat[:,3]),
      neuron = UInt8.(mat[:,4]),
      trial = UInt16.(mat[:,1]),
      view = UInt8.(mat[:,2]),
      spiketime = mat[:,5] )
    return binarybinformat(df_spiketimes,time_bins)
  end
  spikes = vcat(datadfs...)
  categorical!(spikes,:session;compress=true)
  time_stim = 0.0
  times=midpoints(time_bins)
  # read the views
  views = read_views_natural_sizetuning(filenames[1])
  # define the object
  return SpikingData(spikes, views, time_stim,time_bins,times)
end

function population_average_sizetuning_old(dfpop,primary_feature::Symbol=:size)
  return combine(groupby(dfpop, primary_feature; sort=true)) do df
    _spk_means = [s for s in df.spk_rate_rel]
    _spk_ffs = [s for s in df.spk_ff]
    spk_mean, spk_cidown, spk_ciup, spk_ddown, spk_dup = mean_boot(_spk_means)
    ff_geomean, ff_cidown, ff_ciup, ff_ddown,ff_dup = geomean_boot(_spk_ffs)
    @eponymtuple( spk_mean, spk_cidown, spk_ciup , spk_ddown, spk_dup,
      ff_geomean, ff_cidown, ff_ciup , ff_ddown, ff_dup)
  end
end


function population_average_sizetuning(dfpop,primary_feature::Symbol=:size)
  function do_averages(means,ffs)
    bmeans = mean_boot(means)
    bffs = geomean_boot(ffs)
    return merge(bmeans,bffs)
  end
  dfret=combine([:spk_mean_rel,:spk_ff]=>do_averages,
   groupby(dfpop, primary_feature))
  return sort!(dfret,primary_feature)
end
