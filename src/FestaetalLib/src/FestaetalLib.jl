


module FestaetalLib

# load module that computes complex steerable pyramids
pyramidsmodule = abspath(@__DIR__,"..","..","Pyramids","src","Pyramids.jl")
@assert isfile(pyramidsmodule) "file $pyramidsmodule not found"
include(pyramidsmodule)

# load all other library dependencies
using Dates,Formatting
using Statistics, LinearAlgebra, StatsBase
using Distributions, Random

using Images,OffsetArrays

using CmdStan,MCMCChains

using Serialization,JSON,MAT# For input and output
using DataFrames,DataFramesMeta

using SmoothingSplines # smoothing PSTH curves
using HypothesisTests,Bootstrap # confidence intervals and significance

using EponymTuples

# write date as string
date2str() = Dates.format(now(),"yyyymmdd")

# file with paths should be in the folder...
const dir_dirfile = abspath(@__DIR__,"..","..","..","data")
const dir_tmpstuff = abspath(@__DIR__,"..","..","tmp")
const dir_plots = abspath(@__DIR__,"..","..","..","plots")

const savename_gsm_obj = joinpath(dir_tmpstuff,date2str()*"GSM_model.jls")

function read_dirfile(file::Union{String,Nothing}=nothing)
  file=something(file, joinpath(dir_dirfile,"local_dirs.json"))
  @assert isfile(file) "file $file not found!"
  ret = open(file,"r") do f
    JSON.parse(f)
  end
  return ret
end

function read_gsm_path()
  (!isdir(dir_tmpstuff)) && return nothing
  gsm_saved = filter(s->occursin(r"GSM_model.+jls",s),readdir(dir_tmpstuff))
  isempty(gsm_saved) && return nothing
  return joinpath(dir_tmpstuff,gsm_saved[end])
end


function check_rebuild(args)
  isempty(args) &&  return false
  arg1 = args[1]
  (arg1 == "--rebuildGSM") && return true
  if occursin(r"--.+buildG",arg1)
    error("invalid option $arg1 , maybe you mean --rebuildGSM ?")
  end
  return false
end

include("naturalRead.jl")
include("steerPyramids.jl")
include("baseGSM.jl")
include("drawgratings.jl")
include("stanSampl.jl")



# reads the test images used in the area summation experiment from the specified matlab
# file, and adapts them to a specific frame size.
function read_test_images_natural_areasumm(fileimg,framesize)
  @assert isfile(fileimg) "file $fileimg not found! Please point at the correct file"
  images_raw = let d = matread(fileimg)
    [d["imagesTop"][2:2:end] ...]
  end
  ret_images = map(images_raw) do im
        size1 = framesize-1
        imc = Images.centered(im)
        return imc[-size1+1:size1,-size1+1:size1] |> restrict
  end
  return ret_images
end

# changes mean and variance of test images so that
# it reflects the mean and variance of train images
function make_test_samestats!(img_test,img_train)
  vtrain = vcat(img_train...)
  vtest = vcat(img_test...)
  μtr,σtr = mean_and_std(vtrain)
  @assert μtr < 1E-4 "$(μtr) is expected to be close to zero"
  μte,σte = mean_and_std(vtest)
  regufun = x -> ((x-μte)/σte )*σtr
  foreach(img -> (@. img=regufun(img)), img_test)
  return nothing
end


# wrapper of lib function in naturalRead.jl
# reads and stores a large set natural images from the specified folder
function read_natural_images(size_patch)
  dir_natural_images = read_dirfile()["dir_img"]
  @assert isdir(dir_natural_images) "$dir_natural_images not found!"
  @info "reading and processing natural images for training..."
  img_test = Images.load(get_file_paths(dir_natural_images,".jpg")[1])
  img_size= minimum(size(img_test)) - 5
  images_train = read_natural_images(dir_natural_images,size_patch,StandardRegu() ;
    verbose=false)
  @info "Now reading and processing images used in experiments"
  file_images_exp = joinpath(dir_natural_images,"experiment_images.mat")
  images_exp =  read_test_images_natural_areasumm(file_images_exp,size_patch)
  make_test_samestats!(images_exp,images_train)
  return images_train,images_exp
end

# wraps function contained in steerPyramids.jl
function make_filter_bank(pars)
  return OnePyrBank(pars.scale,pars.distance,pars.nsurround ; ncent=pars.ncenter)
end


# source code also in steerPyramids.jl
function show_filter_bank(pars,framesize)
  ncent = pars.ncenter
  nsurr = pars.nsurround
  bank=make_filter_bank(pars)
  img1 = invert_test(framesize,bank;idx_bank=[1,(ncent+1:ncent+nsurr)...])
  img2 = invert_test(framesize,bank;idx_bank=[1,(ncent+1:ncent+nsurr)...],real_part=false)
  padval = maximum(img1)
  imgboth = hcat(img1, fill(padval, framesize,round(Int,0.1framesize)),img2 )
  return imgboth
  #showbw(imgboth , "GSM filter bank (+ and - phase)") # function defined above
end

# source code in GSMLibs/src/naturalRead.jl (cuts random patches from full imgs)
# and in  GSMLib/steerPyramids.jl (applies pyramid to the patch)
function  compute_filter_outputs(bank, nat_imgs, size_patch , nsampl)
  @info """
  Now apply the steerable pyramid filters to $nsampl patches of size $size_patch
  Warning : for a high number of patches (e.g. 10,000) this process might take
  up to 15-20 minutes.
  """
  natpatches = sampling_tiles(nsampl, nat_imgs, size_patch)
  ret = apply_bank( natpatches, bank, Xstd(2.0),false)
  println("Outputs computed!")
  return ret
end

# wrapper of a function in GSMLibs/src/baseGSM.jl
function train_gsm_model(x_values,bank,size_patch,noise_level,nsamples)
  @info """
  Now training the GSM model with noise.
  Warning : for a high number of samples (e.g. 10,000) this process might take
  up to 15-20 minutes (banks are applied to white noise to obtain the noise covariance structure)
  """
  ret = GSM_from_filter_outputs(x_values, bank, size_patch, noise_level, nsamples ;
        mixer = RayleighMixer(1.0))
  println("Outputs computed!")
  return ret
end

# object defined in GSMLibs/src/baseGSM.jl
# basically GSM + filter bank + stimuli and samples of response (empty on initialization)
function build_gsm_object(gsm,noise_level,filt_pars,filter_bank,size_patch,x_values)
  scale =filt_pars.scale
  distance =filt_pars.distance
  ncenter =filt_pars.ncenter
  nsurround =filt_pars.nsurround
  GSM_Model(gsm,size_patch,scale,distance,ncenter,
      nsurround,noise_level,filter_bank,x_values)
end

function save_gsm_object(gsm_obj, experiment_images)
  mkpath(dir_tmpstuff)
  noise_scale_factor = gsm_obj.noise_level
  size_patch = size(experiment_images[1],1)
  savedict = Dict(
    "gsm_obj"=>gsm_obj,
    "noise_scale_factor" => noise_scale_factor,
    "size_patch"=> size_patch ,
    "exp_areasumm_images"=>experiment_images)
  open(savename_gsm_obj,"w") do f
    serialize(f,savedict)
  end
  @info "GSM model and parameters saved as $savename_gsm_obj"
end


# wrapper of a function in naturalRead.jl
function cut_natural_patches(natural_images,nsamples,size_patch)
  tiles = sampling_tiles(nsamples, natural_images, size_patch)
  return map(first,tiles)
end

# wrapper of a function in stanSampl.jl
function sample_posterior!(gsm_obj,n_sampl;addnoise=false)
  # set cmd-stan home directory
  dir_stan = read_dirfile()["dir_stan_home"]
  @assert isdir(dir_stan) "Problem in locating cmd-stan. Directory $dir_stan does not exist. Please install Stan and set the directory as indicated in the README"
  set_cmdstan_home!(dir_stan)
  @info "Now sampling the posterior of the hidden features using stan"
  sample_posterior(n_sampl,gsm_obj;addnoise=addnoise,dir_temp=dir_tmpstuff)
  @info "sampling ended successfully, results stored in gsm model structure"
  return nothing
end

# adds views to gsm object, in preparation for sampling
function store_image_inputs!(gsm_obj,imgs)
  views = DataFrame(
  idx = collect(1:length(imgs)),
  view = imgs)
  gsm_obj.views = views
end

function sample_posterior_imgs!(gsm_obj,n_sampl,imgs;addnoise=false)
  store_image_inputs!(gsm_obj,imgs)
  return sample_posterior!(gsm_obj,n_sampl;addnoise=addnoise)
end

function sample_posterior_views!(gsm_obj,n_sampl,views;addnoise=false)
  gsm_obj.views=views
  return sample_posterior!(gsm_obj,n_sampl;addnoise=addnoise)
end

# see baseGSM.jl
function extract_neural_response(gsm_obj,alph)
  gs = gsm_obj.samples.gs
  conv = GConvNorm(alph)
  return map(g->  g_to_r(g,conv),gs)
end



function fanofactor(spk::AbstractVector{R}; warn::Bool=true) where R<:Real
  μ,σ² = mean_and_var(spk)
  if μ == 0
    @warn "Mean spike count is 0, the Fano factor does not have a meaningful value."
    return 0.0
  end
  return σ²/μ
end


# Bootstrap functions

getdeltaerror(x, down,up) = (x-down, up-x)
"""
  mean_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="mean") where T <: Real
  -> NamedTuple , fields:  (mean, mean_cidown,mean_ciup, mean_down,mean_dup)
"""
function mean_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="mean") where T <: Real
    _meanbs = bootstrap(Statistics.mean, spk, BasicSampling(nrun))
    mean, mean_cidown, mean_ciup = confint(_meanbs, BCaConfInt(conf))[1]
    mean_ddown,mean_dup = getdeltaerror.(mean, mean_cidown, mean_ciup)
    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
    return (; zip(retnames, [mean,mean_cidown,mean_ciup,mean_ddown,mean_dup])...)
end
function var_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="var") where T <: Real
    _varboot = bootstrap(Statistics.var, spk, BasicSampling(nrun))
    var, var_cidown, var_ciup = confint(_varboot, BCaConfInt(conf))[1]
    var_ddown,var_dup = spitdeltaerror.(var, var_cidown, var_ciup)
    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
    return (; zip(retnames, [var,var_cidown,var_ciup,var_ddown,var_dup])...)
end
function ff_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="ff") where T <: Real
    _ffboot = bootstrap(fanofactor, spk, BasicSampling(nrun))
    ff, ff_cidown, ff_ciup = confint(_ffboot, BCaConfInt(conf))[1]
    ff_ddown,ff_dup = spitdeltaerror.(ff, ff_cidown, ff_ciup)
    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
    return (; zip(retnames, [ff,ff_cidown,ff_ciup,ff_ddown,ff_dup])...)
end
function geomean_boot(spk::AbstractVector{T} ;
        nrun = 5_000 , conf = 0.95,prefix="geomean") where T <: Real
    _meanbs = bootstrap(StatsBase.geomean, spk, BasicSampling(nrun))
    geomean, geomean_cidown, geomean_ciup = confint(_meanbs, BCaConfInt(conf))[1]
    geomean_ddown,geomean_dup = spitdeltaerror.(geomean, geomean_cidown, geomean_ciup)
    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
    return (; zip(retnames, [geomean,geomean_cidown,geomean_ciup,
        geomean_ddown,geomean_dup])...)
end
function median_boot(v::AbstractVector{T} ; nrun=5_000 , conf=0.95) where T<:Real
  bs_median = bootstrap(Statistics.median,v,BasicSampling(nrun))
  median,median_cidown,median_ciup =
          confint(bs_median,BCaConfInt(conf))[1]
  median_ddown,median_dup = spitdeltaerror.(median, median_cidown, median_ciup)
  return @eponymtuple(median,median_ciup,median_cidown,  median_ddown,median_dup )
end



# computes mean rate (relative to RF) and FF for each stimulus,
# then computes the averages for each stimulus size
function extract_mean_ff_bysize(gsm_obj,r_alpha)
  conv = GConvNorm(r_alpha)
  sampl = gsm_obj.samples
  views_size = select(gsm_obj.views, [:idx,:size,:img_idx])
  data = innerjoin(sampl,views_size ; on=:idx)
  # add spike counts to dataframe
  transform!(data, :gs => ByRow(g->  g_to_r(g,conv)) => :rs )
  # add mean spike counts and FFs
  transform!(data, [:rs => ByRow(mean)  => :spk_mean ,
      :rs => ByRow(fanofactor)   => :spk_ff] )
  # normalize by spike count at RF size (i.e. max over sizes)
  dat_byimg = groupby(data,:img_idx;sort=true)
  transform!(dat_byimg, :spk_mean => (spkmeans -> spkmeans./maximum(spkmeans)) =>
    :spk_mean_rel) # this changes data in place
  # now average over images, for each size
  dat_bysize = groupby(data,:size;sort=true)
  (sizes,r,r_dup,r_ddown,ff,ff_dup,ff_ddown) = (Float64[] for i in 1:7)
  for dat in dat_bysize
    push!(sizes,dat.size[1])
    # mean over the means, geometric means over FF, across images
    bootmeans = mean_boot(dat.spk_mean_rel)
    bootffs = geomean_boot(dat.spk_ff)
    push!(r,bootmeans.mean)
    push!(r_dup,bootmeans.mean_dup)
    push!(r_ddown,bootmeans.mean_ddown)

    push!(ff,bootffs.geomean)
    push!(ff_dup,bootffs.geomean_dup)
    push!(ff_ddown,bootffs.geomean_ddown)
  end
  return (sizes, r,r_dup,r_ddown,ff,ff_dup,ff_ddown)
end



function get_best_features(filter_bank,size_patch::Integer)
  @info "Computing the most responsive parameters for grating stimuli..."
  sfreqs = collect(4:1:30)
  cshifts = [0.0,]
  sphases = collect(range(0,pi;length=25))
  sizes = [60,]
  oris = [0.0,]
  contrasts = [1.0,]
  stims =  make_stimuli_disk(size_patch, sizes, oris, sfreqs, sphases,
      contrasts,cshifts; contrast_base=3.0)
  dfret1 = bank_best_response(filter_bank,stims,[:sfreq,:sphase];idxs=(1:2))
  best_sfreq = dfret1.sfreq
  best_sphase = dfret1.sphase
  sizes=[7,]
  sfreqs = [best_sfreq,]
  cshifts = range(-0.2,0.2;length=50)
  stims =  make_stimuli_disk(size_patch, sizes, oris, sfreqs, best_sphase,
      contrasts,cshifts; contrast_base=3.0)
  dfret2 = bank_best_response(filter_bank,stims,[:cshift]; idxs=(1:2))
  best_cshift = dfret2.cshift
  @info "... done!"
  return (
    sfreq = best_sfreq ,
    sphase =best_sphase ,
    cshift = best_cshift)
end

function make_surrori_stimuli(orientations,n_realizations::Integer,
    size_patch::Integer, r_center_end::Integer,
    r_surround_start::Integer, r_surround_end::Integer,best_feats)
  oris_all = repeat(orientations;inner=n_realizations)
  stims_surr = make_stimuli_surrori(size_patch, r_center_end, r_surround_start,
    r_surround_end, oris_all, best_feats.sfreq,
    best_feats.sphase,[1.0,],[best_feats.cshift,] )
  # RF only by imposing a surround out of the image
  stims_nosurr = make_stimuli_surrori(size_patch, r_center_end, 120,
    121, oris_all, best_feats.sfreq,
    best_feats.sphase,[1.0,],[best_feats.cshift,] )[1:n_realizations,:]
  stims_nosurr[!,:sori] .= missing
  stims_nosurr[!,:idx] .+= nrow(stims_surr)
  return vcat(stims_surr,stims_nosurr)
end

function extract_mean_ff_surrori(gsm_obj,r_alpha; symmetrize=false)
  conv = GConvNorm(r_alpha)
  sampl = gsm_obj.samples
  views_sori = select(gsm_obj.views, [:idx,:sori])
  data = innerjoin(sampl,views_sori ; on=:idx)
  # add spike counts to dataframe
  transform!(data, :gs => ByRow(g->g_to_r(g,conv)) => :rs )
  # add mean spike counts and FFs
  transform!(data, [:rs => ByRow(mean) => :spk_mean ,
      :rs => ByRow(fanofactor) => :spk_ff] )
  # normalize by response without surround (a single scalar)
  spk_mean_nosurr = mean(@where(data,ismissing.(:sori)).spk_mean)
  transform!(data, :spk_mean => (spkmeans -> spkmeans./spk_mean_nosurr) =>
    :spk_mean_rel)
  # get rid of missing surround part
  data = @where(data,.!ismissing.(:sori))
  # now average over noise_realizations, for each surround orientation
  dat_bysori = groupby(data,:sori;sort=true)
  (soris,r,r_dup,r_ddown,ff,ff_dup,ff_ddown) = (Float64[] for i in 1:7)
  for dat in dat_bysori
    push!(soris,dat.sori[1])
    # mean over the means, geometric means over FF, across images
    bootmeans = mean_boot(dat.spk_mean_rel)
    bootffs = geomean_boot(dat.spk_ff)
    push!(r,bootmeans.mean)
    push!(r_dup,bootmeans.mean_dup)
    push!(r_ddown,bootmeans.mean_ddown)

    push!(ff,bootffs.geomean)
    push!(ff_dup,bootffs.geomean_dup)
    push!(ff_ddown,bootffs.geomean_ddown)
  end
  if symmetrize
    @info "Output will be symmetric"
    symmetrize!.([r,r_dup,r_ddown,ff,ff_dup,ff_ddown] , Ref(soris))
  end
  return (soris, r,r_dup,r_ddown,ff,ff_dup,ff_ddown)
end

function symmetrize!(y::AbstractVector,x::AbstractVector)
  xabs = abs.(x)
  for ax in unique(xabs)
    idxs = findall(xabs .== ax)
    if length(idxs)>1
      y_avg = mean(y[idxs])
      y[idxs].=y_avg
    end
  end
  return nothing
end

## Below , the part used for data analysis


include("data_analysis_base.jl")
include("data_analysis_crcns_pvc8.jl")
include("data_analysis_filters.jl")





end # module
