
@info "Including additional scripts..."

using Revise
using Dates ; date2str() = Dates.format(now(),"yyyymmdd")
using GSMLibs ; const G=GSMLibs
using Plots, NamedColors
using Statistics, StatsBase,Bootstrap
using JLD
using DataFrames, DataFramesMeta
using Images

# plotting utility

function showbw(mat::AbstractMatrix,_title="")
  heatmap(mat, color=:grays,ratio=1,cbar=false,axis=false,title=_title)
end
function showbw_many(title::String,mats...)
  mats = [mats...] # convert tuple to array
  padval=maximum(maximum,mats)
  framesize = size(mats[1],1)
  padmat = fill(padval, framesize,round(Int,0.1framesize))
  padmats = repeat([padmat,],inner=length(mats))
  matspads = permutedims(hcat(mats,padmats))[:]
  imgrow = hcat(matspads[1:end-1]...)
  showbw(imgrow , title)
end

##

function set_stan_dir(dir)
  dir =something(dir,abspath(@__DIR__,"..","libraries","cmd-stan"))
  @assert isdir(dir) "Stan directory $dir not found!"
  ndir = length(readdir(dir))
  @assert ndir > 2 "Stan directory $dir seems to be empty. Please dowload the dependency"
  stanfound = isfile(joinpath(dir,"bin","stanc")) ||
    isfile(joinpath(dir,"bin","stanc.exe"))
  @assert stanfound "Stan executable not found. Was Stan built correctly?"
  G.set_stan_folder(dir)
  return nothing
end


# wrapper of lib function in GSMLibs/src/naturalRead.jl
function read_natural_images(dir_natural_images ; verbose=true)
  @assert isdir(dir_natural_images) "$dir_natural_images not found!"
  @info "reading and processing natural images for training"
  img_test = load(G.get_file_paths(dir_natural_images,".jpg")[1])
  img_size= minimum(size(img_test)) - 5
  all_images =  G.read_natural_images(dir_natural_images,img_size,G.StandardRegu() ;
    verbose=verbose)
  @info "natural images read!"
  return all_images
end

function read_trained_gsm(file_gsm)
  @assert isfile(file_gsm) "File $file_gsm not found! Please train a gsm using the generate_gsm.jl script!"
  f=jldopen(file_gsm,"r")
  gsm_obj = read(f,"gsm_obj")
  noise_scale_factor = read(f,"noise_scale_factor")
  size_patch = read(f,"size_patch")
  exp_areasumm_img = read(f,"exp_areasumm_images")
  close(f)
  return (gsm_obj,noise_scale_factor,size_patch,exp_areasumm_img)
end



# wrapper of a function in GSMLibs/src/naturalRead.jl
function cut_natural_patches(natural_images,nsamples,size_patch)
  tiles = G.sampling_tiles(nsamples, natural_images, size_patch)
  return map(first,tiles)
end

function store_image_inputs!(gsm_obj,imgs)
  views = DataFrame(
    idx = collect(1:length(imgs)),
    view = imgs)
  gsm_obj.views = views
end

# wrapper of a function in GSMLibs/src/stanSampl.jl
function sample_posterior!(gsm_obj,n_sampl;addnoise=false)
  @info "Now sampling the posterior of the hidden features using stan"
  G.sample_posterior(n_sampl,gsm_obj;addnoise=addnoise)
  @info "sampling ended successfully, results stored in gsm model structure"
  return nothing
end

function extract_neural_response_simple(gsm_obj,alph)
  gs = gsm_obj.samples.gs
  conv = G.GConvNorm(alph)
  return map(g->  G.g_to_r(g,conv),gs)
end

# reads the large size images and cut them at the right size
function make_stimuli_natural_areasumm(fileimg,framesize)
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


function make_areasumm_stimulus(imgs_input,model_sizes,
    grey_val,shift_val=0.0)
  x_img = shift_val
  y_img = shift_val
  nsizes =  length(model_sizes)
  nimg = length(imgs_input)
  nviews = nsizes*nimg
  idx = collect(1:nviews)
  sizes = repeat(model_sizes ; outer=nimg)
  images = repeat(1:nimg ; inner=nsizes)
  views = [deepcopy(imgs_input[i]) for i in images]
  views = map(zip(views,sizes)) do (v,r)
      G.cut_outside!(v,r;zero=grey_val, x_rel=x_img,y_rel=y_img)
  end
  return DataFrame(
    idx=idx, img_idx=images,
    size=sizes , view = views )
end

function extract_mean_ff_bysize(gsm_obj,r_alpha)
  conv = G.GConvNorm(r_alpha)
  sampl = gsm_obj.samples
  views_size = select(gsm_obj.views, [:idx,:size,:img_idx])
  data = innerjoin(sampl,views_size ; on=:idx)
  # add spike counts to dataframe
  transform!(data, :gs => ByRow(g->  G.g_to_r(g,conv)) => :rs )
  # add mean spike counts and FFs
  transform!(data, [:rs => ByRow(mean)  => :spk_mean ,
      :rs => ByRow(rs -> var(rs)/mean(rs))   => :spk_ff] )
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

##
# Bootstrap functions

spitdeltaerror(x, down,up) = (x-down, up-x)
"""
  mean_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="mean") where T <: Real
  -> NamedTuple , fields:  (mean, mean_cidown,mean_ciup, mean_down,mean_dup)
"""
function mean_boot(spk::AbstractVector{T} ;
          nrun = 5_000 , conf = 0.95, prefix="mean") where T <: Real
    _meanbs = bootstrap(Statistics.mean, spk, BasicSampling(nrun))
    mean, mean_cidown, mean_ciup = confint(_meanbs, BCaConfInt(conf))[1]
    mean_ddown,mean_dup = spitdeltaerror.(mean, mean_cidown, mean_ciup)
    retnames = Symbol.( [prefix*nm for nm in ("","_cidown","_ciup","_ddown","_dup")] )
    return (; zip(retnames, [mean,mean_cidown,mean_ciup,mean_ddown,mean_dup])...)
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


## Surround polarization scripts

function make_stimuli_disk(size_patch, sizes, stim_ori, freqs, phases,
      stim_contrast, center_shifts ; contrast_base=3.0)
  df_ret = DataFrame(idx=Int64[],
      cshift=Float64[],
      sfreq=Float64[],
      sphase=Float64[],
      contrast=Float64[],
      size = Int64[],
      ori = Float64[],
      view = Matrix{Float64}[] )
  idx = 0
  for sh in center_shifts,
      freq in freqs,
      phase in phases,
      contr in stim_contrast ,
      ori in stim_ori,
      sz in sizes # do
    idx += 1
    grat_pars = G.GratingParams(size=size_patch, ori=ori,frequency=freq,
                contrast = contrast_base, noise_std=0.0 , phase=phase )
    view = G.gratings_enlarging_center([sz],grat_pars;rel_center=(sh,sh))[1]
    view .*= contr
    push!(df_ret,[idx sh freq phase contr sz ori [view]])
  end
  df_ret
end

function make_stimuli_surrori(size_patch::Integer, rad_small, rad_large_start,
    rad_large_end , oris_surr, stim_freq, stim_phase,
    stim_contrast, center_shifts ; contrast_base=3.0)
  df_ret = DataFrame(idx=Int64[],
      cshift=Float64[],
      sfreq=Float64[],
      sphase=Float64[],
      contrast=Float64[],
      rad_small = Int64[],
      rad_large_start = Int64[],
      rad_large_end = Int64[],
      sori = Float64[],
      view = Matrix{Float64}[] )
  idx = 0
  for sh in center_shifts,
      contr in stim_contrast ,
      r_small in rad_small,
      r_large_st in rad_large_start ,
      r_large_en in rad_large_end ,
      ori in oris_surr # do ...
    idx += 1
    grat_pars = G.GratingParams(size=size_patch, ori=0.0,frequency=stim_freq,
                phase=stim_phase,
                contrast = contrast_base, noise_std=0.0)
    view = G.gratings_surround_polarization(r_small,r_large_st,
        r_large_en,ori,grat_pars ; rel_center=(sh,sh))[1]
    view .*= contr
    push!(df_ret,[idx sh stim_freq stim_phase contr r_small r_large_st r_large_en ori  [view]])
  end
  df_ret
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
  dfret1 = G.bank_best_response(filter_bank,stims,[:sfreq,:sphase];idxs=(1:2))
  best_sfreq = dfret1.sfreq
  best_sphase = dfret1.sphase
  sizes=[7,]
  sfreqs = [best_sfreq,]
  cshifts = range(-0.2,0.2;length=50)
  stims =  make_stimuli_disk(size_patch, sizes, oris, sfreqs, best_sphase,
      contrasts,cshifts; contrast_base=3.0)
  dfret2 = G.bank_best_response(filter_bank,stims,[:cshift]; idxs=(1:2))
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
  conv = G.GConvNorm(r_alpha)
  sampl = gsm_obj.samples
  views_sori = select(gsm_obj.views, [:idx,:sori])
  data = innerjoin(sampl,views_sori ; on=:idx)
  # add spike counts to dataframe
  transform!(data, :gs => ByRow(g->G.g_to_r(g,conv)) => :rs )
  # add mean spike counts and FFs
  transform!(data, [:rs => ByRow(mean) => :spk_mean ,
      :rs => ByRow(rs -> var(rs)/mean(rs)) => :spk_ff] )
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
