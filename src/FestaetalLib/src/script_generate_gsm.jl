
@info "Including additional scripts..."

using Revise
using Dates ; date2str() = Dates.format(now(),"yyyymmdd")
using GSMLibs ; const G=GSMLibs
using Plots, NamedColors
using Statistics, StatsBase , LinearAlgebra
using Images
using MAT
# plotting utility
function showbw(mat::AbstractMatrix,_title="")
  heatmap(mat, color=:grays,ratio=1,cbar=false,axis=false,title=_title)
end


##


# wrapper of lib function in GSMLibs/src/naturalRead.jl
# reads and stores a large set natural images from the specified folder
function read_natural_images(dir_natural_images ; verbose=true)
  @assert isdir(dir_natural_images) "$dir_natural_images not found!"
  @info "reading and processing natural images for training"
  img_test = Images.load(G.get_file_paths(dir_natural_images,".jpg")[1])
  img_size= minimum(size(img_test)) - 5
  all_images =  G.read_natural_images(dir_natural_images,img_size,G.StandardRegu() ;
    verbose=verbose)
  @info "natural images read!"
  return all_images
end

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


# source code also in GSMLib/steerPyramids.jl
function show_filter_bank(pars,framesize)
  ncent = pars.ncenter
  nsurr = pars.nsurround
  bank=make_filter_bank(pars)
  img1 = G.invert_test(framesize,bank;idx_bank=[1,(ncent+1:ncent+nsurr)...])
  img2 = G.invert_test(framesize,bank;idx_bank=[1,(ncent+1:ncent+nsurr)...],real_part=false)
  padval = maximum(img1)
  imgboth = hcat(img1, fill(padval, framesize,round(Int,0.1framesize)),img2 )
  showbw(imgboth , "GSM filter bank (+ and - phase)") # function defined above
end

# source code in GSMLibs/src/naturalRead.jl (cuts random patches from full imgs)
# and in  GSMLib/steerPyramids.jl (applies pyramid to the patch)
function  compute_filter_outputs(bank, nat_imgs, size_patch , nsampl)
  @info """
  Now apply the steerable pyramid filters to $nsampl patches of size $size_patch
  Warning : for a high number of patches (e.g. 10,000) this process might take
  up to 15-20 minutes.
  """
  natpatches = G.sampling_tiles(nsampl, nat_imgs, size_patch)
  ret = G.apply_bank( natpatches, bank, G.Xstd(2.0),false)
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
  ret = G.GSM_from_filter_outputs(x_values, bank, size_patch, noise_level, nsamples ;
        mixer = G.RayleighMixer(1.0))
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
  G.GSM_Model(gsm,size_patch,scale,distance,ncenter,
      nsurround,noise_level,filter_bank,x_values)
end
