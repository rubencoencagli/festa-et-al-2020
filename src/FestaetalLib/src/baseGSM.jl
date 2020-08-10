
#=
GSM model constructor and utilities

=#

abstract type MixerType end
struct RayleighMixer <: MixerType
    alpha
end

"""
    get_covariance_g(Sx,Sn,mixer::RayleighMixer)

Covariance matrix for GSM with Rayleigh mixer with moment matching.
"""
function get_covariance_g(Sx,Sn,mixer::RayleighMixer)
    scaling =  2.0mixer.alpha*mixer.alpha
    return @. (Sx-Sn)/scaling
end
"""
    get_covariance_g(Sx,mixer::RayleighMixer)

Covariance matrix for GSM with Rayleigh mixer with moment matching.
Assumes that the provided data is noise-free.
"""
function get_covariance_g(Sx,mixer::RayleighMixer)
    return Sx ./ (2.0mixer.alpha*mixer.alpha)
end


struct GSM{Mx}
    covariance_matrix::AbstractMatrix
    covariance_matrix_noise::AbstractMatrix
    mixer::Mx
end
n_dims(g::GSM) = size(g.covariance_matrix,1)

function GSM_from_data(data,
    covariance_matrix_noise::AbstractMatrix,
    mixer::MixerType)
  Sigma_x = cov(data;dims=2)
  Sigma_g = get_covariance_g(Sigma_x,covariance_matrix_noise,mixer)
  @assert isposdef(Sigma_g) "not positive definite! (too much noise?)"
  return GSM(Sigma_g,covariance_matrix_noise,mixer)
end

function GSM_from_filter_outputs(xs, bank, size_patch, noise_level, nsamples ;
      mixer = RayleighMixer(1.0))
  covariance_noise = get_covariance_noise(nsamples,size_patch,bank ; verbose=false)
  Sigma_x = cov(xs;dims=2)
  covariance_g = get_covariance_g(Sigma_x, mixer)
  scale_covariance_noise!(covariance_noise,covariance_g,noise_level)
  return GSM(covariance_g,covariance_noise,mixer)
end

"""
    get_covariance_noise(n_sampl, patch_size, pyrbank::OnePyrBank ;
        verbose=false)
          -> covariance_noise::Matrix

Computes the *unscaled* noise covariance for the filters from random image patches.
"""
function get_covariance_noise(n_sampl, patch_size, pyrbank::OnePyrBank ;
    verbose=false)
  img_size = round(Integer,patch_size*1.2)
  img_noise = make_noise_images(img_size,n_sampl)
  x_noise = apply_bank(n_sampl,img_noise,patch_size,pyrbank,XNothing(),verbose)
  return cov(x_noise;dims=2)
end

"""
    scale_covariance_noise!(covariance_nose, covariance_g , r )

Scales the covariance noise matrix so that `tr(covariance_noise) ==  r*tr(covariance_g)`
"""
function scale_covariance_noise!(covariance_noise, covariance_g , r )
  scal = r*tr(covariance_g)/tr(covariance_noise)
  lmul!(scal,covariance_noise)
  return nothing
end


"""
    make_rand_cov_mat( dims , diag_val::Float64 ; k_dims=5)

Returns a random covariance matrix that is positive definite
and has off-diagonal elements.
# Arguments
- `d`: dimensions
- `diag_val`: scaling of the diagonal
- `k-dims`: tuning of off diagonal elements
"""
function make_rand_cov_mat( dims::Integer , diag_val::Real , (k_dims::Integer)=5)
  W = randn(dims,k_dims)
  S = W*W'+ Diagonal(rand(dims))
  temp_diag = Diagonal(inv.(sqrt.(diag(S))))
  S = temp_diag * S * temp_diag
  S .*= diag_val
  # perfectly symmetric
  for i in 1:dims
    for j in i+1:dims
      S[i,j]=S[j,i]
    end
  end
  S
end

# """
#     function GSM(natural_images::Vector, bank::OnePyrBank , noise_scal, ray_alpha ;
#       (xpost::XPostProcess)=Xstd(1.0) ,
#       (xsamples::I)=10_000,
#       (noise_samples::I)=1500,
#       (sample_size::I) = 50) where I<:Integer
#
# Generates a GSM model starting from an array of natural images. The images are cut in
# smaller patches and sampled to obtain the filter outputs, that are then used to
# infer the covariance matrix by maximul likelihood.
# The parameter `ray_alpha` defines the mixer prior distribution. `noise_scal` defines
# how large is the noise covariance matrix (becomes the std of the data used for
# the noise cov matrix estimate)
# """
# # Build GSM from natural images, uses a lot of elements from nturalRead and steerPyramids
# function GSM(natural_images::Vector, bank::OnePyrBank , noise_scal, ray_alpha ;
#     (xpost::XPostProcess)=Xstd(1.0) ,
#     (xsamples::I)=10_000,
#     (noise_samples::I)=1500,
#     (sample_size::I) = 50,
#     verbose=false) where I<:Integer
#   # println("Sampling random noise to infer the noise covariance matrix...")
#   noise_cov = get_covariance_noise(noise_samples,sample_size,bank;
#           verbose=verbose)
#   # println("Done! Sampling natural images, to build the convariance matrix...")
#   x_nat =  apply_bank(xsamples,natural_images,sample_size,bank,xpost,verbose)
#   # println("Done! Building the GSM model")
#   GSM_from_data(x_nat,noise_cov,RayleighMixer(ray_alpha))
# end
#

"""
    function get_samples(gsm::GSM{M}, nsamples) where M<:RayleighMixer

Generates `nsamples` random samples from the model. Returns a `NamedTuple` with
fields `( gs , xs , mixers, noise )`.  where `xs = mixers * gs + noise`
"""
function get_samples(gsm::GSM{M}, nsamples) where M<:RayleighMixer
  Gcov=gsm.covariance_matrix
  Ncov = gsm.covariance_matrix_noise
  d1=MvNormal(Gcov)
  G=rand(d1,nsamples)
  d2=MvNormal(Ncov)
  N=rand(d2,nsamples)
  d3=Rayleigh(gsm.mixer.alpha)
  R1=rand(d3,nsamples)
  X = broadcast(*,transpose(R1),G) .+ N
  (gs=G,xs=X,mixers=R1,noise=N)
end


"""
    add_xnoise(gsm::GSM,xs::AbstractMatrix)

Generates additive noise following the generative model specified by `gsm`
and adds it to `xs` (creating a fresh matrix)
"""
function add_xnoise(gsm::GSM,xs::AbstractArray)
  gamma_n = gsm.covariance_matrix_noise
  if tr(gamma_n) < 1E-4
    return copy(xs)
  end
  to_add = rand(MultivariateNormal(gamma_n),size(xs,2))
  to_add .+= xs
end


"""
    get_lambda_etc(gsm::GSM,xs::AbstractMatrix)

given a `gsm` model and inputs `xs` extracts what is most relevant for the
g posterior: the intensity of center stimulus, the intensity of the rest ,
and the matching Vs the model covariance.
"""
function get_lambda_etc(gsm::GSM,xs::AbstractMatrix)
  xcent = mapslices(xs;dims=1) do x
    sqrt(x[1]^2+x[2]^2)
  end
  xsurr = mapslices(xs;dims=1) do x
    2*mean( x[3:end].^2 )
  end
  sigma_g = gsm.covariance_matrix
  lambda = mapslices( x-> sqrt(dot(x,sigma_g\x)),xs;dims=1)
   (xcent=xcent,xsurr=xsurr,lambd=lambda)
end


"""
  mutable struct GSM_Model

Contains a trained GSM model, including its parameters and filter banks.
It also contains additional stimuli (e.g. area summation stimuli)
and the sampled posterior

# Elements
  + `scale` :  filter size, smaller number means bigger size, between 0 and 1
    (more like 0.16 to 0.8)
  + `distance` : distance between center and surround filters, between 0 and 1
  + `nsurround` : number of surround filters
  + `bank::OnePyrBank ` :   bank of steerable pyramid linear filters
  + `noise_level` : noise on the GSM model
  + `x_natural` : matrix that contains the filter outputs for a large
  number of natural patches, used to generate the GSM model
  + `gsm::GSM` : gsm model
  + `contrast` : scaling of tested views
  + `shift` : misalignment of views from center
  + `views::DataFrame`: stimuli
  + `samples::DataFrame` : posterior of features given the views. Match the views
    dataframe by the `idx` column
"""
mutable struct GSM_Model
  scale
  distance
  nsurround::Integer # 4 or 8
  bank::OnePyrBank
  #
  noise_level
  x_natural::Matrix
  gsm::GSM
  #
  contrast
  shift
  #
  views::DataFrame  # collects all the features used , has idx element
  samples::DataFrame # for each view, vs and gs samples, idx matches the above
end

  # this constructor does not sample over Stan, but covers the rest
function GSM_Model(nsamples,size_patch,scale,distance,nsurround,contrast,
                    shift,noise_level,natural_images,all_views;
                    n_center=4 , views_bestfrequency=true)
  bank=OnePyrBank(scale,distance,nsurround;ncent=n_center)
  covariance_noise = get_covariance_noise(nsamples,size_patch,bank ; verbose=false)
  natpatches = sampling_tiles(nsamples, natural_images, size_patch)
  x_natural = apply_bank( natpatches, bank, Xstd(2.0),false)
  Sigma_x = cov(x_natural;dims=2)
  mixer = RayleighMixer(1.)
  covariance_g = get_covariance_g(Sigma_x, mixer)
  scale_covariance_noise!(covariance_noise,covariance_g,noise_level)
  gsm = GSM(covariance_g,covariance_noise,mixer)
  # extract the views
  _views = views_bestfrequency ? get_views_areasumm(bank,contrast,shift,all_views) :
    all_views
  samples = DataFrame()
  # create the struct
  return GSM_Model(scale,distance,nsurround,bank,noise_level,x_natural,gsm,contrast,shift,
    _views,samples)
end

function GSM_Model(gsm::GSM,size_patch,scale,distance,ncenter,
    nsurround,noise_level,bank,x_natural)
  contrast = 1.0
  shift = 0.0
  # extract the views
  views = DataFrame()
  samples = DataFrame()
  # create the struct
  return GSM_Model(scale,distance,nsurround,bank,noise_level,x_natural,gsm,contrast,shift,
    views,samples)
end



function parameters(gm::GSM_Model)
  ( scale=gm.scale,
  surround_distance = gm.distance,
  nsurround = gm.nsurround ,
  noise_level = gm.noise_level,
  stimuli_contrast = gm.contrast,
  stimuli_shift = gm.shift)
end




"""
      apply_bank(img::AbstractMatrix,gsm_mdel::GSM_Model ; (i0j0_all::Vector)=[(-1,-1)] )

Applies the filter bank of gsm_model to a single image, the bank is applied once at each
reference point indicated (reference w.r.t normal matrix coordinates).  The default is to
apply the bank once, in the center of the image (when coordinats are <0 , the matrix is just
centered in the middle)
"""
function apply_bank(img::AbstractMatrix,gsm_model::GSM_Model ; i0j0_all=nothing )
  bank = gsm_model.bank
  if isnothing(i0j0_all)
    return apply_bank(img,bank)
  else
   return apply_bank(img,bank ; i0j0_all=i0j0_all)
 end
end


"""
    function frequency_best(bank,stims)

Selects the best spatial frequency for a grating stimulus for the specified bank of
filters
"""
function frequency_best(bank,stims)
  sz = maximum(stims.size)
  dfs = @where(stims, :contrast .== 1 , :shift .== 0 , :size .== sz)
  tiles = [ (s,[(-1,-1)] ) for s in dfs.view]
  xs = apply_bank(tiles, bank, Xstd(1.0),false)
  norms = map(norm,eachcol(xs))
  dfs.frequency[argmax(norms)]
end

""""
function get_views_areasumm(bank,contrast,shift,all_views)

creates a dataframe that contains data summation stimuli, starting from a global one.
"""
function get_views_areasumm(bank,contrast,shift,all_views)
  freq = frequency_best(bank,all_views)
  df = @where(all_views, :contrast .== contrast, :shift .== shift,
                  :frequency .== freq)
  @assert nrow(df) > 0 " Problem with stimuli... wrong parameters?"
  sort!(df,:size)
#  rename!(df, (:stim_size => :size, :stim_view => :view))
  df[:idx] = collect(1:nrow(df))
  return df[:,[:idx,:size,:view]]
end


"""
Abstract type that indicates the conversion methods
from g samples to spike counts.
"""
abstract type  GConversion end

"""
Norm of first two elements of g, scaled
"""
struct GConvNorm <: GConversion
  a::Float64
end
GConvNorm() = GConvNorm(1.0)

function g_to_r(g::AbstractVector{T}, conv::GConvNorm) where T<:Real
  conv.a*sqrt(g[1]^2 + g[2]^2)
end
function g_to_r(gs::AbstractMatrix{T}, conv::GConvNorm) where T<:Real
  map(g->g_to_r(g,conv), eachslice(gs;dims=2))
end
# when 3d array, first dimension is filter dim
function g_to_r(gs::Array{Float64,3},conv::GConvNorm)
  # Julia has an inconsistency over here...
  dropdims(mapslices(g->g_to_r(g,conv),gs;dims=1);dims=1)
end


"""
    GConvMeanStd <: GConversion

converts the gs by maximizing the log likelihood that the data is gaussian
with the means and the variances indicated by its internal parameters.
"""
struct GConvMeanStd <: GConversion
  means::Vector{Float64}
  stds::Vector{Float64}
end
Base.length(g::GConvMeanStd) = length(g.means)

# assuming that data has first index for stimulus, second index for trials
function GConvMeanStd(data)
  means = mean(data;dims=2)
  stds = std(data;dims=2, mean=means)
  GConvMeanStd(vec.((means,stds))...)
end
function _best_scaling(mu::T,muhat::T,sigma::T,sigmahat::T) where T <: AbstractVector
  n=length(mu)
  invsigmasq=@. inv(sigma^2)
  bterm= mean(  @. (mu * muhat * invsigmasq))
  cterm = @. (sigmahat^2 + muhat ^2)*invsigmasq
  return 0.5(sqrt(bterm.^2 + 4.0mean(cterm) ) - bterm)
end
function g_to_r(gs::Array{T,3},conv::GConvMeanStd) where T<:Real
  r_ret = g_to_r(gs,GConvNorm(1.0))
  muhat,sigmahat = conv.means, conv.stds
  # first index stimulus, second index samples
  mu = dropdims(mean(r_ret;dims=2) ; dims=2)
  sigma = dropdims( std(r_ret;dims=2,mean=mu);dims=2)
  r_ret .*= _best_scaling(mu,muhat,sigma,sigmahat)
  return r_ret
end
