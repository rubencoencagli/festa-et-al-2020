using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Dates ; date2str() = Dates.format(now(),"yyyymmdd")
using Statistics

# function to plot a black/white image
function showbw(mat::AbstractMatrix,_title="")
  heatmap(mat, color=:grays,ratio=1,cbar=false,axis=false,title=_title)
end

# build the GSM or not ?
path_gsm_file = F.read_gsm_path()
buildGSM = F.check_rebuild(ARGS) || isnothing(path_gsm_file)

## Build or read the GSM model

if buildGSM
  @info "The GSM model will be initialized and trained, this might take some time"

  # size of square image patch considered, in pixel
  const size_patch = 160

  # read natural images, convert to BW, normalize
  nat_image_patches,experiment_images = F.read_natural_images(size_patch)

  # this generates the filter bank
  # parameters used
  filter_bank_pars = (
        # 1 center filter (+- phases) , 8 surround filters
        ncenter = 1,
        nsurround = 8,
        # size of filters
        scale = 0.2,
        # distance center_surround
        distance = 0.6)

   filter_bank = F.make_filter_bank(filter_bank_pars)

   # visualize the filter bank in a plot
   to_plot = F.show_filter_bank(filter_bank_pars,size_patch)
   plt = showbw(to_plot , "GSM filter bank (+ and - phase)") # function defined above
   # save the plot
   mkpath(F.dir_plots)
   savname = joinpath(F.dir_plots,date2str()*"filter_bank.png")
   savefig(plt,savname)
   @info "A representation of the bank of linear filter has been saved in $savname"

   # apply the filter bank to natural images, to get the output vector
   const nsampl = 10_000 #10_000 # how many samples to train on ? (suggested: 10_000 )
   x_values = F.compute_filter_outputs(filter_bank, nat_image_patches, size_patch , nsampl)

   #  noise scaling such that the trace of the covariance of the noise is
   #  noise_scale_factor * trace of covariange of the g
   const noise_scale_factor = 0.1

   # use the outputs on natural statistics to train the GSM
   gsm = F.train_gsm_model(x_values,filter_bank,size_patch,noise_scale_factor,nsampl)

   # this is an object that stores internally GMS model, filter bank stimuli  and samples (initialized as empty)
   gsm_obj = F.build_gsm_object(gsm,noise_scale_factor,
      filter_bank_pars,filter_bank,size_patch,x_values)
   # save the GSM model, the filter bank and the patch size
   F.save_gsm_object(gsm_obj, experiment_images)
 else
   @info "Now reading the GSM trained model from $path_gsm_file"
   gsm_dict = open(deserialize,path_gsm_file,"r")
   const size_patch = gsm_dict["size_patch"]
   gsm_obj = gsm_dict["gsm_obj"]
end

## Make the input

# read and preprocess natural images
nat_images, _ = F.read_natural_images(size_patch)
# cut them into small patches
const n_input_patches = 500
natural_images_patches = F.cut_natural_patches(nat_images,n_input_patches,size_patch)

# you can visualize examples here. This plot will not be saved
showbw(natural_images_patches[10] , "stimulus example")

## Sample

# perform Hamiltonian MC sampling
const nsamples = 30 # by default there are 4 chains working in parallel so the total is x4
F.sample_posterior_imgs!(gsm_obj,nsamples,natural_images_patches)


## Extract response, plot

# extract the samples of the latent g, covert them into a "spike count"
const r_alpha = 20.
rs = F.extract_neural_response(gsm_obj,r_alpha)

# plot mean and variance, which corresponds to figure 1D
plt = let _pltlims = (1E-0,1E3)
 plt = scatter(mean.(rs),var.(rs) ; ratio = 1 , xscale=:log10, yscale=:log10,
    xlims=_pltlims, ylims=_pltlims, label="GSM model" , color=:black, markersize=5,
    opacity=0.5)
 plot!(plt, identity, [_pltlims...], label="Poisson process" , linestyle=:dash,
  color=:black, linewidht=5)
 plot!(plt, xlabel="Mean (a.u)" , ylabel="Variance (a.u.)", title="Mean Vs Variance in GSM response to natural images (main text, D1)",leg=false)
 end

# save it locally
mkpath(F.dir_plots)
figname = date2str()*"_main_1D.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
