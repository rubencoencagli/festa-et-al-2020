using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Statistics
using DataFrames, DataFramesMeta

# function to plot one or more black/white images
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

# build the GSM or not ?
path_gsm_file = F.read_gsm_path()
buildGSM = F.check_rebuild(ARGS) || isnothing(path_gsm_file)

## Build or read the GSM model

if buildGSM
  @info "The GSM model will be initialized and trained, this might take some time"

  # size of square image patch considered, in pixel
  const size_patch = 160

  # read natural images, convert to BW, normalize
  nat_image_patches,areasumm_stimuli_images = F.read_natural_images(size_patch)

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
   savname = joinpath(F.dir_plots,date2str()*"_filter_bank.png")
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
   F.save_gsm_object(gsm_obj, areasumm_stimuli_images)
 else
   @info "Now reading the GSM trained model from $path_gsm_file"
   gsm_dict = open(deserialize,path_gsm_file,"r")
   const size_patch = gsm_dict["size_patch"]
   gsm_obj = gsm_dict["gsm_obj"]
   areasumm_stimuli_images=gsm_dict["exp_areasumm_images"]
end

##

# let's show some of the test images (not saved)
_ = let img_idxs=[7,8,11,12]
  showbw_many("Esample of raw input", areasumm_stimuli_images[img_idxs]...)
end

##

# stimulus sizes (in pixels, assuming a RF size of about 10 px)
const rf_size = 10.0
const sizes = [0.33,
           0.60,
           1.0,
           1.6,
           2.6,
           4.3,
           7.0 ] .* rf_size

const grey_value = 0.0
areasumm_stimuli = F.make_areasumm_stimulus(areasumm_stimuli_images,sizes,grey_value)
# the dataframe has 4 colums
# stimulus index (idx) , image index (img_idx) , disk size (size), image (view)

## save an example of the stimuli
img_idx_plot = 3
sizes_idx = [1,3,5,7]
mysizes = sizes[sizes_idx]
myviews=@where(areasumm_stimuli, :img_idx .== img_idx_plot , in.(:size,Ref(mysizes))).view
#make the plot
plt=showbw_many("Example of input", myviews...)
# save the plot
mkpath(F.dir_plots)
savname = joinpath(F.dir_plots,date2str()*"_areasumm_gsm_input.png")
savefig(plt,savname)
@info "Some of the stimuli used for Fig. 2D have been saved in $savname"

##

# perform Hamiltonian MC sampling of the stimuli
const nsamples = 30 # by default there are 4 chains working in parallel so the total is x4
F.sample_posterior_views!(gsm_obj,nsamples,areasumm_stimuli; addnoise=false)

# extract the samples of the hidden variable, covert them into a "spike count"
# normaizes spike counts by RF response
# compute rate and FF with confidence intervals

const r_alpha = 15.0
(sizes, r,r_dup,r_ddown,ff,ff_dup,ff_ddown) = F.extract_mean_ff_bysize(gsm_obj,r_alpha)


# plot (I use two panels for more clarity)
plt = let plt1=plot(
    sizes ./ rf_size, r ; ribbon=(r_dup,r_ddown) , linewidth = 3,
      color=colorant"forestgreen",
      xlabel = "stim size (relative to RF)",
      ylabel = "mean normalized spike-count" ,xscale=:log10,
      leg=false)
    plt2 = plot( sizes ./ rf_size, ff ; ribbon=(ff_dup,ff_ddown) , linewidth = 3,
      color=colorant"darkblue",
    xlabel = "stim size (relative to RF)", ylabel = "Fano factor" ,xscale=:log10,
    leg=false)
    plot(plt1,plt2; layout=(1,2))
end



# save it locally
mkpath(F.dir_plots)
figname = date2str()*"_main_2D.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
