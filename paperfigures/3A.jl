using FestaetalLib; const F=FestaetalLib
using Plots, NamedColors
using Serialization
using Dates ; date2str() = Dates.format(now(),"yyyymmdd")
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
   F.save_gsm_object(gsm_obj, areasumm_stimuli_images)
 else
   @info "Now reading the GSM trained model from $path_gsm_file"
   gsm_dict = open(deserialize,path_gsm_file,"r")
   const size_patch = gsm_dict["size_patch"]
   gsm_obj = gsm_dict["gsm_obj"]
   areasumm_stimuli_images=gsm_dict["exp_areasumm_images"]
end

## Input parameters

# spatial frequency and phase selected as the most responsive
best_feats = F.get_best_features(gsm_obj.bank,size_patch)
# surround orientations (in degrees)
const orientations = range(-90,90;length=7) |> collect
# noise realizations for each orientation
const n_realizations = 50
# sizes (in pixels)
const r_center_end = 8
const r_surround_start = 23
const r_surround_end = 60

# generates the stimuli, the noise is added later, directly in the
# space of filter outputs.
surrori_stimuli = F.make_surrori_stimuli(orientations,n_realizations,
       size_patch, r_center_end, r_surround_start, r_surround_end,best_feats)
# the dataframe has 3 relevant colums
# stimulus index (idx) , surround orientation (sori), image (view)

## plot and save stimuli examples
myangles = [missing, orientations[[2,4,7]]...]
myviews = map(myangles) do ang
  if ismissing(ang)
    @where(surrori_stimuli, ismissing.(:sori)).view[1]
  else
    @where(surrori_stimuli, .! ismissing.(:sori),:sori .==ang ).view[1]
  end
end
plt = showbw_many("Example of input for surround orientation tuning", myviews...)

mkpath(F.dir_plots)
savname = joinpath(F.dir_plots,date2str()*"_surround_orientation_gsm_input.png")
savefig(plt,savname)
@info "Some of the stimuli used for Fig. 3A have been saved in $savname"

# perform Hamiltonian MC sampling of the stimuli
const nsamples = 30 # by default there are 4 chains working in parallel so the total is x4
F.sample_posterior_views!(gsm_obj,nsamples,surrori_stimuli;addnoise=true)

##
# extract the samples of the hidden variable, covert them into a "spike count"
# normaizes spike counts by RF response
# compute rate and FF with confidence intervals

const r_alpha = 40.
(soris, r,r_dup,r_ddown,ff,ff_dup,ff_ddown) =
  F.extract_mean_ff_surrori(gsm_obj,r_alpha;symmetrize=true)

# plot (I use two panels for more clarity)
plt = let plt1=plot(
    soris, r ; ribbon=(r_dup,r_ddown) , linewidth = 3,
      color=colorant"forestgreen",
      xlabel = "surround orientation",
      ylabel = "normalized avg. spike-count" ,
      leg=false)
    plt2 = plot( soris, ff ; ribbon=(ff_dup,ff_ddown) , linewidth = 3,
      color=colorant"darkblue",
    xlabel = "surround orientation", ylabel = "Fano factor" ,
    leg=false)
    plot(plt1,plt2; layout=(1,2))
end


# save it locally
mkpath(F.dir_plots)
figname = date2str()*"_main_3A.png"
fignamefull = joinpath(F.dir_plots,figname)
savefig(plt,fignamefull)
@info "Figure saved as $(fignamefull) . All done!"
