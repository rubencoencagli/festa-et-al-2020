#=
Produces image patches made of gratings
=#


# this controls phase orientation,global patch size.
"""
        struct GratingParams
          size  :: Int32
          frequency:: Float64
          ori :: Float64
          phase   :: Float64
          contrast  :: Float64
          noise_std::Float64
        end
Frequency is in pixel, stimulus goes from -contrast/2  to + constrast/2
 (for zero noise!)  , orientation is in degrees
"""
mutable struct GratingParams
  size  :: Int32    # image size
  frequency:: Float64  # frequency (order of  pixel)
  ori :: Float64  # orientation
  phase   :: Float64  # phase
  contrast  :: Float64  # normalization factor (white level)
  noise_std::Float64 # standard deviation of white noise. If is <= 0, no noise
end
"""
Simplified constructor for Gabor parameters
The defaults are specified here
"""
function GratingParams(;
    size=50, frequency=4,
    ori=0, contrast=80.0,phase=0.0,
    noise_std=5.0)
  GratingParams(size,frequency,ori,phase,contrast,noise_std)
end
import Base.copy
function copy(gabp::GratingParams)
    vals=[getfield(gabp,n) for n in fieldnames(GratingParams)]
    GratingParams(vals...)
end

function (grating_params::GratingParams)(; with_noise = true )
  P=grating_params
  ori_r = P.ori*pi/180
  xp(x,y) = x*cos(ori_r) + y*sin(ori_r)
  RG(x,y) = cos(2*pi/P.frequency * xp(x,y) + P.phase)
  sizef=Float64(P.size)
  myx=LinRange(-sizef/2,sizef/2,P.size)
  myy=LinRange(-sizef/2,sizef/2,P.size)
  # RG goes from -1 to +1
  norm_fact=P.contrast/2.0
  #matrix rows are the vertical position of the figure
  out = broadcast( RG, myx',myy)
  out .*= norm_fact
  if with_noise && P.noise_std > 0.0
     out .+=  rand(Normal(0.0,P.noise_std),size(out)...)
 end
 out
end

test=GratingParams(ori=10,frequency=5.5)()
function add_noise!(img::AbstractMatrix,gt::GratingParams)
    _std=gt.noise_std
    _std  > 0.0 && ( img .+= rand(Normal(0.0,_std),size(img)...) )
    img
end

function grating_transition(field,transition;gratp=GratingParams(),with_noise=true)
    n_out=length(transition)
    map(enumerate(transition)) do (i,tr)
        setfield!(gratp,field,tr)
        gratp(;with_noise=with_noise)
    end
end

"""
function gratings_surround_polarization( rad_small,rad_large_start,
        rad_large_end,
        oris_surr,
        grating_pars;
        rel_center=(0.,0.),relative_oris=true)

Center of fixed orientation, and rotating surround. Center and surround gratings
only differ in orientation.

"""
function gratings_surround_polarization(
        rad_small,rad_large_start,rad_large_end,
        oris_surr,
        grating_pars;rel_center=(0.,0.),relative_oris=true)
    _xr,_yr = rel_center
    oris_surr = relative_oris ? oris_surr .+ grating_pars.ori : oris_surr
    @assert rad_small <= rad_large_start
    gratc = grating_pars(with_noise=false)
    cut_outside!(gratc,rad_small;x_rel=_xr,y_rel=_yr)
    grats_background = grating_transition(:ori,oris_surr;
                            gratp=copy(grating_pars),with_noise=false)
    map(grats_background) do gratb
        cut_inside!(gratb,rad_large_start ;x_rel=_xr,y_rel=_yr)
        cut_outside!(gratb,rad_large_end;x_rel=_xr,y_rel=_yr)
        add_noise!(gratb+gratc,grating_pars)
    end
end

function gratings_enlarging_center(rads,grating_pars;
     rel_center::Tuple{Float64,Float64}=(0.,0.))
    gratc = grating_pars(;with_noise=false)
    map(rads) do rad
        grat=copy(gratc)
        cut_outside!(grat,rad;x_rel=rel_center[1],y_rel=rel_center[2])
        add_noise!(grat,grating_pars)
    end
end


function cut_inside!(img,rad;zero=0.0,x_rel=0.0,y_rel=0.0)
    @assert abs(x_rel) <=1.0 && abs(y_rel) <=1.0
    yc=floor(Int64, 0.5*(size(img,1)+1)*(1+y_rel))
    xc=floor(Int64,0.5*(size(img,2)+1)*(1+x_rel))
    imgc=centered(img,yc,xc)
    for ij in CartesianIndices(imgc)
        i,j=Tuple(ij)
        r = sqrt(i*i+j*j)
        if r < rad
            imgc[i,j]=zero
        end
    end
    img
end

function cut_outside!(img,rad;zero=0.0,x_rel=0.0,y_rel=0.0)
    @assert abs(x_rel) <=1.0 && abs(y_rel) <=1.0
    yc=floor(Int,div(size(img,1)+1,2)*(1+y_rel))
    xc=floor(Int, div(size(img,2)+1,2)*(1+x_rel))
    imgc=centered(img,yc,xc)
    for ij in CartesianIndices(imgc)
        i,j=Tuple(ij)
        r = sqrt(i*i+j*j)
        if r >= rad
            imgc[i,j]=zero
        end
    end
    return img
end

function pixel_shift(img,k=1,zero=0.0)
  if rand(Bool) # cols
    nr = size(img,1)
    filler = fill(zero,nr)
    rand(Bool) ? (return hcat(img[:,1+k:end],filler)) : return hcat(filler,img[:,1:end-k])
  else
      nc = size(img,2)
      filler = fill(zero,(1,nc))
      rand(Bool) ? (return vcat(img[1+k:end,:],filler)) : return vcat(filler,img[1:end-k,:])
  end
end


# functions below are higher level

function make_areasumm_stimulus(imgs_input,model_sizes, grey_val,shift_val=0.0)
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
      cut_outside!(v,r;zero=grey_val, x_rel=x_img,y_rel=y_img)
  end
  return DataFrame(
    idx=idx, img_idx=images,
    size=sizes , view = views )
end



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
    grat_pars = GratingParams(size=size_patch, ori=ori,frequency=freq,
                contrast = contrast_base, noise_std=0.0 , phase=phase )
    view = gratings_enlarging_center([sz],grat_pars;rel_center=(sh,sh))[1]
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
    grat_pars = GratingParams(size=size_patch, ori=0.0,frequency=stim_freq,
                phase=stim_phase,
                contrast = contrast_base, noise_std=0.0)
    view = gratings_surround_polarization(r_small,r_large_st,
        r_large_en,ori,grat_pars ; rel_center=(sh,sh))[1]
    view .*= contr
    push!(df_ret,[idx sh stim_freq stim_phase contr r_small r_large_st r_large_en ori  [view]])
  end
  df_ret
end
