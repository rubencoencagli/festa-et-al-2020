#=
Defines the steerable pyramid filter
and inverts it for representation purposes
=#

struct PyramidPars
    twidth::Real # scale parameter of the pyramid!
    scale::Real # always 0.5
    max_levels::Integer # maximum pyramid levels
    oris_total::Integer # total pyramid orientations
    oris_select::Integer # number of orientations that are selectable
end

struct OnePyrBank{T<:Vector{V} where V<: Vector{<:Integer}}
    pyrpars::PyramidPars
    coords::T # array of arrays -> [ level, #ori , x (horiz) , y (vertical) ]
    # warning! The coordinates depend on the level! e.g. level 2   x_pixels <- 2*x
end
import Base.length
length(pb::OnePyrBank) = length(pb.coords)

# auxiliary functions for OnePyrBank constructor

import Images.centered
function centered(mat::AbstractMatrix,ic::I,jc::I) where I<: Integer
  if  (ic < 0) || (ic < 0)
    return centered(mat)
  end
  (r,c)=size(mat)
  if !checkbounds(Bool,mat,ic,jc)
    error("out of bounds!")
  end
  rows=(1-ic):(r-ic)
  cols=(1-jc):(c-jc)
  OffsetArray(mat,rows,cols)
end
"""
    function get_cs_coords(radius,nsurr=8)
Returns coordinates for filters, the first is always `(0,0)`
followed by `nsurr` coordinates in `(x,y)` format that move
counterclockwise , starting from `(radius,0)`
"""
function get_cs_coords(radius,nsurr)
  angs = LinRange(0,2pi,nsurr+1)[1:nsurr] |> collect
  if radius==0
    return [(0,0)]
  end
  ivals= @. Int64(round(cos(angs)*radius))
  jvals= @. Int64(round(sin(angs)*radius))
  vcat((0,0),collect(zip(ivals,jvals)))
end
"""
    get_cs_coords(radius ; ncent=1, nsurr=8)

Returns coordinates for filters, the first `ncent` are `(0,0)`
followed by `nsurr` coordinates in `(x,y)` format that move
counterclockwise , starting from `(radius,0)`
"""
function get_cs_coords(radius ; ncent=1, nsurr=8)
  onesurr = get_cs_coords(radius,nsurr)
  ncent==1 && return onesurr
  ( [onesurr[1] for _ in 1:ncent-1]... ,  onesurr...)
end

function best_pyr_scale(lev,size; scal=0.5)
  scal_pix = floor(scal*size)
  if scal_pix < 15
    @warn "size too small! Should be >15"
  end
  scal^inv(lev-1) + eps(1.0)
end
# constructors for OnePyrBank

function OnePyrBank(pyrp::PyramidPars,lev,oris,rad,ncent,nsurr ; shift=(0,0))
  __coords = get_cs_coords(rad; nsurr=nsurr , ncent = ncent)
  scoords = [ coo .+ shift for coo in __coords]
  OnePyrBank(pyrp, [ [lev,ori,coo...] for (coo,ori) in zip(scoords,oris) ] )
end
function OnePyrBank(scale::Real,rad::Real,nsurr,shift=(0,0) ;
          ncent=4, size_patch=100,level=2)
  @assert  0 < scale <= 1 "scale should be below 1"
  @assert  0 < rad <= 1 "radius should be between 0 and 1"
  @assert  nsurr in [0,4,8] "invalid number of surround filters"
  scale_pyr = best_pyr_scale(level,size_patch;scal=scale)
  rad = max(1,floor(Integer,0.49*rad*scale*size_patch))
  oris = [ collect(1:ncent)... , fill(1,nsurr)... ]
  pyrp = PyramidPars(1.,scale_pyr,level+1,8,4)
  OnePyrBank(pyrp,level,oris,rad,ncent,nsurr ; shift=shift)
end

#################
# visualize
function true_ori(ori::Integer,p::PyramidPars)
    @assert ori<= p.oris_select "orientation not present"
    fact=div(p.oris_total,p.oris_select)
    oris=1:fact:p.oris_total
    oris[ori]
end
true_ori(ori::Integer,pb::OnePyrBank)=true_ori(ori,pb.pyrpars)

"""
      function invert_test(img_size::Integer,pbank::OnePyrBank ; real_part=true)
inverts one bank of filters and returns the resulting matrix
warning: if no indexes are selected, it superimposes all banks,
including all the ones at the center.
"""
function invert_test(img_size::Integer,pbank::OnePyrBank ;
    real_part=true, idx_bank=Int64[])
  idxb =  if isempty(idx_bank)
      1:length(pbank.coords)
    else
      idx_bank
  end
  pyr= get_pyramid(zeros(img_size,img_size),pbank)
  foreach(pbank.coords[idxb]) do (level,ori,x,y)
      ori=true_ori(ori,pbank)
      new_filt = Pyramids.subband(pyr,level)[ori]
      # shift the indexes so that ceter of matrix is 0,0
      new_filtc=centered(new_filt)
      new_filtc[y,x] += (real_part ? 1.0 : 1.0im)
      Pyramids.update_subband!(pyr,level,new_filt,orientation=ori)
  end
  Pyramids.toimage(pyr)
end


# as above, but center and surround are in separate matrices!
# center is intended as the first coord
function invert_test_cs(img_size::Integer,pbank::OnePyrBank ; real_part=true)
  im1 = invert_test(img_size,pbank ; real_part = real_part , idx_bank = [1])
  idx_surr = collect(2:length(pbank.coords))
  im2 = if isempty(idx_surr)
           zero(im1)
        else
           invert_test(img_size,pbank ; real_part = real_part , idx_bank = idx_surr )
        end
  return (im1,im2)
end

function get_pyramid(img::AbstractArray{<:Real},p::PyramidPars)
    ImagePyramid(img, ComplexSteerablePyramid(),  scale=p.scale,
        max_levels=p.max_levels,twidth=p.twidth,num_orientations=p.oris_total)
end
get_pyramid(img::AbstractArray{<:Real},pb::OnePyrBank)=get_pyramid(img,pb.pyrpars)

"""
  apply_bank(pyr::Pyramids.ComplexSteerablePyramid),
      (x0::Integer)=0,(y0::Integer)=0)
Applies a bank of filters to a pyramid with a coordinate offset,
returns a vector, with real part and imaginary part for each filter
total length = 2*nfilters
Does not check if the filters are within bounds!
"""
function apply_bank(pyr::Pyramids.ImagePyramid,
        pyrbank::OnePyrBank, i0::I ,j0::I) where I<:Integer
    pyrp = pyrbank.pyrpars
    n_filt = length(pyrbank)
    out = Vector{Float64}(undef,2n_filt)
    for i in 1:n_filt
      (level,ori,x,y) = pyrbank.coords[i] # parameters of single filter
      img_pyr = subband(pyr,level)[ori] # extract corresponding pyramid band
      img_pyrc = centered(img_pyr,i0,j0) # coordinates relative to x0, y0
      cval = img_pyrc[y,x] # inversion columns rows here!
      out[2i-1] = real(cval)
      out[2i] = imag(cval)
    end
    out
end

"""
    apply_bank(img::AbstractMatrix,
        pyrbank::OnePyrBank, (x0y0_all::Vector{T})=[(-1,-1)]) where T<:Tuple

Applies a filter bank to a single image, the bank is applied once at each reference
point indicated (reference w.r.t normal matrix coordinates).  The default is to apply the
bank once, in the center of the image (when coordinats are <0 , the matrix is just
centered in the middle)
"""
function apply_bank(img::AbstractMatrix,
    pyrbank::OnePyrBank, (i0j0_all::Vector)=[(-1,-1)] )
    pyr=get_pyramid(img,pyrbank)
    n_sampl = length(i0j0_all)
    n_x = 2length(pyrbank)
    out=Matrix{Float64}(undef,n_x,n_sampl)
    for (i,i0j0) in enumerate(i0j0_all)
        out[:,i] = apply_bank(pyr,pyrbank,i0j0...)
    end
    out
end

# add post processing for banks...
abstract type XPostProcess end
struct XNothing <: XPostProcess end
# rescales for a certain std
struct Xstd <: XPostProcess
  xstd
end
struct Xscale <: XPostProcess
  xscal
end
struct XForceStd <: XPostProcess
  sigma
  idxs
end

function Xstd()
  Xstd(1.0)
end
function xpostprocess!(mat,p::XNothing)
  mat
end
# sets the overall std
function xpostprocess!(mat,p::Xstd)
  c = p.xstd / std(mat)
  mat .*= c
end
# scales all xs
function xpostprocess!(mat,p::Xscale)
  mat .*= p.xscal
end

function xpostprocess!(mat,p::XForceStd)
  sigma_old = std(mat[p.idxs,:])
  mat .*= (p.sigma/sigma_old)
end



# this version goes along with sampling_coords in the naturalRead file
# Careful: elements are not shuffled... so consecutive vectors may
# come from the same image and be highly correlated
function apply_bank(img_and_coords, pyrbank::OnePyrBank,
    xpostproc::XPostProcess, verbose)
  c=0
  ctot=length(img_and_coords)
  out = map(img_and_coords) do (img,x0y0_all)
    c+=1
    if verbose && rem(c,500)==0
      println("element $c of $ctot processed ",
          " $(round(Int,c/ctot*100))% done")
    end
    apply_bank(img,pyrbank,x0y0_all)
  end
  out=xpostprocess!(hcat(out...),xpostproc)
end
# call img_and_coords from here, for simplicity
"""
    apply_bank(n_sampl::Integer,images::Vector{Matrix{Float64}}, tile_size::Integer,
      pyrbank::OnePyrBank,(xpostproc::XPostProcess)=XNothing())

Takes `n_samples` random patches of size `tile_size` for each element of `images`
then applies the pyramid bank to each patch, with postprocessing if needed.
"""
function apply_bank(n_sampl::Integer,images::Vector{Matrix{Float64}}, tile_size::Integer,
     pyrbank::OnePyrBank,xpostproc::XPostProcess, verbose=false)
     img_and_coords = sampling_tiles(n_sampl,images,tile_size)
     apply_bank(img_and_coords, pyrbank,xpostproc,verbose)
 end


"""
    best_response(bank,stims,k)
Selects the `stims.k` entry that has the highest filter response (averaging over every other parameter). `stims` is a dataframe with `view`, that contains image patches, and `k` that contains the parameter of interest (can be a size, a spatial phase, etc)
"""
 function bank_best_response(bank,stims,k;idxs=nothing)
   dfnorms = combine(groupby(stims,k)) do df
     tiles = [ (s,[(-1,-1)] ) for s in df.view]
     xs = apply_bank(tiles, bank, Xstd(1.0),false)
     idxs = something(idxs, (1:size(xs,1)))
     xs = xs[idxs,:]
     return sum(map(norm,eachcol(xs)))
   end
   return dfnorms[argmax(dfnorms.x1),k]
 end
