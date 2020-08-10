#=
Reads natural images
and regularizes them if needed
=#

"""
get_file_paths(dir,extension)
Finds all the files with a certain extension in the indicated directory.
It also explores subdirectories!
Returns the list of all files found.
"""
function get_file_paths(dir,extension)
    @assert isdir(dir) "Directory $dir not found!"
    files_out = String[]
    right_ext(file) = occursin(Regex("$extension\\b"),file)
    for (root , _ , files) in walkdir(dir)
        for f in files
            if right_ext(f)
                push!(files_out,joinpath(root,f))
            end
        end
    end
    @assert !isempty(files_out) "No file with extesion $extension found!"
    files_out
end
# rotate images exactly
function rot_90(img)
    r,c=size(img)
    out=similar(img,c,r)
    for cc in 1:c
        out[cc,:] = img[end:-1:1,cc]
    end
    out
end
function rot_m90(img)
    r,c=size(img)
    out=similar(img,c,r)
    for (cc,mcc) in zip(1:c,c:-1:1)
        out[cc,:] = img[:,mcc]
    end
    out
end
function rot_180(img)
    r,c=size(img)
    out=similar(img)
    for (cc,mcc) in zip(1:c,c:-1:1)
        out[:,cc] = img[r:-1:1,mcc]
    end
    out
end
const rotation_functions = [identity , rot_90, rot_m90, rot_180 ]

function rand_rot(img)
    rfun=rand(rotation_functions)
    rfun(img)
end

# regularization types
abstract type ReguType end
struct NoRegu <: ReguType end
struct StandardRegu <: ReguType end

function regularize_image(img,rt::NoRegu)
    convert(Array{Float64},Gray.(img))
end
function regularize_image(img,rt::StandardRegu)
    im=regularize_image(img,NoRegu())
    _mu,_std = mean(im),std(im)
    @. im = (im -_mu)/_std
end

"""
        function cut_patch(img, sz)

cuts a random square section of size `sz` from the matrix `img`
"""
function cut_patch(img::AbstractArray{T}, sz) where T
  mat=Matrix{T}(undef,sz,sz)
  cut_patch!(mat,img,sz)
end

function cut_patch!(mat,img,sz)
  _rows, _cols = size(img)
  _crows,_ccols =_rows-sz+1, _cols-sz+1
  i,j= rand(1:_crows), rand(1:_ccols)
  mat .= view(img,i:i+sz-1,j:j+sz-1) # use view to make sure no space is allocated
end


"""
Reads ALL images in a directory, cutting them in a square of specified size
if rotate is true, each image is repeated 4 times for each roation
(tiled sampling)
"""
function read_natural_images(dir,sz, (img_regu::ReguType) = StandardRegu() ;
        (rotate::Bool)=true,(verbose::Bool)=true,
        (extension::String)=".jpg")::Vector{Matrix{Float64}}
    @assert isdir(dir) "directory $dir not valid"
    check_size(img,sz) = min(size(img)...) >= sz
    img_files=get_file_paths(dir,extension)
    @assert !isempty(img_files) "no images found in directory $dir"
    ret=Matrix{Float64}[]
    _rot_fun = rotate ? rotation_functions : [identity,]
    nfiles=length(img_files)
    for (k,file) in enumerate(img_files)
        if verbose
            println("reading file $k/$(nfiles)")
        end
        img=load(file)
        !check_size(img,sz) && continue #too small, ignore
        for rot in _rot_fun
            img_rot = rot(img)
            img_cut = cut_patch(img_rot,sz)
            img_reg=regularize_image(img_cut,img_regu)
            push!(ret,img_reg)
        end
    end
    return ret
end

"""
    function sampling_coords(n_samples::Integer, images::Vector{Matrix{Float64}},
        padding::Integer)

Utility function, returns a vector of pairs
`(image, Vector(central points to sample) )`
if the image is too small, the sample coordinates are not
necessarily unique!
Takes an equal number of samples from each images. If the number cannot be divided, picks
extra images randomly from the existing set.
"""
function sampling_coords(n_samples::Integer, images::Vector{Matrix{Float64}},
    padding::Integer)
  @assert 2padding < minimum(size(images[1])) "image too small!"
  n_img=length(images)
  #check how many samples per image
  n_full,n_rem = divrem(n_samples,n_img)
  v_sampl = fill(n_full,n_img)
  # add the rem, picking them randomly!
  if n_rem>0
    idx_rem=sample(1:n_img , n_rem ;replace=false)
    v_sampl[idx_rem] .+= 1
  end
  @assert  sum(v_sampl) == n_samples "something not right"
  out = map(zip(images,v_sampl))  do (im,ns)
    szx,szy = size(im) # for simplicity x referst to fist coord, y to second.
    # easier to just pick square images and avoid complications
    xv=(padding+1):(szx-padding+1)
    yv=(padding+1):(szy-padding+1)
    xsampl= sample(xv, ns) # repetitions allowed !
    ysampl= sample(yv, ns)
    xysampl = collect(zip(xsampl,ysampl))
    (im,xysampl)
  end
  # if images are not used, exclude them
  if n_full == 0
      out[idx_rem]
  else
      out
  end
end



"""
    sampling_tiles(n_samples::Integer, images::Vector{Matrix{Float64}},
        patch_size::Integer)
Generates `n_samples` patches of size `patch_size`,
"""
function sampling_tiles(n_samples::Integer, images::Vector{Matrix{Float64}},
    patch_size::Integer)
  n_img=length(images)
  @assert !isempty(images) "images are missing!"
  #check how many patches per image
  n_full,n_rem = divrem(n_samples,n_img)
  v_sampl = fill(n_full,n_img)
  # add the rem, picking them randomly!
  if n_rem>0
    idx_rem=sample(1:n_img , n_rem ;replace=false)
    v_sampl[idx_rem] .+= 1
  end
  @assert  sum(v_sampl) == n_samples "something not right"
  xycent = (-1,-1) # by convention, this should center the matrix
  out = [ (Matrix{Float64}(undef,patch_size,patch_size),[xycent]) for i in 1:n_samples ]
  k=0
  foreach(zip(images,v_sampl))  do (im,ns)
    for s in 1:ns
      k+=1
      out_img = out[k][1]
      cut_patch!(out_img,im,patch_size)
    end
  end
  out
end

function regularize01!(mat)
    l,u = extrema(mat)
    @. mat = (mat-l)/(u-l)
end
