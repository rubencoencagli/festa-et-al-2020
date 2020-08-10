#=
Here I use Stan to find the posterior of the
GSM given data, code adapted from Abhi
=#

# N are the number of filters*2 , i.e. the filter output dimension
# M are the datapoints (image patches)
# warning: this code will get progressively worse as noise becomes smaller!
const stan_GSM = """
  data {
     int N;
     int M;
     matrix[N,M] X;
     cov_matrix[N] Gcov;
     cov_matrix[N] Ncov;
     real<lower=0> ray_alpha;
  }
  transformed data {
    vector[N] Mean = rep_vector(0,N);
  }
  parameters {
     matrix[N,M] g;
     vector<lower=0>[M] v;
  }
  model {
    v ~ rayleigh(ray_alpha);
     for (j in 1:M){
         g[:,j] ~ multi_normal(Mean,Gcov);
         X[:,j] ~ multi_normal(v[j]*g[:,j],Ncov);
     }
  }
"""

const stan_GSM_1D = """
  data {
     int M;
     vector[M] X;
     float<lower=0> Gcov;
     float<lower=0> Ncov;
     real<lower=0> ray_alpha;
  }
  parameters {
     vector[M] g;
     vector<lower=0>[M] v;
  }
  model {
    v ~ rayleigh(ray_alpha);
     for (j in 1:M){
         g[j] ~ normal(0.0,sqrt(Gcov));
         X[j] ~ normal(v[j]*g[j],sqrt(Ncov));
     }
  }
"""
# noise is a matrix because it can be set to zero
const stan_GSM_nog = """
    data {
        int<lower=1> N;
        int<lower=1> M;
        real<lower=0> ray_alpha;
        cov_matrix[N] Gcov;
        matrix[N,N] Ncov;
        matrix[N,M] X;
    }
    transformed data {
        vector[N] Mean = rep_vector(0,N);
        }
    parameters {
        real<lower=1E-3> v[M];
    }
    model {
        v ~ rayleigh(ray_alpha);
        for (t in 1:M)
            X[:,t] ~ multi_normal(Mean, square(v[t])*Gcov+Ncov);
    }
"""

const stan_GSM_nog_1D = """
    data {
        int<lower=1> M;
        real<lower=0> ray_alpha;
        real<lower=0> Gcov;
        real<lower=0> Ncov;
        vector[M] X;
    }
    parameters {
        real<lower=1E-3> v[M];
    }
    model {
        v ~ rayleigh(ray_alpha);
        for (t in 1:M)
            X[t] ~ normal(0.0, sqrt(square(v[t])*Gcov+Ncov));
    }
"""
# auxiliary function for the function below
function sample_gs(Sigma_g_inv::AbstractMatrix,
        Sigma_n_inv::AbstractMatrix, hasnoise::Bool , x::Vector{Float64},v::Float64)
    # to make sure that the matrix is positive definite
    if  !hasnoise
        return x./v
    end
    S3 = collect(Sigma_g_inv) # get rid of Symmetric type (if present)
    BLAS.axpy!(v*v,Sigma_n_inv,S3)
    Sigma3=inv(S3) |> Symmetric
    mu3=v*Sigma3*(Sigma_n_inv*x)
    rand(MultivariateNormal(mu3,Sigma3))
end

"""
        sample_gs(gsm::GSM,xs::Matrix{Float64},vs::Matrix{Float64})

Returns ``P(g|x,v)`` when noise is present.
# Inputs
  - `gsm` GSM model.
  -  `xs` matrix where each column is a vector of filter output
  -  `vs` vector where the first dimension is the mixer corresponding to the column in `xs` , the secon dimension are all the samples that have been taken
 # Outputs
   - `gs` matrix where each column is a single sample corresponding to ``P(g,x,v)``
"""
function sample_gs(gsm::GSM,xs::AbstractArray,vs::Matrix{Float64})
    n_dims,n_data,is1d = let _d = ndims(xs)
       if _d == 2
         (size(xs)...,false)
       elseif _d==1
         xs=reshape(xs,1,:)
         (size(xs)...,true)
       else
         error("input xs has the wrong structure!")
       end
    end
    n_sampl=size(vs,2)
    Sg_inv = inv(gsm.covariance_matrix) |> Symmetric
    Sn_inv,hasnoise = let Sn=gsm.covariance_matrix_noise
        if isapprox(tr(Sn),0 ; atol=0.0001)
          Matrix{Float64}(undef,1,1),false  # if null, it will not be used
        else
          inv(Sn),true
        end
    end
    gs=Array{Float64}(undef,n_dims, n_data,n_sampl)
    for t in 1:n_data , s in 1:n_sampl
        gs[:,t,s] = sample_gs(Sg_inv,Sn_inv,hasnoise,xs[:,t],vs[t,s])
    end
    gs
end

global stan_folder = ""

function set_stan_folder(folder)
  @assert isdir(folder) "$folder is not a valid directory"
  global stan_folder=folder
  set_cmdstan_home!(folder)
end

"""
    sample_posterior(gsm::GSM{RayleighMixer},X::Matrix, n_samples ;
            thin_val=2,pdir=joinpath(@__DIR__(),"../other/StanTmp") )

Produces samples from the posterior distribution ``P(g|x)`` using Stan
# Inputs
  -  `gsm` : the GSM model, it must have a Rayleigh mixer
  - `X` : each column represents a filter output vector
  - `n_samples` : meh
  - `thin_val` : the actual number of samples is n_samples/thin_val
  - `pdir` : temporary director where Stan writes its files
# Outputs
 The output is a Tuple with fields gs, vs. The dimensions of gs are dim_input x n_points x n_samples . vs are n_points x n_samples
"""
function sample_posterior(gsm::GSM{RayleighMixer},X, n_samples ;
            thin_val=2,pdir=joinpath(@__DIR__(),"../other/StanTmp"),
            nwarmup=1000,
            nchains=4)
    @assert !isempty(stan_folder) "Please set the folder of cmdstan using set_stan_folder"
    println("the following temporary directory will be used" * pdir)
    dim = size(X,1)
    num_data = size(X,2)
    n_samples=thin_val*n_samples
    Gcov = gsm.covariance_matrix
    Ncov = gsm.covariance_matrix_noise
    ray_alpha = gsm.mixer.alpha
    Data=[  Dict("N"=>dim,"M"=>num_data,
                    "X"=>X,"Ncov"=>Ncov,"Gcov"=>Gcov,
                    "ray_alpha"=>ray_alpha ) for _ in 1:nchains]
    stanmodel = Stanmodel(num_samples=n_samples,
            thin=thin_val, name="GSM_posterior",
            model=stan_GSM,
            tmpdir=pdir,
            nchains=nchains,
            num_warmup=nwarmup)
    sim = stan(stanmodel, Data ;
        summary=false , file_run_log=true)
    get_data = get_data_all(sim[2],sim[3])
    gs = get_data("g",dim,num_data)
    vs = get_data("v",num_data)
    (gs=gs,vs=vs)
end

"""
    function sample_posterior_fast(gsm::GSM{RayleighMixer},X::Matrix, n_samples ;
                    thin_val=2,
                    pdir=joinpath(@__DIR__(),"../other/StanTmp"),
                    nchains=4,
                    nwarmup=1000 )

Produces samples from the posterior distribution ``P(g|x)`` using Stan
# Inputs
  - `gsm` : the GSM model, it must have a Rayleigh mixer
  - `X` : each column represents a filter output vector
  - `n_samples` : samples taken from each chain (after thinning)
  - `thin_val` : thinning of the sampling
  - `pdir` : temporary director where Stan writes its files
  - `nchains` : number of parallel chains, the number of samples is
        n_sampl*n_chains
# Outputs
 The output is a Tuple with fields gs, vs. The dimensions of gs are dim_input x n_points x n_samples . vs are n_points x n_samples
"""
function sample_posterior_fast(gsm::GSM{RayleighMixer},X, n_samples ;
            thin_val=2,
            pdir=joinpath(@__DIR__(),"../other/StanTmp"),
            nchains=4,
            nwarmup=1000 )
    @assert !isempty(stan_folder) "Please set the folder of cmdstan using set_stan_folder"
    println("the following temporary directory will be used" * pdir)
    dim = size(X,1)
    isonedim = dim == 1
    num_data = size(X,2)
    n_samples=thin_val*n_samples
    Gcov = gsm.covariance_matrix
    Ncov = gsm.covariance_matrix_noise
    _model = stan_GSM_nog
    if isonedim
      @warn "Sampling will be done using the 1D version of the GSM model"
      Gcov=Gcov[1,1]
      Ncov = Ncov[1,1]
      X = X[:]
      _model = stan_GSM_nog_1D
    end
    ray_alpha = gsm.mixer.alpha
    Data=[  Dict("N"=>dim,"M"=>num_data,
                    "X"=>X,"Ncov"=>Ncov,"Gcov"=>Gcov,
                    "ray_alpha"=>ray_alpha    ) for _ in 1:nchains]
    stanmodel = Stanmodel(num_samples=n_samples,
            thin=thin_val, name="GSM_posterior_faster",
            model= _model, tmpdir=pdir,
            nchains=nchains,
            num_warmup=nwarmup,
            output_format=:mcmcchains)
    sim = stan(stanmodel, Data ;
        summary=false , file_run_log=true)
    vs = get_data_vec(:v,sim[2]) |> permutedims
    println("All done in Stan! Now sampling g using Julia")
    gs = sample_gs(gsm, X,vs)
    println("Julia sampling complete!")
    if isonedim
      gs = dropdims(gs;dims=1)
    end
    (gs=gs,vs=vs)
end


"""
    function sample_posterior(n_samples::Integer,gsm_model::GSM_Model ;
        nwup=1_000,
        nchains=4,
        thin_val=2 )

Produces samples from the posterior distribution ``P(g|x)`` using Stan
Calls the function `sample_posterior_fast` over the gms included in the `gsm_model` struct
and ALL the image patches also included in the dataframe `gsm_model.views`

# Inputs
  - `n_samples` : samples taken from each chain (after thinning)
  - `gsm_model` : the GSM model (Rayleigh mixer) , along with image patches
  - `thin_val` : thinning of the sampling
  - `pdir` : temporary director where Stan writes its files
  - `nchains` : number of parallel chains, the number of samples is
        n_sampl*n_chains
# Outputs
 The output is a Tuple with fields gs, vs. The dimensions of gs are dim_input x n_points x n_samples . vs are n_points x n_samples
"""
function sample_posterior(nsampl::Integer,gm::GSM_Model ;
    nwup=1_000, nchains=4,thin_val=2, addnoise=true, dir_temp=@__DIR__(),
    x_postprocess=nothing)
  views = gm.views.view
  xpost = something(x_postprocess,Xscale(inv(std(gm.x_natural))))
  stim_tiles = [ (s,[(-1,-1)] ) for s in views]
  xstim = apply_bank(stim_tiles, gm.bank, xpost,false)
  x_stim_noise = addnoise ? add_xnoise(gm.gsm , xstim) : xstim
  dirsampl = Base.Filesystem.mktempdir(dir_temp; cleanup=true)
  @info "now sapling on STAN ! Starting from the stimuli. Temp data saved in $dirsampl"
  allsampl = sample_posterior_fast(gm.gsm,x_stim_noise,nsampl ;
    nwarmup=nwup, nchains=nchains, pdir=dirsampl,thin_val=thin_val)
  @info "sampling done!"
  # Base.Filesystem.rm(dirsampl;recursive=true)
  idxs = gm.views.idx
  ndata = length(idxs)
  gs = [ allsampl.gs[:,i,:] for i in 1:ndata]
  vs = [ allsampl.vs[i,:]  for i in 1:ndata]
  sampldf = DataFrame(idx=idxs, gs=gs, vs=vs)
  gm.samples=sampldf
  return nothing
end

function get_data_vec(st::Symbol, ch::MCMCChains.Chains)
  sts=string(st)
  pnames  = filter(p->occursin(Regex("^$sts\\."),string(p)), ch.name_map.parameters)
  vals = get(ch,Symbol.(pnames))
  nc = length(vals)
  nr = length(vals[1])
  ret = Matrix{Float64}(undef,nr,nc)
  for (k,v) in pairs(vals)
    idx = let cap = match(r".*\.([0-9]*)\b", string(k))
      parse(Int32,cap.captures[1])
    end
    ret[:,idx] .= vec(v)
  end
  ret
end
function get_data_mat(st::Symbol, ch::MCMCChains.Chains)
  sts=string(st)
  pnames  = filter(p->occursin(Regex("^$sts\\."),string(p)), ch.name_map.parameters)
  vals = get(ch,Symbol.(pnames))
  idxs =  map(keys(vals)) do k
    cap = match(r".*\.([0-9]*)\.([0-9]*)\b",string(k))
    (parse(Int32,cap.captures[1]), parse(Int32,cap.captures[2]))
  end
  nr = maximum(first.(idxs))
  nc = maximum(last.(idxs))
  sampl = length(vals[1])
  ret = Array{Float64}(undef,sampl,nr,nc)
  for (v,idx) in zip(values(vals),idxs)
    ret[:,nr,nc] .= vec(v)
  end
  ret
end

# general functions to read data of any dimesion from Stan
function get_stan_data_fun(data::Array{Float64},field_names::Vector{String})
  function out(str::String)
    idx=findfirst(s->s==str,field_names)
    idx ==0 && error("cannot read data, the name $str is not in the database")
    vec(data[:,idx,:])
  end
end

"""
Reads all elements or a matrix or a vector with indexes specified
by dims
"""
function get_data_all(data,datanames)
    get_data=get_stan_data_fun(data,datanames)
    function f_out(data_name::String,dims::Integer...)
        nd=length(dims)
        n_sampl=get_data(data_name * ".1"^nd) |> length
        out=Array{Float64}(undef,dims...,n_sampl)
        to_iter = Iterators.product([(1:d) for d in dims]...)
        for ijk in to_iter
            _str=data_name
            for i in ijk
                _str*=".$i"
            end
            out[ijk...,:] = get_data(_str)
        end
        out
    end
end

"""
   gs_vec_to_array(gs_vec::Vector{<:Matrix})

Converts an array of matrices to a single 3D array
"""
function gs_vec_to_array(gs_vec::Vector{Matrix{T}}) where T<:Real
  nstim=length(gs_vec)
  ndims,nsampl=size(gs_vec[1])
  gs = Array{T}(undef,ndims,nstim,nsampl)
  for s in 1:nstim
      gs[:,s,:] .= gs_vec[s]
  end
  gs
end
