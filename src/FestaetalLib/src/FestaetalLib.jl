


module FestaetalLib
  export Pyramids

  pyramidsmodule = abspath(@__DIR__,"..","..","Pyramids","src","Pyramids.jl")
  @assert isfile(pyramidsmodule) "file $pyramidsmodule not found"
  include(pyramidsmodule)


  using Statistics, LinearAlgebra, StatsBase
  using Distributions, Random

  using OffsetArrays
  using Images, FileIO

  using CmdStan , MCMCChains
  using Serialization  # For input and output
  using DataFrames, DataFramesMeta # better storage

  using EponymTuples

  include("naturalRead.jl")
  include("drawgratings.jl")
  include("steerPyramids.jl")
  include("baseGSM.jl")
  include("stanSampl.jl")

end # module
