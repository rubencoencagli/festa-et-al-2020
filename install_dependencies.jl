using Pkg
pkg"dev src/FestaetalLib"
# pkg"add src/FestaetalAnalytic"

pkg"add Images, ImageMagick"
pkg"add Serialization, MAT, DataFrames, DataFramesMeta,Dates,Revise"
pkg"add Plots,NamedColors"
pkg"add Statistics, StatsBase, LinearAlgebra , Bootstrap"
pkg"precompile"
@info "Now testing some dependencies..."
pkg"test FestaetalLib"
@info "All done!"
