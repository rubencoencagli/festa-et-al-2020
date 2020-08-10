using Pkg
pkg"add src/FestaetalLib"
pkg"add src/Pyramids"
pkg"add src/FestaetalAnalytic"
pkg"precompile"

# and test it
pkg"test FestaetalLib"