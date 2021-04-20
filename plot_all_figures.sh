#!/bin/sh
# Plots all figures, one by one.
# warning : the runing time is between one and two hours. 

julia paperfigures/1D.jl --rebuildGSM
julia paperfigures/2C.jl
julia paperfigures/2D.jl --rebuildGSM
julia paperfigures/2E.jl
julia paperfigures/2F.jl
julia paperfigures/3A.jl --rebuildGSM
julia paperfigures/3BC.jl

echo All figures plotted!
