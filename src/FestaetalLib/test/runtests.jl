using Test
using FestaetalLib ; const F=FestaetalLib
const P = Pyramids

# code from Pyramids/test/runtests.jl
function end_to_end(test_im, T)
    if typeof(T) <: P.ComplexSteerablePyramid
        scale = 0.5^(1/4);
        pyramid = P.ImagePyramid(test_im, T, scale=scale, num_orientations=8, max_levels=23, min_size=15, twidth=1.0)
    else
        pyramid = P.ImagePyramid(test_im, T, max_levels=23, min_size=15)
    end

    test_im_recon = P.toimage(pyramid)

    return all((test_im_recon .- test_im) .< 0.0002)
end

rand_im = rand(128, 128)
@testset "Pyramids" begin
    println("Running end-to-end image comparison test for Complex Steerable Pyramid.")
    @test end_to_end(rand_im, P.ComplexSteerablePyramid())
end

# test
@testset "data paths" begin
    dirfile = abspath(@__DIR__,"..","..","..","data","local_dirs.json")
    if !isfile(dirfile)
        @error("""
        File $dirfile not found!
        Please create it and fill it with the paths to data folders
        as indicated in the README.md file.""")
    end
    paths = F.read_dirfile()
    for p in values(paths)
        @test isdir(p)
    end
    # test Stan
    standir_bin = joinpath(paths["dir_stan_home"],"bin")
    @test isdir(standir_bin)
    @test length(readdir(standir_bin)) > 3
    # test natural images
    dirimg = paths["dir_img"]
    imgs = filter(s->occursin(r"jpg"i,s),readdir(dirimg))
    @test length(imgs) > 50
    # test exp data TODO
end
