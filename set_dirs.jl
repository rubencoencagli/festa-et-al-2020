
# JSON package needed
using Pkg
try
    @eval using JSON
catch e
    pkg"add JSON"
    using JSON
end

# default values
this_dir = @__DIR__
path_exp_default = joinpath(this_dir,"data","experiments")
path_img_default = joinpath(this_dir,"data","natural_images")
path_stan_default = joinpath(homedir(),".cmdstan-2.20.0")

function warnpath(path)
    if !isdir(path)
        @error("\nDirectory $path not found! Is it the correct path?"*
          "\n If not, please run this script again with the correct path")
        println()
    end
    return nothing
end

function get_path(prompt::String,default::String)
    println(prompt * " (default is: $default)")
    ret = let _r = readline()
        isempty(_r) ? default : _r
    end
    warnpath(ret)
    return ret
end
get_path(pd::Tuple)=get_path(pd[1],pd[2])


path_img, path_exp, path_stan_home = map(get_path,
    zip( [ "Select directory for natural images",
            "Select directory for experimental data",
            "Select home directory of cmdstan installation"],
         [path_img_default, path_exp_default,path_stan_default]) )


sav_dict = Dict( "dir_img"=> path_img,
                  "dir_exp" => path_exp,
                  "dir_stan_home"=>path_stan_home)

sav_folder=mkpath(joinpath(this_dir,"data"))
sav_file= joinpath(sav_folder,"local_dirs.json")
open(sav_file,"w") do f
    write(f,JSON.json(sav_dict))
end

@info("\nPath information saved in $sav_file\n All done!")
