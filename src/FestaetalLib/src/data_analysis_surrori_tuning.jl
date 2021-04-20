

function name_of_session_cadet(fullpath)
  reg = r"(?!/|\\)([^/|\\]+)_spikesRCC\.mat"
  return string(match(reg,fullpath)[1])
end

function name_of_session_monyet(fullpath)
  reg = r"(?!/|\\)([^/|\\]+)_resorted_GratOriSurr\.mat"
  return string(match(reg,fullpath)[1])
end



### start with Monyet data

# get list of files to read
dir_surrori_tuning = function()
  ret=joinpath(read_dirfile()["dir_exp"],"surrori_tuning")
  @assert ispath(ret) "Directory $ret not found! (surround orientation tuning data)"
  return ret
end

surrori_monyet_datafiles = function()
   ret = joinpath.(dir_surrori_tuning(),
    ["monyetV1p065hs_resorted_GratOriSurr.mat",
    "monyetV1p066-67hs_resorted_GratOriSurr.mat" ,
     "monyetV1p069-70-71hs_resorted_GratOriSurr.mat"])
   for _r in ret
     @assert isfile(_r) "file $_r not found!"
   end
   return ret
 end

function read_surrori_monyet()
  @info """
  reading the matlab data files for surround orientation modulation (Monyet)
  """
  datafiles=surrori_monyet_datafiles()
  trains = [d["resp_train"] for d in matread.(datafiles)]
  dfs = map(zip(trains,datafiles)) do (train,fname)
    fields = (
      neuron = UInt8.(1:size(train, 1)),
      contrast = Float32.([0.5, 1]),
      oriS = vcat(UInt8.([0, 45, 90, 135]), missing),
      oriC = UInt8.([0, 90]),
      trial = UInt8.(1:size(train, 5)),
    )
    df = matrix_to_dataframe_binvect(train, fields)
    df[!,:electrode] .= UInt8(1)
    sname = name_of_session_monyet(fname)
    df[!,:session] .= sname
    categorical!(df,[:session,:contrast,:oriS]; compress = true)
    @info "file completed"
    return df
  end
  dfdata =  vcat(dfs...)
  # now convert it to spiking data
  # remve the missings (due to diff number of trials)
  dropmissing!(dfdata,:spk; disallowmissing=true)
  # create the views
  colw = [:contrast,:oriC,:oriS]
  views = unique(select(dfdata,colw))
  sort!(views,colw)
  views[!,:view] = UInt8.(1:nrow(views))
  dfall = innerjoin(dfdata,views ; on=colw,matchmissing=:equal)
  timestim = 0.0
  T = 501
  time_bins = (collect(0:T) .* 1E-3) .- 150E-3
  times = midpoints(time_bins)
  # define the object
  return SpikingData(dfall, views, timestim,time_bins,times)
end

## Now Cadet data


surrori_cadet_datafiles = function()
  dirdata = dir_surrori_tuning()
  ret= filter(f->occursin(r"_spikesRCC.mat\b",f),readdir(dirdata,join=true))
  for _r in ret
    @assert isfile(_r) "file $_r not found!"
  end
  return ret
end


function read_views_cadet_awake_ori(file)
  views_raw = matread(file)["viewInfo"]
  nviews=size(views_raw,1)
  views = DataFrame()
  views[!,:view] = UInt8.(views_raw[:,1])
  views[!,:oriC] = map(zip(views_raw[:,2],views_raw[:,3])) do (sc,oc)
      sc ≈ 0 ? missing : Int16.(oc)
  end
  views[!,:size] = map(1:nviews) do i
  ismissing(views[i,:oriC]) && return missing
  #  no gap, and surround size > 0 ... then it's size 5
  ( views_raw[i,9] ≈ 0 && !(views_raw[i,7] ≈ 0) ) &&  return  views_raw[i,7]
  # with gap, just pick the value
  return views_raw[i,4]
  end
  views[!,:oriSAbs] = map(1:nviews) do i
   # size 0 means no surround
   views_raw[i,5] ≈ 0 && return  missing
   # but no gap... means no surround in this dataset
   views_raw[i,9] ≈ 0 && return missing
   return views_raw[i,6]
  end
  oriSall = sort(collect(skipmissing(views.oriSAbs)))
  views[!,:hasgap] = .!ismissing.(views.oriSAbs)
  # oriS is  *absolute* in the data,
  # but should be expressed relative to center
  # orientation
  views[!,:oriS] = copy(views[!,:oriSAbs])
  for r in eachrow(views)
    if (! ismissing(r.oriC))
      r.oriS = relative_orientation(r.oriSAbs, r.oriC, oriSall)
    end end
  return views
end


# Selects only the Cadet data with 0.5 center disk and with 0.5 center disk + ring
function filter_ring_cadet!(sd::SpikingData)
  views = sd.views
  idx_hasring = views.hasgap
  idx_lgsize = .!ismissing.(views.size) .& (isapprox.(views.size,5;atol=0.2))
  idx_rfsize = .!ismissing.(views.size) .& (isapprox.(views.size,1;atol=0.1))
  idx_smsize = .!ismissing.(views.size) .& (isapprox.(views.size,0.5;atol=0.05))
  views.oriS[idx_lgsize] .= Float64.(views.oriC[idx_lgsize])
  idx_donut = ( ismissing.(views.size) .& (.!ismissing.(views.oriS)) )
  idx_all = (idx_hasring .| idx_smsize) .& (.!idx_donut)
  sd.views = views[idx_all,:]
  sd.spikes = semijoin(sd.spikes, sd.views; on=:view)
  return sd
end


function read_surrori_cadet()
  datafiles = surrori_cadet_datafiles()
  time_bins = collect(range(-0.15,0.349 ; length=500))
  trains = map(datafiles) do f
    d=matread(f)
    mat = d["spikesNstuff"]
    mat = copy( mat[ (mat[:,4] .!= 0.0) .&
    (mat[:,4] .!= 255.0) , : ])
  end
  names = map(datafiles) do f
    name_of_session_cadet(f)
  end
  dfout  = map(zip(trains, names)) do (tr,nm)
    ret = DataFrame()
    nr = size(tr,1)
    ret[!,:session] = fill(nm,nr)
    ret[!,:electrode] = UInt8.(tr[:,3])
    ret[!,:neuron] = UInt8.(tr[:,4])
    ret[!,:trial] = UInt16.(tr[:,1])
    ret[!,:view] = UInt8.(tr[:,2])
    ret[!,:spiketime] = tr[:,5]
    return ret
  end
  @info "converting spike times to binned binary vectors"
  spikes = binarybinformat(vcat(dfout...),time_bins)
  time_stim = 0.0
  times=midpoints(time_bins)
  spikes =  binarybinformat(vcat(dfout...),time_bins)
  # read the views
  views = read_views_cadet_awake_ori(datafiles[1])
  # define the object
  return filter_ring_cadet!(SpikingData(spikes, views, time_stim,time_bins,times))
end

## Inclusion criteria

function get_views_included_surrori(gratresps::DataFrame;
      kthresh = 1.0, secondary_features=[:contrast,:oriC]  )
  nneus(df)= nrow(dfneus(df))
  nneuspre =nneus(gratresps)
  @info "before selection of response Vs baseline: $nneuspre neurons"
  nneuspre == 0 && return DataFrame(keep=[])
  # surround absent, the series should have a good response vs baseline activity
  serselector = vcat(neuselector,secondary_features)
  _filt = combine(groupby(gratresps,serselector)) do df
    dfrf = @where(df, ismissing.(:oriS))
    @assert nrow(dfrf) == 1
    dfrf=dfrf[1,:]
    DataFrame(keep =   dfrf.spk_mean .>=
                (dfrf.blank_mean .+  kthresh*sqrt.(dfrf.blank_var)) )
  end
  _filt = _filt[_filt.keep,:]
  select!(_filt,Not(:keep))
  grats_select = semijoin(gratresps, _filt ; on=serselector)
  nneuspost=nneus(grats_select)
  @info "after selection: $nneuspost neurons"
  return grats_select
end

function get_views_included_surrori(sd::SpikingData ;
      kthresh = 1.0,  secondary_features=[:contrast,:oriC],
      window_stim = (50E-5,250E-3) , window_blank = (50E-3,75E-3) )
  resps = get_blank_and_window(sd,window_blank,window_stim)
  resps = innerjoin(resps,sd.views ; on =:view,matchmissing=:equal)
  ret = get_views_included_surrori(resps ;
    kthresh=kthresh, secondary_features=secondary_features)
  semijoin(resps,ret ; on=names(ret),matchmissing=:equal)
end

function define_series_surrori( data_spikecounts::DataFrame;
      secondary_features = [:contrast, :oriC])
  nneus(df) = nrow(dfneus(df))
  nneus_initial = nneus(data_spikecounts)
  dfseries = combine(groupby(data_spikecounts, neuselector)) do __df
    idx_series = 0
    combine(groupby(__df, secondary_features)) do df
        # I keep it if best score is good enough (RF size)
        # and if none of them is zero
      keepit = all(df.spk_mean .> 0)
      if keepit
        idx_series += 1
        DataFrame(series = idx_series)
      else
        DataFrame(series = missing)
      end
    end
  end
  dropmissing!(dfseries, [:series])
  ret = innerjoin(data_spikecounts,
    dfseries; on = vcat(neuselector, secondary_features),matchmissing=:equal)
  nneus_final = nneus(ret)
  @info "After the selection by series, we have $nneus_final neurons"
  @info " out of the previous $nneus_initial"
  return ret
end


"""
    add_relative_ff(df_series) -> df_series_relff

For each series, selects the FF for RF stimulus, and uses it to compute a "relative" FF
"""
function add_relative_ff(df_series)
  serselector = vcat(neuselector,:series)
  dfrelff = combine(groupby(df_series,serselector ; sort=true)) do df
    dfrf = @where(df, ismissing.(:oriS) )
    @assert nrow(dfrf) .== 1
    ff_rf = dfrf.spk_ff[1]
    (ff_rel = df.spk_ff ./ ff_rf, oriS = df.oriS )
  end
  return innerjoin(df_series,dfrelff ; on=vcat(serselector,:oriS),matchmissing=:equal)
end


function average_over_series_surrori(dat)
  if !( "ff_rel" in names(dat) )
    dat = add_relative_ff(dat)
  end
  _to_join = vcat(neuselector,:oriS)
  ret = combine(groupby(dat, _to_join)) do df
    _mu = mean(df.spk_mean)
    _ss = mean(df.spk_var)
    _ff = geomean(df.spk_ff)
    _ffrel = geomean(df.ff_rel)
    DataFrame(
      spk_mean = _mu,
      spk_var = _ss, spk_ff = _ff,
      ff_rel=_ffrel)
  end
  # adding relative rates
  retrels = combine(groupby(ret, neuselector)) do df
    dfrf = @where(df, ismissing.(:oriS))
    @assert nrow(dfrf) == 1 "Too many elements without surround! $(nrow(dfrf))"
    rf_mean = dfrf.spk_mean[1]
    spk_rate_rel = df.spk_mean ./ rf_mean
    DataFrame(
      oriS = df.oriS,
      spk_rate_rel = spk_rate_rel
    )
  end
  retboth = innerjoin(ret, retrels; on = _to_join,matchmissing=:equal)
  return retboth
end


function population_average_surrori(dfpop;ci=0.68)
  @assert "ff_rel" in names(dfpop) "Please add relative FF to series dataframe!"
  return combine(groupby(dfpop, :oriS; sort=true)) do df
    _spk_means = [s for s in df.spk_rate_rel]
    _spk_ffs = [s for s in df.spk_ff]
    _spk_ffs_rel = [s for s in df.ff_rel]
    spk_mean, spk_cidown, spk_ciup, spk_ddown, spk_dup = mean_boot(_spk_means;conf=ci)
    ff_geomean, ff_cidown, ff_ciup, ff_ddown,ff_dup = geomean_boot(_spk_ffs;conf=ci)
    ff_rel_geomean, ff_rel_cidown, ff_rel_ciup, ff_rel_ddown,ff_rel_dup =
      geomean_boot(_spk_ffs_rel;conf=ci)
    @eponymtuple( spk_mean, spk_cidown, spk_ciup , spk_ddown, spk_dup,
      ff_geomean, ff_cidown, ff_ciup , ff_ddown, ff_dup,
      ff_rel_geomean, ff_rel_cidown, ff_rel_ciup, ff_rel_ddown,ff_rel_dup)
  end
end

function session_average_surrori(dfdata;ci=0.68)
  return by(dfdata, :oriS; sort=true) do df
    _spk_means = [s for s in df.spk_mean_rel]
    _spk_ffs = [s for s in df.spk_ff]
    spk_mean, spk_cidown, spk_ciup, spk_ddown, spk_dup = mean_boot(_spk_means;conf=ci)
    ff_geomean, ff_cidown, ff_ciup, ff_ddown,ff_dup = geomean_boot(_spk_ffs;conf=ci)
    @eponymtuple( spk_mean, spk_cidown, spk_ciup , spk_ddown, spk_dup,
      ff_geomean, ff_cidown, ff_ciup , ff_ddown, ff_dup)
  end
end

function ff_differences_byseries_surrori(dfdata)
  serselector = vcat(neuselector,:series)
  return by(dfdata,serselector) do df
    is_para = (df.oriS .== 0) .&  ( .! ismissing.(df.oriS))
    is_perp = (df.oriS .== 90) .&  ( .! ismissing.(df.oriS))
    @assert all( count.([is_para,is_perp]) .== 1 )
    ffd, ffd_issigni = ff_diff_signi(df.spk_count[is_perp][1],df.spk_count[is_para][1])
    @eponymtuple(ffd, ffd_issigni)
  end
end

function ff_differences_byseries_surrori_symmetrized(dfdata)
    serselector = vcat(neuselector,:series)
    return by(dfdata,serselector) do df
      is_para = (df.oriS .== 0) .&  ( .! ismissing.(df.oriS))
      is_perp1 = (df.oriS .== 90) .&  ( .! ismissing.(df.oriS))
      is_perp2 = (df.oriS .== -90) .&  ( .! ismissing.(df.oriS))
      @assert all( count.([is_para,is_perp1,is_perp2]) .== 1 )
      spk_para,spk_perp1,spk_perp2 = [df.spk_count[idx][1]
          for idx in [is_para,is_perp1,is_perp2] ]
      ffboot_para = ff_boot(spk_para; conf=0.68)
      ffboot_perp = ff_symmetrize_boot(spk_perp1,spk_perp2;conf=0.68)
      ffd = Float64[]
      ffd_issigni=Bool[]
      for (ff_a,ff_a_ciup,ff_a_cidown, ff_b,ff_b_ciup,ff_b_cidown)  in
              zip(ffboot_perp.ff,ffboot_perp.ff_ciup,ffboot_perp.ff_cidown,
                  ffboot_para.ff,ffboot_para.ff_ciup,ffboot_para.ff_cidown)
        (_d,_s)=ff_diff_signi(ff_a,ff_a_ciup,ff_a_cidown, ff_b,ff_b_ciup,ff_b_cidown)
        push!(ffd,_d)
        push!(ffd_issigni,_s)
      end
      ffd_issigni = BitArray(ffd_issigni)
      @eponymtuple(ffd,ffd_issigni)
    end
end

## Histogram of ff differences, considering the averages over series for each neuron
#

function surrori_get_keyvals_neus(df_neus)
  if !( "ff_rel" in names(df_neus) )
    df_neus = add_relative_ff(df_neus)
  end
  return combine(groupby(df_neus,neuselector)) do df
    dfrf = @where(df , ismissing.(:oriS))
    @assert nrow(dfrf) == 1 ; dfrf = dfrf[1,:]
    dflg = @where(df , .!ismissing.(:oriS),:oriS .== 0)
    @assert nrow(dflg) == 1 ; dflg = dflg[1,:]
    dfpe = @where(df , .!ismissing.(:oriS),:oriS .== 90)
    @assert nrow(dfpe) == 1 ; dfpe = dfpe[1,:]
    # add if difference is is significant !
    ff_diff_pelg,ff_diff_pelg_issigni =
       ff_diff_signi(dfpe.spk_count,dflg.spk_count)
    rate_diff_pelg,rate_diff_pelg_issigni =
      rate_diff_signi(dfpe.spk_count,dflg.spk_count)
    (ff_rf = dfrf.spk_ff ,  ff_lg = dflg.spk_ff ,ff_pe = dfpe.spk_ff,
      ff_rel_rf = dfrf.ff_rel ,  ff_rel_lg = dflg.ff_rel ,ff_rel_pe = dfpe.ff_rel,
      spk_rf = dfrf.spk_mean ,  spk_lg = dflg.spk_mean, spk_pe = dfpe.spk_mean,
      spk_rel_rf = dfrf.spk_mean / dfrf.spk_mean ,
      spk_rel_lg = dflg.spk_mean / dfrf.spk_mean,
      spk_rel_pe = dfpe.spk_mean / dfrf.spk_mean,
      ff_diff_pelg = ff_diff_pelg, ff_diff_pelg_issigni=ff_diff_pelg_issigni,
      rate_diff_pelg = rate_diff_pelg, rate_diff_pelg_issigni=rate_diff_pelg_issigni)
  end
end
function surrori_get_keyvals_means(dfkvals)
  #spk first
  spk_rel_lg = mean_boot(dfkvals.spk_rel_lg; prefix="spk_rel_lg_mean")
  spk_rel_pe = mean_boot(dfkvals.spk_rel_pe; prefix="spk_rel_pe_mean")
  p_spk_rel_pelg = spitp(dfkvals.spk_rel_pe,dfkvals.spk_rel_lg)
  #spk , non relative
  spk_lg = mean_boot(dfkvals.spk_lg; prefix="spk_lg_mean")
  spk_pe = mean_boot(dfkvals.spk_pe; prefix="spk_pe_mean")
  p_spk_pelg = spitp(dfkvals.spk_pe,dfkvals.spk_lg)
  # now fanos
  ff_rel_lg = geomean_boot(dfkvals.ff_rel_lg; prefix="ff_rel_lg_geomean")
  ff_rel_pe = geomean_boot(dfkvals.ff_rel_pe; prefix="ff_rel_pe_geomean")
  #non relative version
  ff_lg = geomean_boot(dfkvals.ff_lg; prefix="ff_lg_geomean")
  ff_pe = geomean_boot(dfkvals.ff_pe; prefix="ff_pe_geomean")
  p_ff_rel_pelg = spitp(dfkvals.ff_rel_pe,dfkvals.ff_rel_lg)
  p_ff_pelg = spitp(dfkvals.ff_pe,dfkvals.ff_lg)
  # score for spk and FF , and p value of score
  scorefun(a,b)= 200*(a-b)/(a+b)
  _scors =   scorefun.(dfkvals.spk_pe,dfkvals.spk_lg)
  scor_spk_pelg = mean_boot(_scors ;  prefix="spk_score_pelg_mean"  )
  p_spk_score_pelg = spitp(_scors)
  _scors =   scorefun.(dfkvals.spk_rf,dfkvals.spk_lg)
  scor_spk_rflg = mean_boot(_scors ; prefix="spk_score_rflg_mean"  )
  p_spk_score_rflg = spitp(_scors)
  _scors =   scorefun.(dfkvals.ff_rf,dfkvals.ff_lg)
  scor_ff_rflg = mean_boot( _scors ; prefix="ff_score_rflg_mean"  )
  p_ff_score_rflg =spitp(_scors)
  _scors =   scorefun.(dfkvals.ff_pe,dfkvals.ff_lg)
  scor_ff_pelg = mean_boot(_scors ; prefix="ff_score_pelg_mean"  )
  p_ff_score_pelg = spitp(_scors)
  return merge(spk_lg,spk_pe,spk_rel_lg,spk_rel_pe,ff_rel_lg,ff_rel_pe,ff_lg,ff_pe,
    scor_spk_pelg,scor_spk_rflg,scor_ff_pelg,scor_ff_rflg,
    @eponymtuple(p_spk_pelg,p_spk_rel_pelg,p_ff_pelg,p_ff_rel_pelg,
    p_spk_score_pelg,p_spk_score_rflg,p_ff_score_pelg,p_ff_score_rflg ))
end
