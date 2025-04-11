function tfce_ds=EEG_clusterstats(cfg,data)

    %requires:
    %cfg.time (all time points)
    %cfg.nIter (number of iterations)
    %cfg.h0 (h0 value)

    ds.samples=data;
    ss=1:size(data,1);
    ds.sa.chunks=ss';
    ds.sa.targets=ones(size(ss))';
    ds.fa.time=1:length(cfg.time);
    ds.a.fdim.values={1:length(cfg.time)};
    ds.a.fdim.labels={'time'};
    
    poststim=find(cfg.time>=0);
    ds=cosmo_slice(ds,ds.fa.time>poststim(1),2);
    
    cluster_nbrhood=cosmo_cluster_neighborhood(ds);

    opt=struct();
    opt.niter=cfg.nIter;
    opt.h0_mean=cfg.h0;
    
    if cfg.use_other_cluster_stat==1
    opt.cluster_stat=cfg.cluster_stat;%These two lines were added by Rico 
    opt.p_uncorrected=cfg.p_uncorrected;%so that you can choose the type of cluster statistic used
    %If commented out the default will be tfce again
    end
    
    tfce_ds=cosmo_montecarlo_cluster_stat(ds,cluster_nbrhood,opt);
    tfce_ds.samples=[zeros(1,poststim(1)),tfce_ds.samples];

end