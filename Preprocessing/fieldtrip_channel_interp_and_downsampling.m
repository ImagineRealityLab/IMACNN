% This script interpolates missing channels and downsamples EEG data using fieldtrip

clear all;
clc; 
clearvars;

% Specify parameters

ss=[2:5 7:12];%subject 1 and 6 were removed
subj_sessions=[1 10 10 10 10 4 10 10 10 10 10 10];

n_subj=numel(ss);
data_dir='C:\MATLAB\Individual Scene Imagery\Data\EEG_Data\fieldtrip preprocessing\';
output_dir='D:\UCL Individual Scene Imagery\EEG_data\fieldtrip preprocessing\';
file_name='individual_scene_imagery_timelock';
sampling_rate=20;

for subj_num=1:n_subj

    for sess_num=1:subj_sessions(ss(subj_num))

        % Load the session data for this subject

        load([data_dir,file_name,num2str(ss(subj_num)),'s',num2str(sess_num),'.mat']);

        % Get the raw data from one subject to get info about all channels in the
        % eeg data before channel removal

        if subj_num==1 && sess_num==1
            cfg=[];
            cfg.dataset='C:\MATLAB\Individual Scene Imagery\Data\EEG_Data\individual_scene_imagery0002s1.eeg';
            raw_data=ft_preprocessing(cfg);
            cfg=[];
            cfg.channel={'all','-photo'};
            raw_data=ft_selectdata(cfg,raw_data);
            full_labels=raw_data.label;
        end

        cfg=[];
        cfg.method='template';
        cfg.layout='easycap-M1.txt';
        neighbours=ft_prepare_neighbours(cfg,raw_data);

        % Interpolate missing channels by calculating the average of all
        % neighbour channels that are not missing

        cfg=[];
        cfg.neighbours=neighbours;
        cfg.method='average';
        timelock.elec=ft_read_sens('easycap-M1.txt');
        cfg.missingchannel={neighbours((~ismember({neighbours(:).label}, timelock.label))).label};
        timelock=ft_channelrepair(cfg,timelock);

        [~, full_labels_order]=sort(full_labels);
        temp=[string(timelock.label) [1:length(timelock.label)]'];
        temp_sorted=sortrows(temp);
        temp_sorted(full_labels_order,:)=temp_sorted;
        timelock.label=timelock.label(double(temp_sorted(:,2)));

        for trial_num=1:size(timelock.trial,1)
            timelock.trial(trial_num,:,:)=timelock.trial(trial_num,double(temp_sorted(:,2)),:);
        end

        % If you only want the posterior hemisphere, do this and change the filename:
        %cfg=[];
        %cfg.channel={'C*','T*','CP*','TP*','P*','O*'};
        %timelock=ft_selectdata(cfg,timelock);% This doesn't interpolate lost channels

        cfg=[];
        cfg.resamplefs=sampling_rate;
        timelock=ft_resampledata(cfg, timelock);

        cfg=[];
        cfg.latency=[-1 4];
        timelock=ft_selectdata(cfg,timelock);

        % Save the interpolated and downsampled data for this session

        save([output_dir,'individual_scene_imagery_timelock_mean_chan_interp_',num2str(sampling_rate),'hz',num2str(ss(subj_num)),'s',num2str(sess_num),'.mat'],'timelock');

    end

end
