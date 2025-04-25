%% Create RDMs across time using multiple sessions of EEG data
% This script calculates a representational dissimilarity matrix (RDM) at
% each time point within a participant's EEG data using cross-validated
% decoding accuracy between each stimulus pair as the distance measure. This
% is implemented using the CoSMoMVPA toolbox (https://www.cosmomvpa.org/).
% Classifiers are trained and tested on all sessions. In order to account
% for differences in EEG pattern distributions across sessions and boost
% the signal-to-noise ratio, the original feature space (EEG scalp voltage
% patterns) is mapped to a session-independent feature space using Maximum
% Independence Domain Adaptation (MIDA) as implemented in the MATLAB domain
% adaptation toolbox (de.mathworks.com/matlabcentral/fileexchange/
% 56704-a-domain-adaptation-toolbox) for each pairwise decoding. Then
% pseudotrials are created for each stimulus within each cross-validation
% subset by averaging together one trial of this stimulus from each session,
% randomly chosen from a fixed set of trials within each session that are
% only used to create averages for that particular subset. We repeat this
% for n permutations, using a different set of trials to create the
% pseudotrials, and then average the decoding accuracy across all permutations.
% In order to see how well the RDMs can differentiate between each stimulus
% based on the neural patterns, we average the decoding accuracies in the
% bottom triangle of the RDMs and plot them across time.
%
% Note that the CoSMoMVPA functions need FieldTrip(www.fieldtriptoolbox.org/)
% to run.

close all
clear
clc

%% Define parameters and pre-calculate variables

% provide the subject IDs and how many sessions of EEG data each part has
ss = [2:5, 7:12]; %only include subjects with 10 sessions
subj_sessions = [1, 10, 10, 10, 10, 4, 10, 10, 10, 10, 10, 10];

n_trials_per_sess = 432;
n_sess = 10;
subj_num = 1:length(ss);
n_chunks = 9;
n_pseudotrls_per_stim_per_chunk = 11;
n_stim = 16;
n_trials_to_avg_per_chunk = 3;

% how many permutations to use for pseudotrial averages during pairwise
% decoding
n_perms = 1;

% predefine the MIDA hyperparameters

param.m = 8; % number of components for final feature space
param.gamma = 0.5;
param.mu = 0.01;
%param.bSmida = false;
%param.ftAugType = 1;
%param.kernName = 'poly';
%param.KerSigma = 20;

% output directory of the RDMs and pairwise decoding and file name
output_dir = 'C:\MATLAB\Individual Scene Imagery\Results\time\RSA\';
file_name_prefix = ['RDMs_pair_acc_time_mult_sess_mida_8_comp_05_gamma_001_mu_d_trl_avg_chunk_', num2str(n_perms), '_perm'];

% activate fieldtrip functions used by CoSMo and make number generation
% always different on each execution

ft_defaults;
rng('shuffle');

% pre-calculate variables used for MIDA and trial averaging outside of
% the loop for optimization

domainFtAll = [];
for sess_num = 1:n_sess % for each session
    temp = zeros(n_trials_per_sess, n_sess);
    temp(:, sess_num) = ones(n_trials_per_sess, 1);
    domainFtAll = [domainFtAll; temp];
end % for each session
maLabeled = true(n_trials_per_sess*n_sess, 1);

% in order to speed up the pseudotrial averaging, we use index vectors
% that indicate what session, target and chunk a given trial belongs to
% and what pseudotrial it will be used for

n_pseudotrls_per_stim_pair = 2 * n_pseudotrls_per_stim_per_chunk * n_chunks;
n_trls_to_avg_per_stim = n_pseudotrls_per_stim_per_chunk * n_sess;

sess_vec = repmat([1:n_sess]', n_pseudotrls_per_stim_pair, 1);
target_vec = repmat(repelem([1:2]', n_pseudotrls_per_stim_per_chunk*n_sess), n_chunks, 1);
chunk_vec = repelem([1:n_chunks]', 2*n_trls_to_avg_per_stim, 1);
pseudotrial_vec = repelem([1:n_pseudotrls_per_stim_pair]', n_sess);
trl_per_sess_for_pseudotrls_id = repmat(repelem([1:n_trials_to_avg_per_chunk]', 18, 1), n_sess, 1);
trial_mean_chunk_vec = repelem([1:n_chunks]', 2*n_pseudotrls_per_stim_per_chunk);
trial_mean_feature_vec = ones(1, param.m);

n_trials_to_avg = n_pseudotrls_per_stim_pair * n_sess;

% pre-calculate target IDs for the decoding dataset that will contain the
% pseudotrials
pseudotrial_target_id = {};
for t1 = 1:n_stim
    for t2 = 1:n_stim
        if t1 > t2
            pseudotrial_target_id{t1, t2} = repmat(repelem([t2, t1]', n_pseudotrls_per_stim_per_chunk), n_chunks, 1);
        end
    end
end

% pre-calculate the permutations

% initialize an empty array to store unique permutations
unique_permutations = zeros(n_trials_to_avg, n_perms);

for perm = 1:n_perms

    % create a vector of random chunk-specifc trial IDs for the trial
    % averages
    shuffled_vector = randi(n_trials_to_avg_per_chunk, n_trials_to_avg, 1);

    if perm >= 2

        % ensure the random vector is a unique permutation
        while any(all(unique_permutations(:, 1:perm-1) == shuffled_vector, 1))
            shuffled_vector = randi(n_trials_to_avg_per_chunk, n_trials_to_avg, 1);
        end

    end

    % store the unique permutation
    unique_permutations(:, perm) = shuffled_vector;

end

%% Conduct the analysis

for s = subj_num % for all subjects

    % stack data from each session

    for sess_num = 1:subj_sessions(ss(s)) % for all sessions

        load(['C:\MATLAB\Individual Scene Imagery\Data\EEG_Data\fieldtrip preprocessing\individual_scene_imagery_timelock_mean_chan_interp_100_hz', num2str(ss(s)), 's', num2str(sess_num)]);

        % convert to cosmo
        ds_sess = cosmo_meeg_dataset(timelock);

        % get time data
        if sess_num == 1
            res.time = timelock.time;
        end

        clear timelock

        % add targets
        ds_sess.sa.targets = ds_sess.sa.trialinfo(:, 1);

        % add sessions
        ds_sess.sa.sessions = repelem(sess_num, n_trials_per_sess)';

        % add chunks within each session to balance sessions within
        % each chunk

        ds_sess.sa.chunks = [1:length(ds_sess.sa.targets)]';
        ds_sess.sa.chunks = cosmo_chunkize(ds_sess, n_chunks);

        % stack session datasets
        if sess_num == 1
            ds = ds_sess;
        else
            ds = cosmo_stack({ds, ds_sess}, 1);
        end

    end % sessions

    clear ds_sess

    % create RDMs

    % preallocate RDM array
    res.diss = zeros(n_perms, n_stim, n_stim, length(res.time));
    for perm = 1:n_perms % for all permutations

        for tp = 1:max(ds.fa.time) % for all time points

            disp(['Subject ', num2str(s), ' of ', num2str(length(ss)), '. Permutation ', num2str(perm), ' of ', num2str(n_perms), '. Time point ', num2str(tp), ' of ', num2str(max(ds.fa.time)), '.']);

            % get data subset at this time point
            ds_tp = cosmo_slice(ds, ismember(ds.fa.time, tp), 2, false);

            for t1 = unique(ds_tp.sa.targets)' % for target dimension 1

                for t2 = unique(ds_tp.sa.targets)' % for target dimension 2

                    if t1 > t2 % only do this in bottom triangle

                        % get data subset for this stimulus pair
                        ds_class = cosmo_slice(ds_tp, ismember(ds_tp.sa.targets, [t1, t2]), 1, false);

                        % conduct (s)MIDA
                        domainFtAll_pair = domainFtAll(ismember(ds_tp.sa.targets, [t1, t2]), :);
                        maLabeled_pair = maLabeled(ismember(ds_tp.sa.targets, [t1, t2]));

                        [ftAllNew, ~] = ftTrans_mida(ds_class.samples, domainFtAll_pair, ds_class.sa.targets, maLabeled_pair, param);

                        % create pseudotrials

                        % create matrix of indices that characterize all
                        % trials within this subset

                        indices = [ds_class.sa.chunks, ds_class.sa.sessions, ds_class.sa.targets, trl_per_sess_for_pseudotrls_id];

                        target_vec = repmat(repelem([t2, t1]', n_trls_to_avg_per_stim), n_chunks, 1);

                        % create a matrix that contains the pseudotrial,
                        % chunk, session and target indices for each trial
                        % that needs to be averaged together, also apply
                        % permutation by adding which of the n trials that
                        % are assigned to each chunk for averaging should
                        % be used for the pseudotrial in this permutation

                        trials_to_avg_id = [];
                        trials_to_avg_id = [pseudotrial_vec, chunk_vec, sess_vec, target_vec, unique_permutations(:, perm)];

                        % get the trial index of these trials and append
                        % it to our index matrix

                        [intersecting_rows, ~, index_B] = intersect(trials_to_avg_id(:, 2:5), indices, 'rows', 'stable');
                        trials_to_avg_id_trial_id = [intersecting_rows, index_B];

                        % go through all trials that will be averaged and
                        % add the corresponding trial index to each trial

                        for i = 1:length(trials_to_avg_id_trial_id)
                            row_match_logical = sum(trials_to_avg_id(:, 2:5) == trials_to_avg_id_trial_id(i, 1:4), 2) == 4;
                            trials_to_avg_id(row_match_logical, 6) = trials_to_avg_id_trial_id(i, 5);
                        end

                        % append copies of the full subset for this
                        % stimulus pair underneath each other for
                        % as many times as we have trials to average,
                        % then scale the trial indices so that they
                        % index the trial of interest in each of the
                        % respective repetitions

                        sample_rep = repmat(ftAllNew, n_trials_to_avg, 1);
                        trials_to_avg_id(:, 6) = trials_to_avg_id(:, 6) + [0:540:1069200 - 540]';

                        % get actual data of the trials
                        trials_to_avg = sample_rep(trials_to_avg_id(:, 6), :);
                        sample_rep = [];

                        % average trials
                        ds_class.samples = [];
                        for trial_num = 1:n_pseudotrls_per_stim_pair
                            ds_class.samples(trial_num, :) = mean(trials_to_avg(trials_to_avg_id(:, 1) == trial_num, :));
                        end

                        % add chunk and target information
                        ds_class.sa.chunks = trial_mean_chunk_vec;
                        ds_class.sa.targets = pseudotrial_target_id{t1, t2};

                        % decoding settings
                        args.classifier = @cosmo_classify_lda;
                        args.partitions = cosmo_nchoosek_partitioner(ds_class, 1);
                        args.check_partitions = false;

                        % run classification and store accuracy
                        acc = cosmo_crossvalidation_measure(ds_class, args);
                        res.diss(perm, t1, t2, tp) = acc.samples;
                        res.diss(perm, t2, t1, tp) = res.diss(perm, t1, t2, tp);

                    end

                end % target dim 2

            end % target dim 1

        end % time points

    end % permutations

    if n_perms > 1
        % average across permutations
        res.diss = squeeze(mean(res.diss));
    else
        % get rid of singleton permutation dimension
        res.diss = squeeze(res.diss);
    end

    % save the RDMs
    save([output_dir, file_name_prefix, '_', num2str(ss(s))], 'res');

end % subjects

%% Average bottom triangle

for s = subj_num % for each subject

    load([output_dir, file_name_prefix, '_', num2str(ss(s))]);

    for tp = 1:size(res.diss, 3) % for each time point
        RDM_vec = squareform(squeeze(res.diss(:, :, tp)));
        RDM_mean_matrix(s, tp) = mean(RDM_vec);
    end % time points

end % subjects

% save the mean pairwise decoding
save([output_dir, file_name_prefix, '_', 'mean_pair_dec'], 'RDM_mean_matrix');

%% Plot the mean pairwise decoding

figure();
set(gcf, 'Color', 'w')
for s = 1:size(RDM_mean_matrix, 1) % for each subject
    sgtitle(['mult sess mean pairw time-resolved scene dec mida 8 comp 0.5 gam 0.01 mu dan chunk avg ', num2str(n_perms), ' perm'])
    subplot(1, length(subj_num), s);
    plot(res.time, RDM_mean_matrix(s, :)*100);
    xline(0, 'k-');
    yline(0.5*100, '--');
    ylabel('accuracy %');
    xlabel('time (s)');
    ylim([0.49, 0.65]*100);
    xlim([min(res.time), max(res.time) + 0.1]);
    title(['Subject ', num2str(ss(s))]);
end % subjects

set(gcf, 'WindowState', 'maximized');

% save the figure using export_fig
export_fig(fullfile(output_dir, 'mean_pair_dec_plot_1_perm_smida'), '-dtiff', '-r300');
