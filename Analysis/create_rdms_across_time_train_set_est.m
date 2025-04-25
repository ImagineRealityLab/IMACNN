%% Create RDMs across time using multiple sessions of EEG data
% This is an alternate approach to the main pipeline that 
% creates RDMs by first estimating the session-independent 
% feature space in the training set, and using the projection  
% matrix that was used for this mapping to estimate map the 
% the test set to this feature space

clear
clc
close all

%% Define parameters and pre-calculate variables

ss = [2:5 7:12];
subj_sessions = [1, 10, 10, 10, 10, 4, 10, 10, 10, 10, 10, 10];

subj_num = 1:length(ss);
n_perms = 1;
n_stimuli = 16;
param.m = 8;
param.gamma = 0.5;
param.mu = 0.01;
%param.bSmida=false;
%param.ftAugType=1;
%param.kernName='poly';
%param.KerSigma=20;

output_dir = 'D:\UCL Individual Scene Imagery\Results\time\RSA\';
file_name_prefix = 'RDMs_pair_acc_mult_sess_time_train_est_smida_8_comp_05_gamma_001_mu_d_trl_avg_chunk_1_perm';

ft_defaults
rng('shuffle');

% calculate variables outside of the loop for optimization

domainFtAll = [];
for sess_num = 1:10
    temp = zeros(432, 10);
    temp(:, sess_num) = ones(432, 1);
    domainFtAll = [domainFtAll; temp];
end
maLabeled = true(432*10, 1);

sess_vec = repmat([1:10]', 198, 1);
target_vec = repmat(repelem([1:2]', 110), 9, 1);
chunk_vec = repelem([1:9]', 220, 1);
pseudotrial_vec = repelem([1:198]', 10);
target_trl_count_per_ch_per_sess = repmat(repelem([1:3]', 18, 1), 10, 1);
trial_mean_chunk_vec = repelem([1:9]', 22);
trial_mean_feature_vec = ones(1, param.m);

n_pairw_comp = (n_stimuli * n_stimuli - n_stimuli) / 2;

pseudotrial_target_id = {};
for t1 = 1:n_stimuli
    for t2 = 1:n_stimuli
        if t1 > t2
            pseudotrial_target_id{t1, t2} = repmat(repelem([t2, t1]', 11), 9, 1);
        end
    end
end

% pre-calculate the permutations

% initialize an empty array to store unique permutations
unique_permutations = zeros(1980, n_perms);

for perm = 1:n_perms

    % create a vector of random chunk-specifc trial IDs for the trial
    % averages
    shuffled_vector = randi(3, 1980, 1);

    if perm >= 2

        % ensure the random vector is a unique permutation
        while any(all(unique_permutations(:, 1:perm-1) == shuffled_vector, 1))
            shuffled_vector = randi(3, 1980, 1);
        end

    end

    % store the unique permutation
    unique_permutations(:, perm) = shuffled_vector;

end

%% Conduct analysis

for s = subj_num

    % stack data across sessions

    for sess_num = 1:subj_sessions(ss(s))

        load(['D:\UCL Individual Scene Imagery\EEG_data\fieldtrip preprocessing\individual_scene_imagery_timelock_mean_chan_interp_20hz', num2str(ss(s)), 's', num2str(sess_num)]);

        % convert to cosmo
        ds_sess = cosmo_meeg_dataset(timelock);

        if sess_num==1
            res.time = timelock.time;
        end

        clear timelock

        % add targets
        ds_sess.sa.targets = ds_sess.sa.trialinfo(:, 1);

        % add session number
        ds_sess.sa.sessions = repelem(sess_num, 432)';

        % add chunks
        nch = 9;

        ds_sess.sa.chunks = [1:length(ds_sess.sa.targets)]';
        ds_sess.sa.chunks = cosmo_chunkize(ds_sess, nch);

        % stack session data sets
        if sess_num == 1
            ds = ds_sess;
        else
            ds = cosmo_stack({ds, ds_sess}, 1);
        end

    end

    clear ds_sess

    n_time_points = max(ds.fa.time);
    
    % create RDMs

    res.diss = zeros(n_perms, n_stimuli, n_stimuli, length(res.time));
    all_data_sets = {};
    all_test_ids = {};

    disp('Mapping feature space...')
    
    tic
    % estimate feature space for every cross-validation split
    parfor tp = 1:n_time_points
 
        ds_tp = cosmo_slice(ds, ismember(ds.fa.time, tp), 2, false);
        
        % go through all target pairs in the bottom triangle

        for t1 = 1:n_stimuli
            for t2 = 1:n_stimuli
                if t1 > t2

                    % get data
                    ds_class = cosmo_slice(ds_tp, ismember(ds_tp.sa.targets, [t1, t2]), 1, false);

                    % pre-calculate (s)MIDA variables
                    domainFtAll_pair = domainFtAll(ismember(ds_tp.sa.targets, [t1, t2]), :);
                    maLabeled_pair = maLabeled(ismember(ds_tp.sa.targets, [t1, t2]));

                    % split data into train and test sets
                    partitions = cosmo_nchoosek_partitioner(ds_class, 1);

                    % estimate (s)MIDA space for each cross-validation fold
                    for fold_num = 1:nch

                        ds_class_fold = ds_class;

                        % get train and test sets for this fold
                        train_id = partitions.train_indices{fold_num}(:);
                        test_id = partitions.test_indices{fold_num}(:);
                        train_set = ds_class_fold.samples(train_id, :);
                        test_set = ds_class_fold.samples(test_id, :);

                        % get vectors that indicate what domain (i.e. session)
                        % each sample in the train and test belongs to and
                        % how many labels samples have labels
                        domainFtAll_train = domainFtAll_pair(train_id, :);
                        maLabeled_train = maLabeled_pair(train_id);

                        domainFtAll_test = domainFtAll_pair(test_id, :);
                        maLabeled_test = maLabeled_pair(test_id);

                        % map feature space in the train set to (s)MIDA space
                        [ftAllNew_train, transMdl_train] = ftTrans_mida(train_set, domainFtAll_train, ds_class_fold.sa.targets(train_id), maLabeled_train, param);

                        % use the projection matrix used on the train set
                        % to map the test set to the (s)MIDA space
                        [ftAllNew_test] = ftTrans_mida_project(test_set, domainFtAll_test, train_set, domainFtAll_train, transMdl_train.W, param);

                        % save the mapped features in a cosmo data set

                        ds_class_fold.samples = [];
                        ds_class_fold.samples(train_id, :) = ftAllNew_train;
                        
                        ds_class_fold.samples(test_id, :) = ftAllNew_test;
                        ds_class_fold.samples=single(ds_class_fold.samples);

                        all_data_sets{tp, t1, t2, fold_num} = ds_class_fold;
                        all_test_ids{tp, t1, t2, fold_num} = test_id;
                    end

                end
            end
        end

    end
    
    clear ds
    
    disp('Done!')
 
    % reset parallel pool after second parfor loop
    delete(gcp('nocreate')); 
    parpool;
    
    % create a matrix of indices that indicate what permutation
    % should be used for each classifcation, what time point it
    % should be done on and what target pair should be predicted

    n_decodings = n_perms * n_time_points * n_pairw_comp;

    matrix = reshape(1:n_stimuli*n_stimuli, n_stimuli, n_stimuli);
    matrix(eye(n_stimuli) == 1) = 0;
    feature_id = repelem(1:n_time_points, n_pairw_comp)';

    id_mat = reshape(1:n_stimuli*n_stimuli, n_stimuli, n_stimuli);
    id_mat(eye(n_stimuli) == 1) = 0;
    tril_id_vec = squareform(id_mat)';
    [target1_id, target2_id] = ind2sub([n_stimuli, n_stimuli], tril_id_vec);

    perm_id = repelem(1:n_perms, n_time_points*n_pairw_comp)';
    feature_id = repmat(feature_id, n_perms, 1);
    target1_id = repmat(target1_id, n_perms*n_time_points, 1);
    target2_id = repmat(target2_id, n_perms*n_time_points, 1);
    dec_id = [perm_id, feature_id, target1_id, target2_id];

    % run the decoding
    
    disp('Running the decoding...')

    % pre-allocate accuracy vector
    dec_acc = zeros(n_decodings, 1);
    
    numWorkers = max(1, gcp('nocreate').NumWorkers - 2); % reduce workers by 2, ensuring at least 1 remains
    delete(gcp('nocreate'));  % close the existing pool if it's open
    parpool(numWorkers);

    tic
    parfor dec_num = 1:n_decodings

        % train and test classifier for each fold
        fold_acc = zeros(nch, 1);
        for fold_num = 1:nch

            % get mapped data set for this fold
            ds_class_fold = all_data_sets{dec_id(dec_num, 2), dec_id(dec_num, 3), dec_id(dec_num, 4), fold_num};

            % get the chunk ID of the test set in this fold
            test_id = all_test_ids{dec_id(dec_num, 2), dec_id(dec_num, 3), dec_id(dec_num, 4), fold_num};
            test_chunk_id = unique(ds_class_fold.sa.chunks(test_id));

            % create matrix that indicates which chunk, session, target and
            % what subset for the pseudo-trial averaging each trial belongs to
            indices = [ds_class_fold.sa.chunks, ds_class_fold.sa.sessions, ds_class_fold.sa.targets, target_trl_count_per_ch_per_sess];

            % create a matrix that indicates which chunk, session, target
            % and pseudo-trial subset the trials we need for our averaging
            % belong to, also apply permutation by picking which trial in
            % the chunk-specific subset should be used for averaging

            target_vec = repmat(repelem([target2_id(dec_num), target1_id(dec_num)]', 110), 9, 1);

            trials_to_avg_id = [];
            trials_to_avg_id = [pseudotrial_vec, chunk_vec, sess_vec, target_vec, unique_permutations(:, perm)];

            % get the trial indices of those trials
            [intersecting_rows, ~, index_B] = intersect(trials_to_avg_id(:, 2:5), indices, 'rows', 'stable');
            trials_to_avg_id_trial_id = [intersecting_rows, index_B];

            for i = 1:length(trials_to_avg_id_trial_id)
                row_match_logical = sum(trials_to_avg_id(:, 2:5) == trials_to_avg_id_trial_id(i, 1:4), 2) == 4;
                trials_to_avg_id(row_match_logical, 6) = trials_to_avg_id_trial_id(i, 5);
            end

            % get trial data for averaging
            trials_to_avg_id(:, 6) = trials_to_avg_id(:, 6) + [0:540:1069200 - 540]';

            sample_rep = repmat(ds_class_fold.samples, 1980, 1);
            indices_rep = repmat(indices, 1980, 1);
            trials_to_avg = sample_rep(trials_to_avg_id(:, 6), :);

            sample_rep = [];

            % average trials
            ds_class_fold.samples = [];
            for trial_num = 1:198
                ds_class_fold.samples(trial_num, :) = mean(trials_to_avg(trials_to_avg_id(:, 1) == trial_num, :));
            end

            % add chunk and target indices for the data set with pseudo-trials
            ds_class_fold.sa.chunks = trial_mean_chunk_vec;
            ds_class_fold.sa.targets = pseudotrial_target_id{target1_id(dec_num), target2_id(dec_num)};

            % get train and test data for classification
            train_set_avg = ds_class_fold.samples(ds_class_fold.sa.chunks ~= test_chunk_id);
            targets_train_set_avg = ds_class_fold.sa.targets(ds_class_fold.sa.chunks ~= test_chunk_id);

            test_set_avg = ds_class_fold.samples(ds_class_fold.sa.chunks == test_chunk_id);
            targets_test_set_avg = ds_class_fold.sa.targets(ds_class_fold.sa.chunks == test_chunk_id);

            % train and test the classifier and calculate fold accuracy
            predicted = cosmo_classify_lda(train_set_avg, targets_train_set_avg, test_set_avg);
            fold_acc(fold_num) = mean(predicted == targets_test_set_avg);

        end

        % calculate overall accuracy
        dec_acc(dec_num) = squeeze(mean(fold_acc));

    end
    
    numWorkers = max(1, gcp('nocreate').NumWorkers+2);
    delete(gcp('nocreate')); 
    parpool(numWorkers);
    
    toc
    disp('Done!')

    % get the RDMs
    for perm=1:n_perms
        for tp = 1:n_time_points
            perm_tp_logical=and(dec_id(:,1)==perm,dec_id(:,2)==tp);
            RDM_vec = dec_acc(perm_tp_logical);
            res.diss(perm,:,:,tp) = squareform(RDM_vec);
        end
    end

    if n_perms > 1
        % average across permutations
        res.diss = squeeze(mean(res.diss));
    else
        % get rid of singleton permutation dimension
        res.diss = squeeze(res.diss);
    end

    % save the RDMs
    save([output_dir, file_name_prefix, '_', num2str(ss(s))], 'res');

end

%% Average the bottom triangle

for s = subj_num

    load([output_dir, file_name_prefix, '_', num2str(ss(s))]);

    for tp = 1:size(res.diss, 3)
        RDM_mean_matrix(s, tp) = mean(squareform(squeeze(res.diss(:, :, tp))));
    end

end

%% Plot the mean pairwise decoding

figure();
set(gcf, 'Color', 'w')
for s = 1:size(RDM_mean_matrix, 1)
    sgtitle('mult sess time-resolved mean pairw scene dec train set est smida 8 comp 0.5 gam 0.01 mu (not optimized) dan chunk avg 1 perm')
    subplot(1, 10, s);
    plot(res.time, RDM_mean_matrix(s, :)*100);
    yline(0.5*100, '--');
    ylabel('accuracy %');
    xlabel('time (s)');
    ylim([0.46, 0.64]*100);
    xlim([min(res.time), max(res.time)]);
    title(['Subject ', num2str(ss(s))]);
end

% save the mean pairwise decoding
save([output_dir, file_name_prefix, '_', 'mean_pair_dec'], 'RDM_mean_matrix');