%% Predict imagery RDMs at every time point with DNN layer group RDMs
%% using a ridge regression

clearvars
clc
close all

%% Define parameters and prepare data

ss = [2:5, 7:12];
subj_num = 1:length(ss);

output_dir = 'C:\MATLAB\Individual Scene Imagery\Results\DNN\DNN correlations\';
file_name = 'Imagery_RDM_DNN_RDM_ridge_coefficients_across_time';

analysis_colors = [250, 237, 49; 32, 115, 23; 150, 31, 5; 224, 123, 57];

smooth_data = true;
smoothing_window = 201;

lambda = 1;

% load DNN RDMs
load('C:\MATLAB\Individual Scene Imagery\Results\DNN\vgg16_places365_RDM_100_img_high_var_set_SD_2_cfg_3_no_pos_prompts.mat')

% vectorize them
n_layers = length(rdm_avg);
for layer = 1:n_layers
    dnn_rdm_vec_mat(:, layer) = squeeze(squareform(rdm_avg{layer}));
end

% average layer RDM vectors into three layer groups: early,
% intermediate and fully connected layers

early_RDM_vec = mean(dnn_rdm_vec_mat(:, 1:4), 2);
intermediate_RDM_vec = mean(dnn_rdm_vec_mat(:, 5:13), 2);
FC_RDM_vec = mean(dnn_rdm_vec_mat(:, 14:16), 2);
layer_groups(:, 1) = early_RDM_vec;
layer_groups(:, 2) = intermediate_RDM_vec;
layer_groups(:, 3) = FC_RDM_vec;

%% Predict the imagery RDM of each participant across time
%% with the layer group rdms

for s = subj_num

    % get RDMs
    load(['C:\MATLAB\Individual Scene Imagery\Results\time\RSA\RDMs_pair_acc_time_mult_sess_smida_8_comp_05_gamma_001_mu_d_trl_avg_chunk_5_perm_', num2str(ss(s))]);

    if smooth_data
        res.diss = smoothdata(res.diss, 3, ["movmean"], smoothing_window);
    end

    for tp = 1:length(res.time)

        % convert DNN layer RDMs to a matrix of predictors
        X = layer_groups;

        % neural RDMs for the current participant
        current_RDM = squeeze(res.diss(:, :, tp));
        y = squareform(current_RDM)';

        % fit ridge regression
        b = ridge(y, X, lambda, 0);

        % save the ridge coefficients for all predictors
        ridge_coeff(s, :, tp) = b(2:end);

    end
end

% define parameters for TFCE/cluster permutation test

cfg = [];
cfg.time = res.time;
cfg.h0 = 0;
cfg.nIter = 10000;
cfg.use_other_cluster_stat = 0;

if cfg.use_other_cluster_stat == 1
    cfg.cluster_stat = 'maxsize';
    cfg.p_uncorrected = 0.005;
end

%% Plot the data

for layer_group = 1:size(layer_groups, 2)

    dat.betas = squeeze(ridge_coeff(:, layer_group, :));

    dat.time = res.time;

    if layer_group == 1

        hf = figure('Position', [1, 1, 2400, 600], 'unit', 'centimeters');

        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20, 12], 'PaperPosition', [0, 0, 24, 21]);
        colordef white
        set(0, 'DefaultAxesFontName', 'Helvetica')
        set(0, 'DefaultTextFontname', 'Helvetica')
        set(0, 'DefaultAxesFontSize', 15)
        set(0, 'DefaultTextFontSize', 15)
        set(gcf, 'Color', 'w')

        yline(0, 'color', 'k', 'LineWidth', 2, 'LineStyle', '--');
        xline(0, 'color', 'k', 'LineWidth', 2);

        title_text = 'Predicting imagery RDMs across time with DNN layer group RDMs using a ridge regression';

        smooth_time_window = (res.time(2) - res.time(1)) * smoothing_window * 1000;

        if smooth_data
            title([title_text, ' (', num2str(smooth_time_window), 'ms smoothing window)']);
        elseif ~smooth_data
            title(title_text);
        end

    end

    hold on
    x = dat.time;
    y = mean(dat.betas, 1);
    e = std(dat.betas, 0, 1) / sqrt(size(dat.betas, 1));
    cm = analysis_colors(layer_group, :) ./ 255;
    h(layer_group) = boundedline(x, y, e, 'cmap', cm, 'alpha');
    set(h(layer_group), 'linewidth', 3);

    axis tight
    xlabel('time (s)');
    ylabel('ridge coefficients');

    set(gca, 'linewidth', 3);
    set(gca, 'xlim', [dat.time(1), dat.time(end)]);
    set(gca, 'YLim', [-0.2, 0.4]);

    data = squeeze(ridge_coeff(:, layer_group, :));

    % run TFCE/cluster permutation test
    clusterstat_ds = EEG_clusterstats(cfg, data);

    pvalues = normcdf(clusterstat_ds.samples, 'upper');
    for tp = 1:length(x)
        if pvalues(tp) < 0.05
            marker_h = plot(x(tp), 0.3);

            set(marker_h, 'marker', 's');
            set(marker_h, 'markersize', 7);
            set(marker_h, 'markeredgecolor', cm);
            set(marker_h, 'markerfacecolor', cm);
            set(marker_h, 'linewidth', 1);

        elseif pvalues(tp) >= 0.05 && pvalues(tp) <= 0.1

            marker_h = plot(x(tp), 0.3);

            set(marker_h, 'marker', '+');
            set(marker_h, 'markersize', 6);
            set(marker_h, 'markeredgecolor', cm);
            set(marker_h, 'markerfacecolor', cm);
            set(marker_h, 'linewidth', 1);

        end
    end

end

legend([h(1), h(2), h(3)], 'early layers', 'intermediate layers', 'fully connected layers', 'location', 'eastoutside');

% save the figure using export_fig

set(gcf, 'WindowState', 'maximized');

if smooth_data
    file_name = [file_name, '_', num2str(smooth_time_window), '_smoothing']
end

export_fig(fullfile(output_dir, file_name), '-dtiff', '-r300');
