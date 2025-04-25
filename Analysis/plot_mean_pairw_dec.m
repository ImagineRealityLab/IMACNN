%% Plot the mean pairwise decoding across time

clear
clc
close all

%% Define parameters and load data

analysis_color = [91, 50, 86];
output_dir = 'C:\MATLAB\Individual Scene Imagery\Results\time\Decoding\';

load('D:\UCL Individual Scene Imagery\Results\time\RSA\RDMs_pair_acc_mult_sess_time_train_est_smida_8_comp_05_gamma_001_mu_d_trl_avg_chunk_1_perm_mean_pair_dec');

%% Conduct TFCE/cluster permutation test at each time point

data = [];
cfg = [];

data = RDM_mean_matrix;

cfg.time = -0.9775:0.05:4.025;
cfg.h0 = 0.5;
cfg.nIter = 10000;
cfg.use_other_cluster_stat = 0;

% run TFCE/cluster permutation test
clusterstat_ds = EEG_clusterstats(cfg, data);

%% Plot results

% get data
dat.acc = data * 100;
dat.time = cfg.time;

% create figure
hf = figure('position', [1, 1, 2400, 600], 'unit', 'centimeters');

set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20, 12], 'PaperPosition', [0, 0, 24, 11]);
colordef white
set(0, 'DefaultAxesFontName', 'Helvetica')
set(0, 'DefaultTextFontname', 'Helvetica')
set(0, 'DefaultAxesFontSize', 19) 
set(0, 'DefaultTextFontSize', 22) 
set(gcf, 'Color', 'w')

% add horizontal and vertical lines
xline(0, 'k-', 'LineWidth', 2)
hold on
h = line([min(dat.time), max(dat.time)], [0.5, 0.5]*100);
set(h, 'color', 'k');
set(h, 'linewidth', 2);
set(h, 'linestyle', '--');

% plot data
x = dat.time;
y = mean(dat.acc, 1);
e = std(dat.acc, 0, 1) / sqrt(size(dat.acc, 1));
cm = analysis_color ./ 255;
h = boundedline(x, y, e, 'cmap', cm, 'alpha');
set(h, 'linewidth', 2);

% format axes
axis tight
xlabel('time (s)');
ylabel('% accuracy');
set(gca, 'linewidth', 3);
set(gca, 'box', 'off');

set(gca, 'xlim', [dat.time(1), dat.time(end)]);
set(gca, 'Layer', 'top');
set(gca, 'ylim', [0.49, 0.56]*100);

% plot significance markers

pvalues = normcdf(clusterstat_ds.samples, 'upper');
for tp = 1:length(x)
    if pvalues(tp) < 0.05
        h = plot(x(tp), 0.55*100);

        set(h, 'marker', 's');
        set(h, 'markersize', 7);
        set(h, 'markeredgecolor', cm);
        set(h, 'markerfacecolor', cm);
        set(h, 'linewidth', 1);

    elseif pvalues(tp) >= 0.05 && pvalues(tp) <= 0.1

        h = plot(x(tp), 0.55*100);

        set(h, 'marker', '+');
        set(h, 'markersize', 10);
        set(h, 'markeredgecolor', cm);
        set(h, 'markerfacecolor', cm);
        set(h, 'linewidth', 3);

    end 
end

% set title
title('mean pairwise scene decoding across time (training set estimate sMIDA, 1-permutation-pseudotrials)')

%% Save plot

cd(output_dir);

% save the figure using export_fig
set(gcf, 'WindowState', 'maximized');
export_fig(fullfile(output_dir, 'Imagery_time-resolved_mean_pairwise_decoding_1_perm_train_est_smida'), '-dtiff', '-r300');
