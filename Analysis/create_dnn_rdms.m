%% Create RDMs averaged across all AI-generated image sets for 
%% each DNN layer

clc
clear
close all

% !! This code assumes the order of the images to follow the Automatic1111
% format, in which all generated images of a scene are grouped together (so
% the folder first contains all images of scene 1, then all images of
% scene 2... etc.). If you don't do this, the RDMs will not be calculated
% correctly.

%% Define parameters, pre-calculate variables and prepare data

img_dir = 'C:\MATLAB\Individual Scene Imagery\Scene Images\high variance set 100 img SD 2 0 cfg 3 no pos prompts\';
model_dir = 'C:\MATLAB\Individual Scene Imagery\VGG16places_forRico\';
output_dir = 'C:\MATLAB\Individual Scene Imagery\Results\DNN\';
n_scenes = 16;
n_img_sets = 100;
n_subsets = 4; 
n_layers = 16;
n_img = n_scenes * n_img_sets;
n_img_per_subset = n_img / n_subsets;
n_img_per_scene_per_subset = n_img_per_subset / n_scenes;

% create an index for the images that will be assigned to each subset, so
% that an equal amount of images is put into each subset and there are no
% repetitions

subset_idx_temp = repmat(1:n_img_per_scene_per_subset, n_subsets, n_scenes);
offset1 = repelem([0:n_img_sets:(n_img - n_img_sets)], n_img_per_scene_per_subset);
subset_idx_offset1 = repmat(offset1, n_subsets, 1);
offset2 = [0:n_img_per_scene_per_subset:(n_img_sets - n_img_per_scene_per_subset)]';
subset_idx_offset2 = repelem(offset2, 1, n_img_per_subset);
subset_idx = subset_idx_temp + subset_idx_offset1 + subset_idx_offset2;

% reorder these indices so that the final output will have RDMs of the
% different image sets along the diagonal

subset_idx_reordered = [];
for img_set_num = 1:n_img_per_scene_per_subset
    temp_idx = subset_idx(:, img_set_num:n_img_per_scene_per_subset:n_img_per_subset);
    subset_idx_reordered = [subset_idx_reordered, temp_idx];
end

% load a model
load([model_dir, 'vggnet16_places365.mat']);

% get the conv and fc layers
lx = 0;
for layer = 1:length(net.Layers)
    if strfind(net.Layers(layer).Name, 'conv') == 1
        lx = lx + 1;
        my_layers(lx) = layer;
        layer_names{lx} = net.Layers(layer).Name;
    elseif strfind(net.Layers(layer).Name, 'fc') == 1
        lx = lx + 1;
        my_layers(lx) = layer;
        layer_names{lx} = net.Layers(layer).Name;
    end
end

% get all possible images
allImg = dir([img_dir, '*.png']);

%% Iterate through all layers and subsets and create DNN RDMs for all 
%% image sets in each subset

rdm = zeros(n_layers, n_subsets, n_img_per_subset, n_img_per_subset);
for layer = 1:length(layer_names)

    for subset_num = 1:n_subsets

        % create a subset that contains an equal amount of images from each
        % scene. We create subsets to avoid running out of memory.

        allImg_subset = allImg(subset_idx_reordered(subset_num, :));

        % extract activations
        for i = 1:n_img_per_subset

            display(['Computing - Image #', num2str(i)]);

            % image filename
            imgName = [allImg_subset(i).folder, '/', allImg_subset(i).name];

            % load the image
            im_ = imread(imgName);
            im_ = single(im_);
            im_ = imresize(im_, net.Layers(1).InputSize(1:2));

            % run the DNN
            x = activations(net, im_, layer_names{layer});
            x = x(:);
            if i == 1
                xx = zeros(length(x), n_img_per_subset);
            end
            xx(:, i) = x;

        end
        
        % create RDMs for all image sets in this subset
        rdm(layer, subset_num, :, :) = 1 - corr(xx);

    end

end

%% Average RDMs in each layer across image sets and subsets

idx = repelem(1:n_img_per_scene_per_subset, 1, n_scenes);
rdm = squeeze(mean(rdm, 2));
for layer = 1:size(rdm, 1)

    for img_set_num = 1:n_img_per_scene_per_subset
        rdm_array(img_set_num, :, :) = rdm(layer, img_set_num == idx, img_set_num == idx);
    end

    rdm_avg{layer} = squeeze(mean(rdm_array));

end

% save the averaged DNN RDMs
save([output_dir, 'vgg16_places365_RDMs_100_img_high_var_set_SD_2_cfg_3_no_pos_prompts'], 'rdm_avg');