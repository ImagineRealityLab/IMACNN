# IMACNN
 
This repository contains the code for the temporal analysis of the project "Using generative AI, CNNs and extensive individual sampling to characterize the visual features encoded in alpha rhythms during imagery" which Rico conducted during his lab visit in 2025.

**Task & Analysis**

10 participants imagined 16 natural scenes according to short prompts in the EEG for 10 sessions (4320 trials per participant).

In the main analysis, we first create representational dissimilarity matrices (RDMs) at each time point by training classifiers to discriminate between each pair of the 16 imagined scenes using a session-independent feature space that is mapped to from the evoked spatial voltage patterns space in the EEG. In order to assess the general information content regarding these imagined scenes across time, we assessed their mean pairwise discriminability by averaging the pairwise decoding accuracies in the bottom triangle of these RDMs at every time point. In the final part of the analysis, we investigated in what order visual features of varying levels of abstraction are reactivated during imagery. To that end, we correlated the imagery RDMs at each time point with average RDMs in early, intermediate and late layers of a CNN that was fed with AI-generated images as an estimate of the imagined visual contents (100 images per scene prompt). 

|  Step  | Procedure & Script                |
|-------|-----------------|
| 1.1.  | Preprocessing: preprocess_EEG_individual_scene_imagery.m|
| 1.2.  | Channel interpolation and downsampling: fieldtrip_channel_interp_and_downsampling.m |
| 2.1.  | EEG RDM creation & mean pairwise decoding: create_rdms_across_time.m     |
| 2.2.  | DNN RDM creation:      |
| 3.1.  | Conduct statistical test and plot mean pairwise decoding: plot_mean_pairwise_decoding     |
| 3.2.  | Correlate EEG RDMs with DNN RDMs, run statistical test and plot correlations: plot_dnn_temporal_corr     |

The preprocessing scripts and RDM creation script require [FieldTrip](https://www.fieldtriptoolbox.org/). The RDM creation script also requires [CosMoMVPA](https://www.cosmomvpa.org/) and the MATLAB [domain adaptation toolbox](https://de.mathworks.com/matlabcentral/fileexchange/56704-a-domain-adaptation-toolbox). The shaded error bars in the plots use the [boundedline function](https://de.mathworks.com/matlabcentral/fileexchange/27485-boundedline-m). The plots are saved using [export_fig](https://de.mathworks.com/matlabcentral/fileexchange/23629-export_fig?s_tid=srchtitle). All scripts were run in MATLAB 2022a.

