# IMACNN
 
This repository contains the code for the temporal analysis of the project "Using generative AI, CNNs and extensive individual sampling to characterize the visual features encoded in alpha rhythms during imagery" which Rico conducted during his lab visit in 2025.

|  Step  | Procedure & Script                |
|-------|-----------------|
| 1.1.  | Preprocessing: preprocess_EEG_individual_scene_imagery.m|
| 1.2.  | Channel interpolation and downsampling: fieldtrip_channel_interp_and_downsampling.m |
| 2.1.  | EEG RDM creation & mean pairwise decoding: create_rdms_across_time.m     |
| 2.2.  | DNN RDM creation:      |
| 3.1.  | Conduct statistical test and plot mean pairwise decoding: plot_mean_pairwise_decoding     |
| 3.2.  | Correlate EEG RDMs with DNN RDMs, run statistical test and plot correlations: plot_dnn_temporal_corr     |

The preprocessing scripts and RDM creation script require [FieldTrip](https://www.fieldtriptoolbox.org/). The RDM creation script also requires [CosMoMVPA](https://www.cosmomvpa.org/) and the MATLAB [domain adaptation toolbox](https://de.mathworks.com/matlabcentral/fileexchange/56704-a-domain-adaptation-toolbox). The shaded error bars in the plots use the [boundedline function](https://de.mathworks.com/matlabcentral/fileexchange/27485-boundedline-m). The plots are saved using [export_fig](https://de.mathworks.com/matlabcentral/fileexchange/23629-export_fig?s_tid=srchtitle). All scripts were run in MATLAB 2022a.

