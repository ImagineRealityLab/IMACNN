# IMACNN
 
This repository contains the code for the time-resolved representational similarity analysis (RSA) for the project "Using generative AI, CNNs and extensive individual sampling to characterize the visual features encoded in alpha rhythms during imagery" which Rico Stecher conducted during his lab visit in 2025.

**Task & Analysis**

10 participants imagined 16 natural scenes according to short prompts while their EEG was recorded for 10 sessions (4320 trials per participant).

We first created representational dissimilarity matrices (RDMs) at each time point. We obtained the RDMs by first mapping the evoked voltage patterns at each time point in the EEG to a session-independent feature space, creating pseudotrials in this feature space by averaging together multiple trials across sessions and then finally training classifiers to discriminate between each pair of the 16 imagined scenes. We first assessed how well the imagined scenes can be discriminated from the voltage patterns across time by averaging the pairwise decoding accuracies in the bottom triangle of these RDMs at every time point. In the main analysis, we then investigated in what order visual features of varying levels of abstraction are reactivated during imagery. To that end, we first simulated visual cortex responses to the imagined scene images by feeding a CNN trained on scene classification 100 sets of AI-generated images as an estimate of the imagined visual contents. We then created RDMs for each image set at each layer and averaged the RDMs across all 100 image sets. We averaged RDMs of the early, intermediate and fully connected layers into three 3 respective layer groups that capture the discriminability of the images based on low-level, mid-level and high-level visual feature representations. Finally, we predicted the imagery RDMs at each time point with these layer group RDMs using a ridge regression. 

Note, that this pipeline results in an overfit of the classifier during the pairwise decoding within the RDM. While this shouldn't influence the interpretability of RSA results, we still added an approach that fixes this by estimating the session-independent feature space on the training set and mapping it to the test set.

|  Step  | Procedure & Script                |
|-------|-----------------|
| 1.1.  | Preprocessing: preprocess_EEG.m|
| 1.2.  | Channel interpolation and downsampling: fieldtrip_channel_interp_and_downsampling.m |
| 2.1.  | EEG RDM creation & mean pairwise decoding: create_rdms_across_time.m  or create_rdms_across_time_train_set_est.m   |
| 2.2.  | DNN RDM creation: create_dnn_rdms.m     |
| 3.1.  | Conduct statistical test and plot mean pairwise decoding: plot_mean_pairw_dec.m     |
| 3.2.  | Predict EEG RDMs across time with DNN layer RDMs, run statistical test and plot ridge coefficients: dnn_layer_rdm_ridge_reg.m     |

The preprocessing scripts (step 1) and EEG RDM creation scripts require [FieldTrip](https://www.fieldtriptoolbox.org/). The EEG RDM creation scripts also requires [CosMoMVPA](https://www.cosmomvpa.org/) and the MATLAB [domain adaptation toolbox](https://de.mathworks.com/matlabcentral/fileexchange/56704-a-domain-adaptation-toolbox). The EEG RDM creation script that uses the train set estimate for the feature mapping requires the ftTrans_mida_project function. The statistical tests in both plotting scripts requires the EEG_clusterstats function. The shaded error bars in the plots use the [boundedline function](https://de.mathworks.com/matlabcentral/fileexchange/27485-boundedline-m). The plots are saved using [export_fig](https://de.mathworks.com/matlabcentral/fileexchange/23629-export_fig?s_tid=srchtitle). All scripts were run in MATLAB 2022a.

