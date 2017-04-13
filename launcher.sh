#!/bin/bash

# $ chmod u+x launcher.sh
# $ launcher.sh

# executable from /etc/crontab
# adding
# m  h    d m n
# 00 22   * * 5   root    /home/do/mozaik/mozaik/mozaik-contrib/T2/launcher.sh
# touch /home/do/test_cron.txt


# just to check
# mpirun python run_size_V1_inhibition_negative.py nest 8 param/defaults_mea 'data_size_V1_inh' 
# mpirun python run_size_V1_inhibition_positive.py nest 8 param/defaults_mea 'data_size_V1_inh' 





################################################
# PARAMETER SEARCH
# echo "starting PARAMETER SEARCH"
# echo 

# SPATIAL
# full closed loop model
# python parameter_search_full_all_models.py run_spatial_V1_full.py nest param/defaults
# python parameter_search.py run_spatial_V1.py nest param/defaults

# SIZE decorticated
# python parameter_search.py run_size.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size.py nest param/defaults_mea

# SIZE full closed loop model
python parameter_search_full_all_models.py run_size_closed_nonoverlapping.py nest param/defaults_mea
python parameter_search_full_all_models.py run_size_V1_inhibition_nonoverlapping.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_closed_overlapping.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_V1_inhibition_overlapping.py nest param/defaults_mea





################################################
#PLOTTING
# echo "starting PLOTTING"
# echo 

# python parameter_search_analysis.py CombinationParamSearch_closed_PGNLGN_150_200_250_300__V1PGN_90_70_50_30
# python parameter_search_analysis.py CombinationParamSearch_nonover_PGNLGN_150_200_250_300__V1PGN_90_70_50_30

# python parameter_search_analysis.py CombinationParamSearch_full_3sigma_1
# python parameter_search_analysis.py CombinationParamSearch_nonover_3sigma_1
# python parameter_search_analysis.py CombinationParamSearch_over_3sigma_1

# python parameter_search_analysis.py CombinationParamSearch_size_decorticated_8

# python parameter_search_analysis.py CombinationParamSearch_size_full_8
# python parameter_search_analysis.py CombinationParamSearch_size_inhibition_8
# python parameter_search_analysis.py CombinationParamSearch_size_inhibition_nonoverlapping_8


# python comparison_tuning.py



################################################
# EXPERIMENTS Closed Loop 1 (LGN+PGN)

# echo "starting Luminance"
# echo
# mpirun -np 1 python run_luminance_open.py nest 8 param/defaults 'data_luminance_open'


# echo "starting Spatial"
# echo
# mpirun -np 1 python run_spatial_open.py nest 8 param/defaults 'data_spatial_open'


# echo "starting Temporal"
# echo
# mpirun -np 1 python run_temporal_open.py nest 8 param/defaults 'data_temporal_open'


# echo "starting Contrast"
# echo
# mpirun -np 1 python run_contrast_open.py nest 8 param/defaults 'data_contrast_open'


# echo "starting Size"
# echo
# mpirun -np 1 python run_size_open.py nest 8 param/defaults_mea 'data_size_open'


# echo "starting XCorr"
# echo
# mpirun -np 1 python run_xcorr_open.py nest 8 param/defaults_xcorr 'data_xcorr_open'


# echo "starting XCorr"
# echo
# mpirun -np 1 python run_xcorr_closed.py nest 8 param/defaults_xcorr 'data_xcorr_closed'


# echo "starting Orientation"
# echo
# mpirun -np 1 python run_orientation_open.py nest 8 param/defaults_mea 'data_orientation_open'



################################################
# EXPERIMENTS Closed loop 2 (LGN+PGN+V1)

# echo "starting Luminance"
# echo
# mpirun -np 1 python run_luminance_closed.py nest 8 param/defaults 'data_luminance_closed'


# echo "starting Contrast"
# echo
# mpirun -np 1 python run_contrast_closed.py nest 8 param/defaults 'data_contrast_closed'


# echo "starting Spatial"
# echo
# mpirun -np 1 python run_spatial_closed.py nest 8 param/defaults 'data_spatial_closed'


# echo "starting Spatial Kimura"
# echo
# mpirun -np 1 python run_spatial_Kimura.py nest 8 param/defaults 'data_spatial_Kimura'


# echo "starting Temporal"
# echo
# mpirun -np 1 python run_temporal_closed.py nest 8 param/defaults 'data_temporal_closed'


# echo "starting Orientation"
# echo
# mpirun -np 1 python run_orientation_closed.py nest 8 param/defaults_mea 'data_orientation_closed'


# echo "starting Size"
# echo
# mpirun -np 1 python run_size_closed.py nest 8 param/defaults_mea 'data_size_closed'
# mpirun -np 1 python run_size_V1_inhibition_nonoverlapping.py nest 8 param/defaults 'data_size_nonoverlapping'
# mpirun -np 1 python run_size_V1_inhibition_overlapping.py nest 8 param/defaults 'data_size_overlapping'


# echo "starting XCorr"
# echo
# mpirun -np 1 python run_xcorr_closed.py nest 8 param/defaults_xcorr 'data_xcorr_closed'



################################################
# EXPERIMENTS Closed loop 3 (LGN+PGN+V1)
# with V1 but without feedback


# echo "starting Contrast"
# echo
# mpirun -np 1 python run_size_open2.py nest 8 param/defaults 'data_contrast_open2'


# echo "starting Spatial"
# echo
# mpirun -np 1 python run_orientation_open2.py nest 8 param/defaults 'data_spatial_open2'

# mpirun -np 1 python run_bar_feedforward.py nest 8 param/defaults_mea 'data_bar_feedforward'
