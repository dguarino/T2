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



# mpirun -np 1 python run_size_closed.py nest 8 param/defaults_mea 'data_size_closed_conductances'
# mpirun -np 1 python run_size_feedforward.py nest 8 param/defaults_mea 'data_size_feedforward_conductances'




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
# python parameter_search_full_all_models.py run_size_closed.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_open.py nest param/defaults_mea

# python parameter_search_full_all_models.py run_size_V1_inhibition_nonoverlapping.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_V1_inhibition_overlapping.py nest param/defaults_mea

# VSDI
# python parameter_search_full_all_models.py run_size_closed.py nest param/defaults_mea

# ANALYSIS
# python parameter_search_analysis.py CombinationParamSearch_intact_nonoverlapping
# python parameter_search_analysis.py CombinationParamSearch_altered_nonoverlapping


################################################
# EXPERIMENTS Closed Loop 1 (LGN+PGN)

# echo "starting Luminance"
# echo
# mpirun -np 1 python run_luminance_open.py nest 8 param/defaults_mea 'data_luminance_open'


# echo "starting Spatial"
# echo
# mpirun -np 1 python run_spatial_open.py nest 8 param/defaults_mea 'data_spatial_open'


# echo "starting Temporal"
# echo
# mpirun -np 1 python run_temporal_open.py nest 8 param/defaults_mea 'data_temporal_open'


# echo "starting Contrast"
# echo
# mpirun -np 1 python run_contrast_open.py nest 8 param/defaults_mea 'data_contrast_open'


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
# mpirun -np 1 python run_luminance_closed.py nest 8 param/defaults_mea 'data_luminance_closed'


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


# echo "starting bar"
# echo
# mpirun -np 1 python run_interrupted_bar.py nest 8 param/defaults 'data_interrupted_bar'



echo "starting Size"
echo
mpirun -np 1 python run_size_closed.py nest 8 param/defaults_mea 'data_size_closed'
# mpirun -np 1 python run_size_closed_nonoverlapping.py nest 8 param/defaults_mea 'data_size_closed_nonoverlapping'
# mpirun -np 1 python run_size_V1_inhibition_nonoverlapping.py nest 8 param/defaults_mea 'data_size_nonoverlapping'
# mpirun -np 1 python run_size_closed_overlapping.py nest 8 param/defaults_mea 'data_size_closed_overlapping'
# mpiru