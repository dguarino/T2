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
echo "starting PARAMETER SEARCH"
echo 

# python parameter_search.py run_size.py nest param/defaults_mea
# python parameter_search.py run_spatial_V1.py nest param/defaults
# python parameter_search.py run_size_V1.py nest param/defaults_mea
# python parameter_search_full.py run_spatial_V1_full.py nest param/defaults

# SPATIAL
# full closed loop model
# python parameter_search_full_all_models.py run_spatial_V1_full.py nest param/defaults

# SIZE
# decorticated
# python parameter_search_PGNLGN_1.py run_size.py nest param/defaults_mea
# python parameter_search_PGNLGN_2.py run_size.py nest param/defaults_mea
# python parameter_search_PGNLGN_3.py run_size.py nest param/defaults_mea

# python parameter_search_full_all_models.py run_size.py nest param/defaults_mea
# # full closed loop model
# python parameter_search_full_all_models.py run_size_V1_full.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_V1_inhibition_nonoverlapping.py nest param/defaults_mea
# python parameter_search_full_all_models.py run_size_V1_inhibition_overlapping.py nest param/defaults_mea




################################################
#PLOTTING
# echo "starting PLOTTING"
# echo 

# python parameter_search_analysis.py CombinationParamSearch_inactive_PGNLGN_1
# python parameter_search_analysis.py CombinationParamSearch_inactive_PGNLGN_2
# python parameter_search_analysis.py CombinationParamSearch_inactive_PGNLGN_3

# python parameter_search_analysis.py CombinationParamSearch_size_decorticated_8

# python parameter_search_analysis.py CombinationParamSearch_size_full_8
# python parameter_search_analysis.py CombinationParamSearch_size_inhibition_8
# python parameter_search_analysis.py CombinationParamSearch_size_inhibition_nonoverlapping_8


# python comparison_tuning.py



################################################
# EXPERIMENTS Closed Loop 1 (LGN+PGN)

echo "starting Luminance"
echo 
mpirun python run_luminance.py nest 8 param/defaults 'data_luminance'


echo "starting Spatial"
echo 
mpirun python run_spatial.py nest 8 param/defaults 'data_spatial'


echo "starting Temporal"
echo 
mpirun python run_temporal.py nest 8 param/defaults 'data_temporal'


echo "starting Contrast"
echo 
mpirun python run_contrast.py nest 8 param/defaults 'data_contrast'


# echo "starting Size"
# echo 
# mpirun python run_size.py nest 8 param/defaults_mea 'data_size'




################################################
# EXPERIMENTS Closed loop 2 (LGN+PGN+V1)


# echo "starting Size"
# echo 
# mpirun python run_size_V1.py nest 8 param/defaults_mea 'data_size_V1' 
# mpirun python run_size_V1_full.py nest 8 param/defaults_mea 'data_size_V1' 
#V1 Model: SomersTodorovSiapasTothKimSur1998 reaches 60 sp/s
#V1 Data: DeAngelisFreemanOhzawa1994 20-15 sp/s


# echo "starting Contrast"
# echo 
# mpirun python run_contrast_V1.py nest 8 param/defaults 'data_contrast_V1'


# echo "starting Spatial"
# echo 
# mpirun python run_spatial_V1.py nest 8 param/defaults 'data_spatial_V1'


# echo "starting Spatial Kimura"
# echo 
# mpirun python run_spatial_Kimura.py nest 8 param/defaults 'data_spatial_Kimura'


# echo "starting Temporal"
# echo 
# mpirun python run_temporal_V1.py nest 8 param/defaults 'data_temporal_V1'

