#!/bin/bash

# $ chmod u+x launcher.sh
# $ launcher.sh

clear


################################################
# PARAMETER SEARCH
# echo "starting PARAMETER SEARCH"
# echo 

# python parameter_search.py run_spatial.py nest param/defaults
# python parameter_search.py run_spatial_V1.py nest param/defaults


################################################
# EXPERIMENTS Closed Loop 1 (LGN+PGN)

# echo "starting Luminance"
# echo 
# mpirun python run_luminance.py nest 8 param/defaults 'data_luminance'


# echo "starting Contrast"
# echo 
# mpirun python run_contrast.py nest 8 param/defaults 'data_contrast'


# echo "starting Spatial"
# echo 
# mpirun python run_spatial.py nest 8 param/defaults 'data_spatial'


# echo "starting Size"
# echo 
# mpirun python run_size.py nest 8 param/defaults_mea 'data_size'


# echo "starting Temporal"
# echo 
# mpirun python run_temporal.py nest 8 param/defaults 'data_temporal'




################################################
# EXPERIMENTS Closed loop 2 (LGN+PGN+V1)


echo "starting Size"
echo 
mpirun python run_size_V1.py nest 8 param/defaults_mea 'data_size_V1' 
   # #Model: SomersTodorovSiapasTothKimSur1998 reaches 60 sp/s
   # #Data: DeAngelisFreemanOhzawa1994 20-15 sp/s


echo "starting Contrast"
echo 
mpirun python run_contrast_V1.py nest 8 param/defaults 'data_contrast_V1'


echo "starting Spatial"
echo 
mpirun python run_spatial_V1.py nest 8 param/defaults 'data_spatial_V1'


echo "starting Spatial Kimura"
echo 
mpirun python run_spatial_Kimura.py nest 8 param/defaults 'data_spatial_Kimura'


echo "starting Temporal"
echo 
mpirun python run_temporal_V1.py nest 8 param/defaults 'data_temporal_V1'

