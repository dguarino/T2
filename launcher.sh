#!/bin/bash

# $ chmod u+x launcher.sh
# $ launcher.sh

clear


echo "starting Size (redo PGN)"
echo 
mpirun python run_size.py nest 8 param/defaults_mea 'data_size'



# echo "starting Luminance"
# echo 
# mpirun python run_luminance.py nest 8 param/defaults 'data_luminance'
# mpirun python run_luminance_V1.py nest 8 param/defaults 'data_luminance_V1'


echo "starting Contrast"
echo 
# mpirun python run_contrast.py nest 8 param/defaults 'data_contrast'
mpirun python run_contrast_V1.py nest 8 param/defaults 'data_contrast_V1'


# echo "starting Spatial"
# echo 
# mpirun python run_spatial.py nest 8 param/defaults 'data_spatial'
# mpirun python run_spatial_V1.py nest 8 param/defaults 'data_spatial'


# echo "starting Temporal"
# echo 
# mpirun python run_temporal.py nest 8 param/defaults 'data_temporal'
# mpirun python run_temporal_V1.py nest 8 param/defaults 'data_temporal_V1'



echo "starting Size"
echo 
# mpirun python run_size.py nest 8 param/defaults_mea 'data_size'
mpirun python run_size_V1.py nest 8 param/defaults_mea 'data_size_V1'