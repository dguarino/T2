# -*- coding: utf-8 -*-
"""
This is 
"""
from pyNN import nest
import sys
import mozaik
import mozaik.controller
from mozaik.controller import run_workflow, setup_logging
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet

from model_xcorr_full import ThalamoCorticalModel
    
from experiments import create_experiments_spontaneous
from experiments import create_experiments_luminance
from experiments import create_experiments_contrast
from experiments import create_experiments_spatial
from experiments import create_experiments_temporal
from experiments import create_experiments_size
from experiments import create_experiments_orientation
from experiments import create_experiments_correlation

from analysis_and_visualization import perform_analysis_test
from analysis_and_visualization import perform_analysis_and_visualization
from analysis_and_visualization import perform_analysis_and_visualization_radius


try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0

logger = mozaik.getMozaikLogger()


# Model execution
if True:
    data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_correlation )
    data_store.save()

# or only load pickled data
else:
    setup_logging()
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'ThalamoCorticalModel_data_xcorr_closed_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')

    # Analysis and Plotting
    if mpi_comm.rank == MPI_ROOT:
        perform_analysis_and_visualization( data_store, 'xcorr', withPGN, withV1 )
        
    data_store.save()
