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

from model_V1_full import ThalamoCorticalModel
    
from experiments import create_experiments_size_V1_inactivated_overlapping

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


# Manage what is executed
# a set of variable here to manage the type of experiment and whether the pgn, cortex are there or not.
withPGN = True  # 
withV1 = True  # open-loop
withFeedback_CxPGN = True # closed loop
withFeedback_CxLGN = True # closed loop

# Model execution
if True:
    data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_size_V1_inactivated_overlapping )
    data_store.save()

# or only load pickled data
else:
    setup_logging()
    # data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'Deliverable/ThalamoCorticalModel_data_size_overlapping_____', 'store_stimuli' : False}),replace=True)
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'ThalamoCorticalModel_data_size_overlapping_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')

    # Analysis and Plotting
    if mpi_comm.rank == MPI_ROOT:
        # perform_analysis_test( data_store )
        # perform_analysis_and_visualization( data_store, 'luminance', withPGN, withV1 )
        # perform_analysis_and_visualization( data_store, 'contrast', withPGN, withV1 )
        # perform_analysis_and_visualization( data_store, 'spatial_frequency', withPGN, withV1 )
        # perform_analysis_and_visualization( data_store, 'temporal_frequency', withPGN, withV1 )
        perform_analysis_and_visualization( data_store, 'size', withPGN, withV1 )
        # perform_analysis_and_visualization( data_store, 'size_radius', withPGN, withV1 )
        # perform_analysis_and_visualization( data_store, 'orientation', withPGN, withV1 )
        # import numpy
        # step = .2
        # for i in numpy.arange(step, 2.+step, step):
        #     perform_analysis_and_visualization_radius( data_store, 'size_radius', [i-step,i], withPGN, withV1 )
        
    data_store.save()
