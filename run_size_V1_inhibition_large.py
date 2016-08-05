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
    
from experiments import create_experiments_spontaneous
from experiments import create_experiments_luminance
from experiments import create_experiments_contrast
from experiments import create_experiments_spatial
from experiments import create_experiments_temporal
from experiments import create_experiments_size
from experiments import create_experiments_size_V1_inactivated_small
from experiments import create_experiments_size_V1_inactivated_large
from experiments import create_experiments_orientation
from experiments import create_experiments_correlation
from experiments import create_experiments_combined

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
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_spontaneous )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_luminance )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_contrast )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_spatial )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_temporal )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_size )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_size_V1_inactivated_small )
    data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_size_V1_inactivated_large )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_orientation )
    # data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_combined )

    # if False: # save connections
    #     if withPGN: # PGN
    #         model.connectors['LGN_PGN_ConnectionOn'].store_connections(data_store)    
    #         model.connectors['LGN_PGN_ConnectionOff'].store_connections(data_store)    
    #         model.connectors['PGN_PGN_Connection'].store_connections(data_store)    
    #         model.connectors['PGN_LGN_ConnectionOn'].store_connections(data_store)    
    #         model.connectors['PGN_LGN_ConnectionOff'].store_connections(data_store)    
        # if withV1: # CORTEX
        #     # model.connectors['V1AffConnectionOn'].store_connections(data_store)    
        #     # model.connectors['V1AffConnectionOff'].store_connections(data_store)    
        #     # model.connectors['V1AffInhConnectionOn'].store_connections(data_store)    
        #     # model.connectors['V1AffInhConnectionOff'].store_connections(data_store)    
        #     # model.connectors['V1L4ExcL4ExcConnection'].store_connections(data_store)    
        #     # model.connectors['V1L4ExcL4InhConnection'].store_connections(data_store)    
        #     # model.connectors['V1L4InhL4ExcConnection'].store_connections(data_store)    
        #     # model.connectors['V1L4InhL4InhConnection'].store_connections(data_store)    
        #     # model.connectors['V1L4ExcL4ExcConnectionRand'].store_connections(data_store)    
        #     # model.connectors['V1L4ExcL4InhConnectionRand'].store_connections(data_store)    
        #     # model.connectors['V1L4InhL4ExcConnectionRand'].store_connections(data_store)    
        #     # model.connectors['V1L4InhL4InhConnectionRand'].store_connections(data_store)
        #     if withFeedback_CxLGN:
        #         model.connectors['V1EffConnectionOn'].store_connections(data_store)    
        #         model.connectors['V1EffConnectionOff'].store_connections(data_store)    
        #     # if withPGN and withFeedback_CxPGN:
        #     #     model.connectors['V1EffConnectionPGN'].store_connections(data_store)    

    data_store.save()

# or only load pickled data
else:
    setup_logging()
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'ThalamoCorticalModel_data_size_V1_inh_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')

# # Analysis and Plotting
# if mpi_comm.rank == MPI_ROOT:
#     # perform_analysis_test( data_store )
#     # perform_analysis_and_visualization( data_store, 'luminance', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'contrast', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'spatial_frequency', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'temporal_frequency', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'size', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'size_radius', withPGN, withV1 )
#     # perform_analysis_and_visualization( data_store, 'orientation', withPGN, withV1 )
#     import numpy
#     step = .2
#     for i in numpy.arange(step, 2.+step, step):
#         perform_analysis_and_visualization_radius( data_store, 'size_radius', [i-step,i], withPGN, withV1 )