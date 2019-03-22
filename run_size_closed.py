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


# Manage what is executed
# a set of variable here to manage the type of experiment and whether the pgn, cortex are there or not.
withPGN = True  # 
withV1 = True  # open-loop
withFeedback_CxPGN = True # closed loop
withFeedback_CxLGN = True # closed loop

# Model execution
data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_experiments_size )
data_store.save()

