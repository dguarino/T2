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

from model_feedforward import ThalamoCorticalModel
    
from experiments import create_interrupted_bar


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
withFeedback_CxPGN = False # ffw loop
withFeedback_CxLGN = False # ffw loop

withRandomV1conns = False

# Model execution
data_store,model = run_workflow('ThalamoCorticalModel', ThalamoCorticalModel, create_interrupted_bar )
data_store.save()

