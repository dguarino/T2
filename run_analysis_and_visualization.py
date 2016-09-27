# -*- coding: utf-8 -*-
"""
"""
import sys
from mozaik.controller import setup_logging
import mozaik
from mozaik.storage.datastore import Hdf5DataStore,PickledDataStore
from analysis_and_visualization import perform_analysis_and_visualization
from analysis_and_visualization import perform_analysis_and_visualization_radius
from parameters import ParameterSet

from mozaik.controller import Global
Global.root_directory = sys.argv[1]+'/'


withPGN = True  # 
withV1 = True  # open-loop


setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':sys.argv[1],'store_stimuli' : False}),replace=True)

# perform_analysis_and_visualization( data_store, 'spatial_frequency', withPGN, withV1 )
# perform_analysis_and_visualization( data_store, 'size', withPGN, withV1 )

# # Simple Analysis (just one group)
# perform_analysis_and_visualization( data_store, 'cortical_map', withPGN, withV1 )
perform_analysis_and_visualization_radius( data_store, 'size_radius', [[.0],[.0],[.0]], [.0,.5], withPGN, withV1 )
# perform_analysis_and_visualization_radius( data_store, 'size_radius', [[1.6],[.0],[.0]], [.0,.5], withPGN, withV1 )

# # Several Grouping Analysis
# import numpy
# step = .2
# for i in numpy.arange(step, 1.+step, step):
#     print i
#     perform_analysis_and_visualization_radius( data_store, 'size_radius', [[.0],[.0],[.0]], [i-step,i], withPGN, withV1 )
#     # perform_analysis_and_visualization_radius( data_store, 'size_radius', [[1.6],[.0],[.0]], [i-step,i], withPGN, withV1 )


data_store.save() 
