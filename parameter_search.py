# Parameter search for Gain control
#
# the function LocalSequentialBackend takes a model with parameters 
# and executes the simulation replacing the parameters listed in the CombinationParameterSearch
#
# usage:
# python parameter_search.py run.py nest param/defaults

from mozaik.meta_workflow.parameter_search import CombinationParameterSearch
from mozaik.meta_workflow.parameter_search import LocalSequentialBackend
import numpy

CombinationParameterSearch(
	LocalSequentialBackend( num_threads=8 ),
	{
        #'lgn.params.noise.stdev' : [1,2,3,4,5],
        # 'lgn.params.gain_control.gain' : [40, 30, 20, 10],
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [100, 150, 200, 250, 300],
        # 'lgn.params.gain_control.non_linear_gain.luminance_scaler' : [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1],
        'lgn.params.gain_control.non_linear_gain.contrast_scaler' : [0.5, 0.1, 0.01, 0.001, 0.0001],
        # 'lgn.params.gain_control.non_linear_gain.contrast_scaler' : [10., 1.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],

    	# 'pgn.LGN_PGN_ConnectionOn.base_weight': [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004], # automatic (ref) assignment also to Off neurons
    	# 'pgn.PGN_PGN_Connection.base_weight': [0.003, 0.0035, 0.004, 0.0045, 0.005], 
    	# 'pgn.PGN_LGN_ConnectionOn.base_weight': [0.0001, 0.0005, 0.001], # automatic (ref) assignment also to Off neurons
        # 'pgn.PGN_PGN_Connection.weight_functions.f1.params.arborization_constant': [200.0, 400.0, 600.0],  # um decay distance from the innervation point
        # 'pgn.params.cell.params.tau_m': [17.0]

        # 'l4_cortex_exc.AfferentConnection.base_weight' : [0.0001, 0.0005, 0.001],
    	# 'l4_cortex_exc.EfferentConnection_LGN.base_weight' : [0.0001, 0.0005, 0.001],
    	# 'l4_cortex_exc.EfferentConnection_PGN.base_weight' : [0.0005, 0.001, 0.0015],
  	}
).run_parameter_search()

# RGC tuning
# luminance

# contrast


# PGN responses are 
# - 100 sp/s with DG at optimal LGN freq
# - reducing 

# LGN-PGN and PGN-LGN
#                                                     LGN_PGN            PGN_LGN           PGN_PGN
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.003 ++ 39 sp/s
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.003_base_weight:0.0001_base_weight:0.003 + 31 sp/s
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.0035 +
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.004 +
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.003_base_weight:0.0001_base_weight:0.004
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.0035_base_weight:0.0001_base_weight:0.003


# PGN-PGN
# base_weight       arborization_constant
# 0.005             200


# with V1
# Ratio             weight      match data
# 0.7               0.0005      
# 0.5               0.0005      
# 0.7               0.0003       
