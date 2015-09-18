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
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [10.0, 20.0],
        # 'lgn.params.gain_control.non_linear_gain.luminance_scaler' : [.001, .01, 0.1],
        # 'lgn.params.gain_control.non_linear_gain.contrast_scaler' : [.001, .005, .01, .015, .02], #.001
    	'pgn.LGN_PGN_ConnectionOn.base_weight': [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004], # automatic (ref) assignment also to Off neurons
    	'pgn.PGN_PGN_Connection.base_weight': [0.003, 0.0035, 0.004, 0.0045, 0.005], 
    	'pgn.PGN_LGN_ConnectionOn.base_weight': [0.0001, 0.0005, 0.001], # automatic (ref) assignment also to Off neurons
        # 'pgn.params.cell.params.tau_m': [17.0]
        # 'l4_cortex_exc.ExcInhAfferentRatio' : [0.5, 0.7],
        # 'l4_cortex_exc.AfferentConnection.base_weight' : [0.002, 0.003, 0.004, 0.005],
    	# 'l4_cortex_exc.EfferentConnection_LGN.base_weight' : [0.0001, 0.0005, 0.001],
    	# 'l4_cortex_exc.EfferentConnection_PGN.base_weight' : [0.0005, 0.001, 0.0015],
  	}
).run_parameter_search()

# PGN responses are 
# - 100 sp/s with DG at optimal LGN freq
# - reducing 
#                                                     LGN_PGN            PGN_LGN           PGN_PGN
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.003 ++ 39 sp/s
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.003_base_weight:0.0001_base_weight:0.003 + 31 sp/s
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.0035 +
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.004_base_weight:0.0001_base_weight:0.004 +
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.003_base_weight:0.0001_base_weight:0.004
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.0035_base_weight:0.0001_base_weight:0.003

