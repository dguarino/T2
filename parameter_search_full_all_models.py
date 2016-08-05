# Parameter search for Gain control
#
# the function LocalSequentialBackend takes a model with parameters 
# and executes the simulation replacing the parameters listed in the CombinationParameterSearch
#
# usage:
# python parameter_search.py run.py nest param/defaults

# python parameter_search.py run_spatial.py nest param/defaults
# python parameter_search.py run_spatial_V1.py nest param/defaults

# python parameter_search.py run_size.py nest param/defaults_mea
# python parameter_search.py run_size_V1.py nest param/defaults_mea

# python parameter_search.py run_size_V1_inhibition.py nest param/defaults_mea


from mozaik.meta_workflow.parameter_search import CombinationParameterSearch
from mozaik.meta_workflow.parameter_search import LocalSequentialBackend
import numpy

CombinationParameterSearch(
	LocalSequentialBackend( num_threads=8 ),
	{
        # 'lgn.params.receptive_field.func_params.c2' : [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #'lgn.params.receptive_field.func_params.K1' : [0.525, 1.05, 2.1, 3.15, 4.2],
        #'lgn.params.receptive_field.func_params.K2' : [0.35, 0.7, 1.4, 2.1, 2.8],

        # 'lgn.params.noise.stdev' : [1,2,3,4,5],
        # 'lgn.params.retino_thalamic_weight' : [.005, .01, .02],

        # 'lgn.params.gain_control.gain' : [150, 200, 250, 300], #, 400, 450, 500, 550],
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [200000., 230000., 250000., 280000.],
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [0.001, 0.0001, 0.00001],

        'pgn.LGN_PGN_ConnectionOn.base_weight': [.001], #orig: .002 # automatic (ref) assignment also to Off neurons

        # 'pgn.PGN_LGN_ConnectionOn.base_weight': [.0008], # automatic (ref) assignment also to Off neurons
        'pgn.PGN_PGN_Connection.base_weight': [.0001],
        # 'pgn.PGN_PGN_Connection.num_samples': [40], 
        # 'pgn.params.cell.params.tau_m': [17.0]

        'l4_cortex_exc.AfferentConnection.base_weight' : [.002],

        'l4_cortex_exc.EfferentConnection_LGN.base_weight' : [.0001],        
        # 'l4_cortex_exc.EfferentConnection_LGN.num_samples' : [100, 150], #200],
        'l4_cortex_exc.EfferentConnection_LGN.weight_functions.f1.params.arborization_constant': [150],

        'l4_cortex_exc.EfferentConnection_PGN.base_weight' : [.001],
        # 'l4_cortex_exc.EfferentConnection_PGN.num_samples' : [50, 80],
        # 'l4_cortex_exc.EfferentConnection_PGN.weight_functions.f1.params.arborization_constant': [40, 60],
  	}
).run_parameter_search()

#                                                      Aff                LGN               PGN
# ThalamoCorticalModel_ParameterSearch_____base_weight:0.0005_base_weight:5e-05_base_weight:0.0001


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
