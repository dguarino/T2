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
        # 'lgn.params.receptive_field.func_params.K1' : [0.525, 1.05, 2.1, 3.15, 4.2],
        # 'lgn.params.receptive_field.func_params.K2' : [0.35, 0.7, 1.4, 2.1, 2.8],

        # 'lgn.params.noise.stdev' : [1,2,3,4,5],

        # 'lgn.params.gain_control.gain' : [150, 200, 250, 300],
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [200000., 230000., 250000., 280000.],
        # 'lgn.params.gain_control.non_linear_gain.luminance_gain' : [0.001, 0.0001, 0.00001],

        'lgn.params.retino_thalamic_weight' : [.006], #[.006],

        # 'pgn.params.cell.params.tau_refrac': [0.5],
        # 'pgn.params.cell.params.tau_m': [17.0]

        'pgn.LGN_PGN_ConnectionOn.weight_functions.f1.params.arborization_constant': [50, 75], # model A of feedback: 75
        # 'pgn.LGN_PGN_ConnectionOn.base_weight': [.0012], #[.0009, .001], # model A of feedback: .0012
        # 'pgn.LGN_PGN_ConnectionOn.num_samples': [60], # model A of feedback: 60

        'pgn.PGN_PGN_Connection.weight_functions.f1.params.arborization_constant': [70], # model A of feedback: 70
        # 'pgn.PGN_PGN_Connection.base_weight': [.0001], #[.0008], # model A of feedback: .0001 
        # 'pgn.PGN_PGN_Connection.num_samples': [20], # model A of feedback: 20

        'pgn.PGN_LGN_ConnectionOn.weight_functions.f1.params.arborization_constant': [100, 150], # model A of feedback: 150
        # 'pgn.PGN_LGN_ConnectionOn.base_weight': [.0005, .0007, .0009], # model A of feedback: .0005
        # 'pgn.PGN_LGN_ConnectionOn.num_samples': [110], # model A of feedback: 110

        # 'l4_cortex_exc.AfferentConnection.base_weight' : [.0013],

        # 'l4_cortex_exc.EfferentConnection_LGN.base_weight' : [.00005, .00007], # model A of feedback: .0004
        # 'l4_cortex_exc.EfferentConnection_LGN.num_samples' : [400], # model A of feedback: 250,
        'l4_cortex_exc.EfferentConnection_LGN.weight_functions.f1.params.arborization_constant': [60], # model A of feedback: 60

        # 'l4_cortex_exc.EfferentConnection_PGN.base_weight' : [.0006], # model A of feedback: .0006
        # 'l4_cortex_exc.EfferentConnection_PGN.num_samples' : [40], # model A of feedback: 40
        'l4_cortex_exc.EfferentConnection_PGN.weight_functions.f1.params.arborization_constant': [90], # model A of feedback: 90 
  	}
).run_parameter_search()
