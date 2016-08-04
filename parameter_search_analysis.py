# Usage
# $ python parameter_search_analysis.py CombinationParamSearch_PGN_LGN___PGN_PGN
# $ python parameter_search_analysis.py CombinationParamSearch_l4_LGN_PGN

# $ python parameter_search_analysis.py CombinationParamSearch_l4_LGN_PGN_small

# $ python parameter_search_analysis.py CombinationParamSearch_l4_LGN_60_150_200_full
# $ python parameter_search_analysis.py CombinationParamSearch_l4_LGN_60_150_200_large_inhibition

# $ python parameter_search_analysis.py CombinationParamSearch_size_V1_full
# $ python parameter_search_analysis.py CombinationParamSearch_size_V1_inhibition_large

import sys
from mozaik.meta_workflow.parameter_search import parameter_search_run_script
assert len(sys.argv) == 2
directory = sys.argv[1]

#                            simulation_name,  master_results_dir, run_script, core_number
parameter_search_run_script( "ThalamoCorticalModel", directory, 'run_analysis_and_visualization.py', 1 )
