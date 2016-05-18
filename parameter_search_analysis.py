# Usage
# $ python parameter_search_analysis.py 20160504-100834[param.defaults]CombinationParamSearch{3}

import sys
from mozaik.meta_workflow.parameter_search import parameter_search_run_script
assert len(sys.argv) == 2
directory = sys.argv[1]

#                            simulation_name,  master_results_dir, run_script, core_number
parameter_search_run_script( "ThalamoCorticalModel", directory, 'run_analysis_and_visualization.py', 1 )
