# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys

from functools import reduce # forward compatibility
import operator

import gc
import numpy
import scipy.stats
import pylab
import matplotlib.pyplot as plt

from parameters import ParameterSet

import mozaik
import mozaik.controller
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.TrialAveragedFiringRateCutout import TrialAveragedFiringRateCutout
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore



def perform_pairwise_comparison( sheet, folder_full, folder_inactive, parameter, indices, xlabel="", ylabel="" ):
	print folder_full
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	print "Checking data..."
	# Full
	dsv1 = queries.param_filter_query( data_store_full, identifier='PerNeuronValue', sheet_name=sheet )
	# dsv1.print_content(full_recordings=False)
	pnvs1 = [ dsv1.get_analysis_result() ]
	# get stimuli
	st1 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs1[-1]]
	# print st1

	# Inactivated
	dsv2 = queries.param_filter_query( data_store_inac, identifier='PerNeuronValue', sheet_name=sheet )
	pnvs2 = [ dsv2.get_analysis_result() ]
	# get stimuli
	st2 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs2[-1]]

	# rings analysis
	neurons_full = []
	neurons_inac = []

	spike_ids = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	sheet_ids = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
	neurons_full = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)

	spike_ids = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	sheet_ids = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
	neurons_inac = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)

	print "neurons_full:", len(neurons_full)
	print "neurons_inac:", len(neurons_inac)

	tc_dict1 = []
	tc_dict2 = []

	# Full
	# group values 
	dic = colapse_to_dictionary([z.get_value_by_id(neurons_full) for z in pnvs1[-1]], st1, parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))
	tc_dict1.append(dic)

	# Inactivated
	# group values 
	dic = colapse_to_dictionary([z.get_value_by_id(neurons_inac) for z in pnvs2[-1]], st2, parameter)
	for k in dic:
	    (b, a) = dic[k]
	    par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
	    dic[k] = (par,numpy.array(val))
	tc_dict2.append(dic)

	# Plotting tuning curves
	x_full = reduce( operator.getitem, indices, tc_dict1[0].values() )
	x_inac = reduce( operator.getitem, indices, tc_dict1[0].values() ) + numpy.random.uniform(low=0.2, high=2., size=(len(neurons_full)) ) #!!!!!!!!!!!!! only for test

	fig,ax = plt.subplots()
	ax.scatter( x_full, x_inac, marker="D", facecolor="k", edgecolor="k", label=sheet )
	x0,x1 = ax.get_xlim()
	y0,y1 = ax.get_ylim()
	# to make it squared
	if x1 >= y1:
		y1 = x1
	else:
		x1 = y1
	ax.set_xlim( (x0,x1) )
	ax.set_ylim( (y0,y1) )
	ax.set_aspect( abs(x1-x0)/abs(y1-y0) )
	# add diagonal
	ax.plot( [x0,x1], [y0,y1], linestyle='--', color="k" )

	# add regression line
	m,b = numpy.polyfit(x_full,x_inac, 1)
	x = numpy.arange(x0, x1)
	ax.plot(x, m*x+b, 'k-')

	# add correlation coefficient
	corr = numpy.corrcoef(x_full,x_inac)

	# text
	ax.set_title( sheet + " r=" + '{:.3f}'.format(corr[0][1]) )
	ax.set_xlabel( xlabel )
	ax.set_ylabel( ylabel )
	ax.legend( loc="lower right", shadow=False, scatterpoints=1 )
	# plt.show()
	plt.savefig( folder_inactive+"/TrialAveragedPairwiseComparison_"+parameter+"_"+sheet+".png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()






###################################################
# Execution
import os


full_list = [ 
	"ThalamoCorticalModel_data_luminance_V1_fake_____"
	]

inac_large_list = [ 
	"ThalamoCorticalModel_data_luminance_____.0012"
	]


sheets = ['X_ON', 'X_OFF'] #['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']


for i,l in enumerate(full_list):
	# for parameter search
	full = [ l+"/"+f for f in os.listdir(l) if os.path.isdir(os.path.join(l, f)) ]
	large = [ inac_large_list[i]+"/"+f for f in os.listdir(inac_large_list[i]) if os.path.isdir(os.path.join(inac_large_list[i], f)) ]

	for i,f in enumerate(full):
		print i

		for s in sheets:

			perform_pairwise_comparison( 
				sheet=s, 
				folder_full=f, 
				folder_inactive=large[i],
				parameter='background_luminance',
				indices=[0,1,2], # data, trialaveraged, 3rd stimulus
				xlabel="sponatneous activity before cooling (spikes/s)",
				ylabel="sponatneous activity during cooling (spikes/s)"
			)