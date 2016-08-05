# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import mozaik
import mozaik.controller
from parameters import ParameterSet

import numpy
import pylab
import matplotlib.pyplot as plt

from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.TrialAveragedFiringRateCutout import TrialAveragedFiringRateCutout
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.circ_stat import circular_dist
from mozaik.controller import Global



def select_ids_by_radius(radius, sheet_ids, positions):
	radius_ids = []
	distances = []
	min_radius = radius[0] # over: 0. # non: 1.
	max_radius = radius[1] # over: .7 # non: 3.

	for i in sheet_ids:
		a = numpy.array((positions[0][i],positions[1][i],positions[2][i]))
		l = numpy.linalg.norm(a - 0.0)

		if l>min_radius and l<max_radius:
			radius_ids.append(i[0])
			distances.append(l)

	# sort by distance
	# print radius_ids
	# print distances
	return [x for (y,x) in sorted(zip(distances,radius_ids), key=lambda pair: pair[0])]
	# return radius_ids


def perform_comparison_size_tuning( sheet, step, sizes, folder_full, folder_inactive ):
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	# data_store_full.print_content(full_recordings=False)
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	# data_store_inac.print_content(full_recordings=False)

	print "Checking data..."
	# Full
	dsv1 = queries.param_filter_query( data_store_full, identifier='PerNeuronValue', sheet_name=sheet )
	# dsv.print_content(full_recordings=False)
	pnvs1 = [ dsv1.get_analysis_result() ]
	# get stimuli
	st1 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs1[-1]]

	# Inactivated
	dsv2 = queries.param_filter_query( data_store_inac, identifier='PerNeuronValue', sheet_name=sheet )
	pnvs2 = [ dsv2.get_analysis_result() ]
	# get stimuli
	st2 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs2[-1]]

	# rings analysis
	neurons_full = []
	neurons_inac = []
	rowplots = 0
	max_size = 2.

	slice_ranges = numpy.arange(step, max_size+step, step)
	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step,cur_range]

		# get the list of all recorded neurons in X_ON
		# Full
		spike_ids = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		position = data_store_full.get_neuron_postions()[sheet]
		sheet_ids = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
		radius_ids = select_ids_by_radius(radius, sheet_ids, position)
		neurons = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=radius_ids)
		if len(neurons) > rowplots:
			rowplots = len(neurons)
		neurons_full.append(neurons)
		# Inactivated
		spike_ids = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		position = data_store_inac.get_neuron_postions()[sheet]
		sheet_ids = data_store_inac.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
		radius_ids = select_ids_by_radius(radius, sheet_ids, position)
		neurons = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=radius_ids)
		neurons_inac.append(neurons)

		assert len(neurons_full[col]) == len(neurons_inac[col]) , "ERROR: the number of recorded neurons is different"
		assert set(neurons_full[col]) == set(neurons_inac[col]) , "ERROR: the neurons in the two arrays are not the same"

	# subplot figure creation
	print 'rowplots', rowplots
	print "Starting plotting ..."
	fig, axes = plt.subplots(nrows=len(slice_ranges), ncols=rowplots+1, figsize=(4*rowplots, 4*len(slice_ranges)), sharey=False)

	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step,cur_range]
		interval = str(radius[0]) +" - "+ str(radius[1]) +" deg radius"
		axes[col,0].set_ylabel(interval+"\n\nResponse change (%)")
		print "range:",col
		print "neurons_full:", len(neurons_full[col]), neurons_full[col]
		print "neurons_inac:", len(neurons_inac[col]), neurons_inac[col]

		tc_dict1 = []
		tc_dict2 = []

		# Full
		# group values 
		dic = colapse_to_dictionary([z.get_value_by_id(neurons_full[col]) for z in pnvs1[-1]], st1, 'radius')
		for k in dic:
		    (b, a) = dic[k]
		    par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		    dic[k] = (par,numpy.array(val))
		tc_dict1.append(dic)

		# Inactivated
		# group values 
		dic = colapse_to_dictionary([z.get_value_by_id(neurons_inac[col]) for z in pnvs2[-1]], st2, 'radius')
		for k in dic:
		    (b, a) = dic[k]
		    par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		    dic[k] = (par,numpy.array(val))
		tc_dict2.append(dic)

		x_full = tc_dict1[0].values()[0][0]
		x_inac = tc_dict2[0].values()[0][0]

		# each cell couple 
		print tc_dict1[0].values()[0][1].shape
		axes[col,1].set_ylabel("Response (spikes/sec)", fontsize=10)
		for j,nid in enumerate(neurons_full[col]):
			# print col,j,nid
			if len(neurons_full[col])>1:
				y_full = tc_dict1[0].values()[0][1][:,j]
				y_inac = tc_dict2[0].values()[0][1][:,j]
			else:
				y_full = tc_dict1[0].values()[0][1]
				y_inac = tc_dict2[0].values()[0][1]
			axes[col,j+1].plot(x_full, y_full, linewidth=2, color='b')
			axes[col,j+1].plot(x_inac, y_inac, linewidth=2, color='r')
			axes[col,j+1].set_title(str(nid), fontsize=10)
		# Population histogram
		diff_full_inac = []
		std_full_inac = []
		smaller = -tc_dict1[0].values()[0][1][0:3]+tc_dict2[0].values()[0][1][0:3] # difference:  cell(inactivated V1) - cell(full)
		equal = -tc_dict1[0].values()[0][1][3:5]+tc_dict2[0].values()[0][1][3:5]
		larger = -tc_dict1[0].values()[0][1][5:]+tc_dict2[0].values()[0][1][5:]
		diff_full_inac.append( numpy.mean(numpy.sum(smaller, axis=1)) / 100 ) # mean percentage difference over cells
		diff_full_inac.append( numpy.mean(numpy.sum(equal, axis=1)) / 100 )
		diff_full_inac.append( numpy.mean(numpy.sum(larger, axis=1)) / 100 )
		std_full_inac.append( numpy.std(smaller)/100 ) 
		std_full_inac.append( numpy.std(equal)/100 ) 
		std_full_inac.append( numpy.std(larger)/100 ) 
		print diff_full_inac
		# print std_full_inac
		# bp = axes[col,0].boxplot(diff_full_inac, vert=True, patch_artist=True, widths=(.9, .9, .9))
		barlist = axes[col,0].bar([0.5,1.5,2.5], diff_full_inac, width=0.8, color='r', yerr=std_full_inac)
		axes[col,0].plot([0,4], [0,0], 'k-') # horizontal 0 line
		barlist[0].set_color('brown')
		barlist[1].set_color('darkgreen')
		barlist[2].set_color('blue')
		# colors = ['brown', 'darkgreen', 'blue']
		# for patch, color in zip(bp['boxes'], colors):
		# 	patch.set_facecolor(color)

	fig.subplots_adjust(hspace=0.4)
	# fig.suptitle("All recorded cells grouped by circular distance", size='xx-large')
	fig.text(0.5, 0.04, 'cells', ha='center', va='center')
	fig.text(0.06, 0.5, 'ranges', ha='center', va='center', rotation='vertical')
	for ax in axes.flatten():
		ax.set_ylim([0,80])
		ax.set_xticks(sizes)
		ax.set_xticklabels([0.1, '', '', '', '', 1, '', 2, 4, 6])

	for col,_ in enumerate(slice_ranges):
		axes[col,0].set_ylim([-.6,.6])
		axes[col,0].set_yticks([-.6, -.4, -.2, 0., .2, .4, .6])
		axes[col,0].set_yticklabels([-60, -40, -20, 0, 20, 40, 60])
		axes[col,0].set_xlim([0,4])
		axes[col,0].set_xticks([.9,1.9,2.9])
		axes[col,0].set_xticklabels(['small', 'equal', 'larger'])
		axes[col,0].spines['right'].set_visible(False)
		axes[col,0].spines['top'].set_visible(False)
		axes[col,0].spines['bottom'].set_visible(False)

	# plt.show()
	plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_step"+str(step)+".png", dpi=100 )
	# plt.savefig( folder_full+"/TrialAveragedSizeTuningComparison_"+sheet+"_"+interval+".png", dpi=100 )
	plt.close()





###################################################
# Execution
import os

full_list = [ 
	"CombinationParamSearch_size_V1_full", 
	"CombinationParamSearch_size_V1_full_more", 
	"CombinationParamSearch_size_V1_full_more2" 
	]

inac_large_list = [ 
	"CombinationParamSearch_size_V1_inhibition_large", 
	"CombinationParamSearch_size_V1_inhibition_large_more", 
	"CombinationParamSearch_size_V1_inhibition_large_more2" 
	]

# inac_small_list = [
# 	"CombinationParamSearch_size_V1_inhibition_small",
# 	"CombinationParamSearch_size_V1_inhibition_small_more",
# 	"CombinationParamSearch_size_V1_inhibition_small_more2"
# 	]


# for i,l in enumerate(full_list):
# 	full = [ l+"/"+f for f in os.listdir(l) if os.path.isdir(os.path.join(l, f)) ]
# 	# small = [ inac_small_list[i]+"/"+f for f in os.listdir(inac_small_list[i]) if os.path.isdir(os.path.join(inac_small_list[i], f)) ]
# 	large = [ inac_large_list[i]+"/"+f for f in os.listdir(inac_large_list[i]) if os.path.isdir(os.path.join(inac_large_list[i], f)) ]

# 	# print "\n\nFull:", i
# 	# print full
# 	# # print "\n\nSmall:", i
# 	# # print small
# 	# print "\n\nLarge:", i
# 	# print large

# 	for i,f in enumerate(full):
# 		print i
# 		print f," vs ",large[i]
# 		perform_comparison_size_tuning( 
# 			sheet='X_ON', 
# 			step=.3,
# 			sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
# 			folder_full=f, 
# 			folder_inactive=large[i]
# 			)




# Testing
perform_comparison_size_tuning( 
	sheet='X_ON', 
	step=.25,
	sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
	folder_full='ThalamoCorticalModel_data_size_____full', 
	folder_inactive='ThalamoCorticalModel_data_size_____inactivated'
	)
