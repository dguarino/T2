# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import mozaik
import mozaik.controller
from parameters import ParameterSet

import numpy
import scipy.stats
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



def select_ids_by_position(position, radius, sheet_ids, positions):
	radius_ids = []
	distances = []
	min_radius = radius[0] # over: 0. # non: 1.
	max_radius = radius[1] # over: .7 # non: 3.

	for i in sheet_ids:
		a = numpy.array((positions[0][i],positions[1][i],positions[2][i]))
		# print a, " - ", position
		l = numpy.linalg.norm(a - position)
		# print "distance",l
		if l>min_radius and l<max_radius:
			# print "taken"
			radius_ids.append(i[0])
			distances.append(l)

	# sort by distance
	# print radius_ids
	# print distances
	return [x for (y,x) in sorted(zip(distances,radius_ids), key=lambda pair: pair[0])]
	# return radius_ids


def perform_comparison_size_tuning( sheet, reference_position, inactivated_position, step, sizes, folder_full, folder_inactive ):
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

	# Inactivated
	dsv2 = queries.param_filter_query( data_store_inac, identifier='PerNeuronValue', sheet_name=sheet )
	pnvs2 = [ dsv2.get_analysis_result() ]
	# get stimuli
	st2 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs2[-1]]

	# rings analysis
	neurons_full = []
	neurons_inac = []
	rowplots = 0
	max_size = 0.6

	slice_ranges = numpy.arange(step, max_size+step, step)
	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step,cur_range]

		# get the list of all recorded neurons in X_ON
		# Full
		spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		positions1 = data_store_full.get_neuron_postions()[sheet]
		sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids1)
		radius_ids1 = select_ids_by_position(reference_position, radius, sheet_ids1, positions1)
		neurons1 = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=radius_ids1)
		if len(neurons1) > rowplots:
			rowplots = len(neurons1)
		neurons_full.append(neurons1)

		# Inactivated
		spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		positions2 = data_store_inac.get_neuron_postions()[sheet]
		sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids2)
		radius_ids2 = select_ids_by_position(reference_position, radius, sheet_ids2, positions2)
		# if we are plotting the effect of non-overlapping cells, 
		# we want to plot the recorded receiving cells response sorted by distance from the inactivated site 
		# (but after having selected them by distance from the center anyway)
		# so re-select them (not for real but just to use the function) from the inactivated_position
		# if inactivated_position != 0:
		# 	radius_ids2 = select_ids_by_position(inactivated_position, radius, sheet_ids2, positions2)

		neurons2 = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=radius_ids2)
		neurons_inac.append(neurons2)

		print "radius_ids", radius_ids2

		assert len(neurons_full[col]) == len(neurons_inac[col]) , "ERROR: the number of recorded neurons is different"
		assert set(neurons_full[col]) == set(neurons_inac[col]) , "ERROR: the neurons in the two arrays are not the same"

	# subplot figure creation
	print 'rowplots', rowplots
	print "Starting plotting ..."
	print len(slice_ranges), slice_ranges
	fig, axes = plt.subplots(nrows=len(slice_ranges), ncols=rowplots+1, figsize=(3*rowplots, 3*len(slice_ranges)), sharey=False)
	print axes.shape

	p_significance = .02
	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step,cur_range]
		interval = str(radius[0]) +" - "+ str(radius[1]) +" deg radius"
		print interval
		axes[col,0].set_ylabel(interval+"\n\nResponse change (%)")
		print "range:",col
		if len(neurons_full[col]) < 1:
			continue
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

		# Plotting tuning curves
		x_full = tc_dict1[0].values()[0][0]
		x_inac = tc_dict2[0].values()[0][0]
		# each cell couple 
		print tc_dict1[0].values()[0][1].shape
		axes[col,1].set_ylabel("Response (spikes/sec)", fontsize=10)
		for j,nid in enumerate(neurons_full[col]):
			# print col,j,nid
			if len(neurons_full[col])>1: # case with just one neuron in the group
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
		sem_full_inac = []
		num_cells = tc_dict1[0].values()[0][1].shape[1]
		smaller_pvalue = 0.
		equal_pvalue = 0.
		larger_pvalue = 0.

		# -------------------------------------
		# NON-PARAMETRIC TWO-TAILED TEST
		# We want to have a summary statistical measure of the population of cells with and without inactivation.
		# Our null-hypothesis is that the inactivation does not change the activity of cells.
		# A different result will tell us that the inactivation DOES something.
		# Therefore our null-hypothesis is the result obtained in the intact system.
		# Procedure:
		# We have several stimulus sizes
		# We want to group them in three: smaller than optimal, optimal, larger than optimal
		# We do the mean response for each cell for the grouped stimuli
		#    i.e. sum the responses for each cell across stimuli in the group, divided by the number of stimuli in the group
		# We pass the resulting array to the wilcoxon (non-parametric two-tailed test)
		# and we get the value for one group.
		# We repeat for each group

		# average of all trial-averaged response for each cell for grouped stimulus size
		diff_smaller = numpy.sum(tc_dict2[0].values()[0][1][0:3], axis=0)/3 - numpy.sum(tc_dict1[0].values()[0][1][0:3], axis=0)/3
		diff_equal = numpy.sum(tc_dict2[0].values()[0][1][3:5], axis=0)/2 - numpy.sum(tc_dict1[0].values()[0][1][3:5], axis=0)/2
		diff_larger = numpy.sum(tc_dict2[0].values()[0][1][5:], axis=0)/5 - numpy.sum(tc_dict1[0].values()[0][1][5:], axis=0)/5

		# and we want to compare the responses of full and inactivated
		smaller, smaller_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][0:3], axis=0)/3, numpy.sum(tc_dict1[0].values()[0][1][0:3], axis=0)/3 )
		equal, equal_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][3:5], axis=0)/2, numpy.sum(tc_dict1[0].values()[0][1][3:5], axis=0)/2 )
		larger, larger_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][5:], axis=0)/5, numpy.sum(tc_dict1[0].values()[0][1][5:], axis=0)/5 )

		print "smaller, smaller_pvalue:", smaller, smaller_pvalue
		print "equal, equal_pvalue:", equal, equal_pvalue
		print "larger, larger_pvalue:", larger, larger_pvalue
		diff_full_inac.append( smaller / 100 ) # percentage
		diff_full_inac.append( equal / 100 )
		diff_full_inac.append( larger / 100 )

		# -------------------------------------
		# Standard Error Mean
		sem_full_inac.append( scipy.stats.sem(diff_smaller) )
		sem_full_inac.append( scipy.stats.sem(diff_equal) )
		sem_full_inac.append( scipy.stats.sem(diff_larger) )
		# sem_full_inac.append( numpy.std(diff_smaller) / numpy.sqrt(num_cells) )
		# sem_full_inac.append( numpy.std(diff_equal) / numpy.sqrt(num_cells) ) 
		# sem_full_inac.append( numpy.std(diff_larger) / numpy.sqrt(num_cells) ) 

		# print diff_full_inac
		# print sem_full_inac
		barlist = axes[col,0].bar([0.5,1.5,2.5], diff_full_inac, width=0.8, color='r', yerr=sem_full_inac)
		axes[col,0].plot([0,4], [0,0], 'k-') # horizontal 0 line
		for ba in barlist:
			ba.set_color('white')
		if smaller_pvalue < p_significance:
			barlist[0].set_color('brown')
		if equal_pvalue < p_significance:
			barlist[1].set_color('darkgreen')
		if larger_pvalue < p_significance:
			barlist[2].set_color('blue')
		# colors = ['brown', 'darkgreen', 'blue']
		# for patch, color in zip(bp['boxes'], colors):
		# 	patch.set_facecolor(color)

	fig.subplots_adjust(hspace=0.4)
	# fig.suptitle("All recorded cells grouped by circular distance", size='xx-large')
	fig.text(0.5, 0.04, 'cells', ha='center', va='center')
	fig.text(0.06, 0.5, 'ranges', ha='center', va='center', rotation='vertical')
	for ax in axes.flatten():
		ax.set_ylim([0,90])
		ax.set_xticks(sizes)
		ax.set_xticklabels([0.1, '', '', '', '', 1, '', 2, 4, 6])

	for col,_ in enumerate(slice_ranges):
		# axes[col,0].set_ylim([-.8,.8])
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
	plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_position"+str(reference_position)+"_step"+str(step)+".png", dpi=100 )
	# plt.savefig( folder_full+"/TrialAveragedSizeTuningComparison_"+sheet+"_"+interval+".png", dpi=100 )
	plt.close()







def perform_comparison_size_inputs( sheet, sizes, folder_full, folder_inactive ):
	print folder_full
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	print "Checking data..."

	analog_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_vm_ids()
	print analog_ids1
	analog_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_vm_ids()
	print analog_ids2

	assert len(analog_ids1) == len(analog_ids2) , "ERROR: the number of recorded neurons is different"
	assert set(analog_ids1) == set(analog_ids2) , "ERROR: the neurons in the two arrays are not the same"

	num_sizes = len( sizes )

	for _,idd in enumerate(analog_ids1):

		# get trial averaged gsyn for each stimulus condition
		# then subtract full - inactive for each stimulus condition (size)
		# then summarize the time differences in one number, to have one point for each size

		# Full
		segs = sorted( 
			param_filter_query(data_store_full, st_name='DriftingSinusoidalGratingDisk', sheet_name=sheet).get_segments(), 
			key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).radius 
		)
		print "full idd", idd # 
		# print len(segs), "/", num_sizes
		trials = len(segs) / num_sizes
		# print trials
		full_gsyn_es = [s.get_esyn(idd) for s in segs]
		full_gsyn_is = [s.get_isyn(idd) for s in segs]
		# print "len full_gsyn_e/i", len(full_gsyn_es) # 61 = 1 spontaneous + 6 trial * 10 num_sizes
		# print "shape gsyn_e/i", full_gsyn_es[0].shape
		# mean input over trials
		mean_full_gsyn_e = numpy.zeros((num_sizes, full_gsyn_es[0].shape[0])) # init
		mean_full_gsyn_i = numpy.zeros((num_sizes, full_gsyn_es[0].shape[0]))
		# print "shape mean_full_gsyn_e/i", mean_full_gsyn_e.shape
		sampling_period = full_gsyn_es[0].sampling_period
		t_stop = float(full_gsyn_es[0].t_stop - sampling_period)
		t_start = float(full_gsyn_es[0].t_start)
		time_axis = numpy.arange(0, len(full_gsyn_es[0]), 1) / float(len(full_gsyn_es[0])) * abs(t_start-t_stop) + t_start
		# sum by size
		t = 0
		for e,i in zip(full_gsyn_es, full_gsyn_is):
			s = int(t/trials)
			e = e.rescale(mozaik.tools.units.nS) #e=e*1000
			i = i.rescale(mozaik.tools.units.nS) #i=i*1000
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] + numpy.array(e.tolist())
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] + numpy.array(i.tolist())
			t = t+1
		# average by trials
		for s in range(num_sizes):
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] / trials
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] / trials
		# print "mean_full_gsyn_e", len(mean_full_gsyn_e), mean_full_gsyn_e
		# print "mean_full_gsyn_i", len(mean_full_gsyn_i), mean_full_gsyn_i

		# Inactivated
		segs = sorted( 
			param_filter_query(data_store_inac, st_name='DriftingSinusoidalGratingDisk', sheet_name=sheet).get_segments(), 
			key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).radius 
		)
		print "inactivation idd", idd # 
		# print len(segs), "/", num_sizes
		trials = len(segs) / num_sizes
		# print trials
		inac_gsyn_es = [s.get_esyn(idd) for s in segs]
		inac_gsyn_is = [s.get_isyn(idd) for s in segs]
		# print "len full_gsyn_e/i", len(inac_gsyn_es) # 61 = 1 spontaneous + 6 trial * 10 num_sizes
		# print "shape gsyn_e/i", inac_gsyn_es[0].shape
		# mean input over trials
		mean_inac_gsyn_e = numpy.zeros((num_sizes, inac_gsyn_es[0].shape[0])) # init
		mean_inac_gsyn_i = numpy.zeros((num_sizes, inac_gsyn_es[0].shape[0]))
		# print "shape mean_inac_gsyn_e/i", mean_inac_gsyn_e.shape
		sampling_period = inac_gsyn_es[0].sampling_period
		t_stop = float(inac_gsyn_es[0].t_stop - sampling_period)
		t_start = float(inac_gsyn_es[0].t_start)
		time_axis = numpy.arange(0, len(inac_gsyn_es[0]), 1) / float(len(inac_gsyn_es[0])) * abs(t_start-t_stop) + t_start
		# sum by size
		t = 0
		for e,i in zip(inac_gsyn_es, inac_gsyn_is):
			s = int(t/trials)
			e = e.rescale(mozaik.tools.units.nS) #e=e*1000
			i = i.rescale(mozaik.tools.units.nS) #i=i*1000
			mean_inac_gsyn_e[s] = mean_inac_gsyn_e[s] + numpy.array(e.tolist())
			mean_inac_gsyn_i[s] = mean_inac_gsyn_i[s] + numpy.array(i.tolist())
			t = t+1
		# average by trials
		for s in range(num_sizes):
			mean_inac_gsyn_e[s] = mean_inac_gsyn_e[s] / trials
			mean_inac_gsyn_i[s] = mean_inac_gsyn_i[s] / trials
		# print "mean_inac_gsyn_e", len(mean_inac_gsyn_e), mean_inac_gsyn_e
		# print "mean_inac_gsyn_i", len(mean_inac_gsyn_i), mean_inac_gsyn_i

		ttest_sizes_e = []
		ttest_sizes_e_err = []
		ttest_sizes_i = []
		ttest_sizes_i_err = []
		for s in range(num_sizes):
			ttest_sizes_e.append( scipy.stats.ttest_rel( mean_full_gsyn_e[s], mean_inac_gsyn_e[s] )[0] )
			ttest_sizes_e_err.append( scipy.stats.sem(mean_full_gsyn_e[s] - mean_inac_gsyn_e[s]) )
			ttest_sizes_i.append( scipy.stats.ttest_rel( mean_full_gsyn_i[s], mean_inac_gsyn_i[s] )[0] )
			ttest_sizes_i_err.append( scipy.stats.sem(mean_full_gsyn_i[s] - mean_inac_gsyn_i[s]) )
		# 
		plt.title("full - inactivated conductances (ttest)")
		plt.ylabel("percentage input change", fontsize=10)
		plt.xlabel("sizes", fontsize=10)
		plt.xticks(sizes, (0.1, '', '', '', '', 1, '', 2, 4, 6))
		plt.errorbar(sizes, ttest_sizes_e, yerr=ttest_sizes_e_err, color='r', linewidth=2)
		plt.errorbar(sizes, ttest_sizes_i, yerr=ttest_sizes_i_err, color='b', linewidth=2)
		plt.savefig( folder_inactive+"/TrialAveragedInputComparison_"+sheet+".png", dpi=100 )
		plt.close()





###################################################
# Execution
import os

full_list = [ 
	# "ThalamoCorticalModel_data_size_____full2"
	"CombinationParamSearch_size_V1_2sites_full13",
	# "CombinationParamSearch_size_V1_full", 
	# "CombinationParamSearch_size_V1_full_more", 
	# "CombinationParamSearch_size_V1_full_more2" 
	]

inac_large_list = [ 
	# "ThalamoCorticalModel_data_size_____inactivated2"
	# "CombinationParamSearch_size_V1_2sites_inhibition_small14",
	"CombinationParamSearch_size_V1_2sites_inhibition_large13",
	# "CombinationParamSearch_size_V1_2sites_inhibition_large_nonoverlapping13",
	# "CombinationParamSearch_size_V1_inhibition_large", 
	# "CombinationParamSearch_size_V1_inhibition_large_more", 
	# "CombinationParamSearch_size_V1_inhibition_large_more2" 
	]


sheets = ['X_ON', 'X_OFF', 'PGN'] #, 'V1_Exc_L4']
steps = [.2, .4]
sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46]

for i,l in enumerate(full_list):
	full = [ l+"/"+f for f in os.listdir(l) if os.path.isdir(os.path.join(l, f)) ]
	# small = [ inac_small_list[i]+"/"+f for f in os.listdir(inac_small_list[i]) if os.path.isdir(os.path.join(inac_small_list[i], f)) ]
	large = [ inac_large_list[i]+"/"+f for f in os.listdir(inac_large_list[i]) if os.path.isdir(os.path.join(inac_large_list[i], f)) ]

	# print "\n\nFull:", i
	# print full
	# # print "\n\nSmall:", i
	# # print small
	# print "\n\nLarge:", i
	# print large

	for i,f in enumerate(full):
		print i

		for s in sheets:

			perform_comparison_size_inputs( 
				sheet=s,
				sizes = sizes,
				folder_full=f, 
				folder_inactive=large[i]
				)

			# for step in steps:

			# 	perform_comparison_size_tuning( 
			# 		sheet=s, 
			# 		reference_position=[[0.0], [0.0], [0.0]],
			# 		inactivated_position=0, # !!!!!!!!!!!!!!
			# 		# inactivated_position=[[1.6], [0.0], [0.0]], # !!!!!!!!!!!!!!
			# 		step=step,
			# 		sizes = sizes,
			# 		folder_full=f, 
			# 		folder_inactive=large[i]
			# 		)




# # Testing
# perform_comparison_size_tuning( 
# 	sheet='X_ON', 
# 	reference_position=[[0.0], [0.0], [0.0]],
#   # inactivated_position=[[1.6], [0.0], [0.0]],
# 	step=.4,
# 	sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
# 	folder_full='ThalamoCorticalModel_data_size_____full2', 
# 	folder_inactive='ThalamoCorticalModel_data_size_____inactivated2'
# 	)
