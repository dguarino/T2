# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import os
import ast
import re
import glob

from functools import reduce # forward compatibility
import operator

import gc
import numpy
import scipy.stats
import scipy
import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import quantities as qt

from parameters import ParameterSet

import mozaik
import mozaik.controller
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.TrialAveragedFiringRateCutout import TrialAveragedFiringRateCutout, SpikeCountCutout
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore

from mozaik.tools.mozaik_parametrized import colapse

import neo




def select_ids_by_position(positions, sheet_ids, position=[], radius=[0,0], box=[], reverse=False):
	selected_ids = []
	distances = []
	min_radius = radius[0]
	max_radius = radius[1]

	for i in sheet_ids:
		a = numpy.array((positions[0][i],positions[1][i],positions[2][i]))

		if len(box)>1:
			# Ex box: [ [-.3,.0], [.3,-.4] ]
			# Ex a: [ [-0.10769224], [ 0.16841423], [ 0. ] ]
			# print box[0][0], a[0], box[1][0], "      ", box[0][1], a[1], box[1][1]
			if a[0]>=box[0][0] and a[0]<=box[1][0] and a[1]>=box[0][1] and a[1]<=box[1][1]:
				selected_ids.append(i[0])
				distances.append(0.0)
		else:
			#print a, " - ", position
			l = numpy.linalg.norm(a - position)

			# print "distance",l
			if abs(l)>min_radius and abs(l)<max_radius:
				# print "taken"
				selected_ids.append(i[0])
				distances.append(l)

	# sort by distance
	# print len(selected_ids)
	# print distances
	return [x for (y,x) in sorted(zip(distances,selected_ids), key=lambda pair:pair[0], reverse=reverse)]




def get_per_neuron_sliding_window_spike_count( datastore, sheet, stimulus, stimulus_parameter, start, stop, window=50.0, step=10.0, neurons=[] ):

	if not len(neurons)>1:
		spike_ids = param_filter_query(datastore, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = datastore.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		neurons = datastore.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)
	print "neurons", len(neurons)

	SpikeCount( 
		param_filter_query( datastore, sheet_name=sheet, st_name=stimulus ), 
		ParameterSet({'bin_length':window}) 
	).analyse()
	# datastore.save()

	TrialMean(
		param_filter_query( datastore, name='AnalogSignalList', analysis_algorithm='SpikeCount' ),
		ParameterSet({'vm':False, 'cond_exc':False, 'cond_inh':False})
	).analyse()
	# datastore.save()

	TrialVariability(
		param_filter_query( datastore, name='AnalogSignalList', analysis_algorithm='SpikeCount' ),
		ParameterSet({'vm':False, 'cond_exc':False, 'cond_inh':False})
	).analyse()
	# datastore.save()

	dsvTM = param_filter_query( datastore, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialMean' )
	dsvTV = param_filter_query( datastore, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialVariability' )
	# dsvTM.print_content(full_recordings=False)
	pnvsTM = [ dsvTM.get_analysis_result() ]
	pnvsTV = [ dsvTV.get_analysis_result() ]
	# print pnvsTM

	# get stimuli from PerNeuronValues
	st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvsTM[-1]]

	dic = colapse_to_dictionary([z.get_asl_by_id(neurons) for z in pnvsTM[-1]], st, stimulus_parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))

	stimuli = dic.values()[0][0]
	means = numpy.array( dic.values()[0][1] )
	# print "means:", means.shape

	dic = colapse_to_dictionary([z.get_asl_by_id(neurons) for z in pnvsTV[-1]], st, stimulus_parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))

	variances = numpy.array( dic.values()[0][1] )
	# print "variances:", variances.shape

	return stimuli, numpy.swapaxes(means,1,2), numpy.swapaxes(variances,1,2)




def get_per_neuron_spike_count( datastore, stimulus, sheet, start, end, stimulus_parameter, bin=10.0, neurons=[], spikecount=True ):
	if not spikecount:
		TrialAveragedFiringRateCutout( 
			param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
			ParameterSet({}) 
		).analyse(start=start, end=end)
	else:
		SpikeCount( 
			param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
			ParameterSet({'bin_length' : bin}) 
		).analyse()
		# datastore.save()

	mean_rates = []
	stimuli = []
	if not isinstance(sheet, list):
		sheet = [sheet]
	for sh in sheet:
		dsv = param_filter_query( datastore, identifier='PerNeuronValue', sheet_name=sh, st_name=stimulus )
		# dsv.print_content(full_recordings=False)
		pnvs = [ dsv.get_analysis_result() ]
		# print pnvs
		# get stimuli from PerNeuronValues
		st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs[-1]]

		if not len(neurons)>1:
			spike_ids = param_filter_query(datastore, sheet_name=sh, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
			sheet_ids = datastore.get_sheet_indexes(sheet_name=sh, neuron_ids=spike_ids)
			neurons = datastore.get_sheet_ids(sheet_name=sh, indexes=sheet_ids)
		print "neurons", len(neurons)

		dic = colapse_to_dictionary([z.get_value_by_id(neurons) for z in pnvs[-1]], st, stimulus_parameter)
		for k in dic:
			(b, a) = dic[k]
			par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
			dic[k] = (par,numpy.array(val))
		# print dic

		if len(mean_rates)>1:
			# print mean_rates.shape, dic.values()[0][1].shape
			mean_rates = numpy.append( mean_rates, dic.values()[0][1], axis=1 )
		else:
			# print dic
			mean_rates = dic.values()[0][1]

		stimuli = dic.values()[0][0]
		neurons = [] # reset, if we are in a loop we don't want the old neurons id to be still present
		print mean_rates.shape

	return mean_rates, stimuli




def size_tuning_comparison( sheet, folder_full, folder_inactive, stimulus, parameter, sizes, reference_position, box=[], csvfile=None, plotAll=False ):
	print folder_full
	folder_nums = re.findall(r'\d+', folder_full)
	print folder_nums
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
	num_stimuli = len(st1)

	# Inactivated
	dsv2 = queries.param_filter_query( data_store_inac, identifier='PerNeuronValue', sheet_name=sheet )
	pnvs2 = [ dsv2.get_analysis_result() ]
	# get stimuli
	st2 = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs2[-1]]

	# rings analysis
	rowplots = 0

	# GET RECORDINGS BY BOX POSITION

	# get the list of all recorded neurons in X_ON
	# Full
	spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	print spike_ids1
	positions1 = data_store_full.get_neuron_postions()[sheet]
	sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids1)
	radius_ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
	neurons_full = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=radius_ids1)
	print neurons_full

	# Inactivated
	spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	positions2 = data_store_inac.get_neuron_postions()[sheet]
	sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids2)
	radius_ids2 = select_ids_by_position(positions2, sheet_ids2, box=box)
	neurons_inac = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=radius_ids2)
	print neurons_inac

	if not set(neurons_full)==set(neurons_inac):
		neurons_full = numpy.intersect1d(neurons_full, neurons_inac)
		neurons_inac = neurons_full

	if len(neurons_full) > rowplots:
		rowplots = len(neurons_full)

	print "neurons_full:", len(neurons_full), neurons_full
	print "neurons_inac:", len(neurons_inac), neurons_inac

	assert len(neurons_full) > 0 , "ERROR: the number of recorded neurons is 0"

	tc_dict1 = []
	tc_dict2 = []

	# Full
	# group values 
	dic = colapse_to_dictionary([z.get_value_by_id(neurons_full) for z in pnvs1[-1]], st1, 'radius')
	for k in dic:
	    (b, a) = dic[k]
	    par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
	    dic[k] = (par,numpy.array(val))
	tc_dict1.append(dic)

	# Inactivated
	# group values 
	dic = colapse_to_dictionary([z.get_value_by_id(neurons_inac) for z in pnvs2[-1]], st2, 'radius')
	for k in dic:
	    (b, a) = dic[k]
	    par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
	    dic[k] = (par,numpy.array(val))
	tc_dict2.append(dic)
	print "tc_dict1",tc_dict1[0].values()[0][1].shape
	print "tc_dict2",tc_dict2[0].values()[0][1].shape

	# print "(stimulus conditions, cells):", tc_dict1[0].values()[0][1].shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)

	# Population histogram
	diff_full_inac = []
	sem_full_inac = []
	num_cells = tc_dict1[0].values()[0][1].shape[1]

	# -------------------------------------
	# DIFFERENCE BETWEEN INACTIVATED AND CONTROL
	# We want to have a summary measure of the population of cells with and without inactivation.
	# Our null-hypothesis is that the inactivation does not change the activity of cells.
	# A different result will tell us that the inactivation DOES something.
	# Therefore our null-hypothesis is the result obtained in the intact system.

	# 1. MASK IN ONLY CHANGING UNITS
	all_closed_values = tc_dict1[0].values()[0][1]
	all_open_values = tc_dict2[0].values()[0][1]

	# 1.1 Search for the units that are NOT changing (within a certain absolute tolerance)
	unchanged_units = numpy.isclose(all_closed_values, all_open_values, rtol=0., atol=10.) # 4 spikes/s
	# print unchanged_units

	# 1.2 Reverse them into those that are changing
	changed_units_mask = numpy.invert( unchanged_units )

	# 1.3 Get indexes for printing
	changed_units = numpy.nonzero( changed_units_mask )
	changing_idxs = zip(changed_units[0], changed_units[1])
	# print changing_idxs

	# 1.4 Mask the array to apply later computations only on visible values
	closed_values = numpy.ma.array( all_closed_values, mask=changed_units_mask )
	open_values = numpy.ma.array( all_open_values, mask=changed_units_mask )
	print "chosen closed units:", closed_values.shape
	print "chosen open units:", open_values.shape
	num_cells = closed_values.shape[1]

	# 2. Automatic search for intervals
	minimums = closed_values.argmin( axis=0 ) #
	minimums = numpy.minimum(minimums, 0)
	print "index of the stimulus triggering the minimal response for each chosen cell:", minimums
	peaks = closed_values.argmax( axis=0 )
	print "index of the stimulus triggering the maximal response for each chosen cell:", peaks
	# larger are 1 more than optimal, clipped to the largest index
	largers = numpy.minimum(peaks+1, len(closed_values)-1)
	print "index of the stimulus after the maximal (+1) for each chosen cell:", largers

	# 3. Calculate difference (data - control)
	diff_smaller = numpy.array([open_values[s][c] for c,s in enumerate(minimums)]) - numpy.array([closed_values[s][c] for c,s in enumerate(minimums)])
	diff_equal = numpy.array([open_values[s][c] for c,s in enumerate(peaks)]) - numpy.array([closed_values[s][c] for c,s in enumerate(peaks)])
	# we have to get for each cell the sum of all its results for all stimulus conditions larger than peak
	sum_largers_open = numpy.zeros(num_cells)
	sum_largers_closed = numpy.zeros(num_cells)
	for c,l in enumerate([open_values[s:][:,c] for c,s in enumerate(largers)]):
		sum_largers_open[c] = sum(l)
	for c,l in enumerate([closed_values[s:][:,c] for c,s in enumerate(largers)]):
		sum_largers_closed[c] = sum(l)
	diff_larger = sum_largers_open - sum_largers_closed

	# 3.1 get sign over all cells
	sign_smaller = numpy.sign( sum(diff_smaller) )
	sign_equal = numpy.sign( sum(diff_equal) )
	sign_larger = numpy.sign( sum(diff_larger) )
	print "sign smaller",sign_smaller
	print "sign equal", sign_equal
	print "sign larger", sign_larger

	# 3.2 Standard Error Mean calculated on the difference
	sem_full_inac.append( scipy.stats.sem(diff_smaller) )
	sem_full_inac.append( scipy.stats.sem(diff_equal) )
	sem_full_inac.append( scipy.stats.sem(diff_larger) )
	print "SEM: ", sem_full_inac

	# 4. Compute Wilcoxon Test, given in percentage to the maximum possible (W statistics)
	smaller, p_smaller = scipy.stats.wilcoxon( diff_smaller )
	equal, p_equal = scipy.stats.wilcoxon( diff_equal )
	larger, p_larger = scipy.stats.wilcoxon( diff_larger )
	# this test uses W statistics: the maximum possible value is the sum from 1 to N.
	norm = numpy.sum( numpy.arange( diff_smaller.shape[0] ) ) # W-statistics
	# percentage of change
	perc_smaller = sign_smaller * smaller/norm *100
	perc_equal = sign_equal * equal/norm *100
	perc_larger = sign_larger * larger/norm *100
	print "Wilcoxon for smaller", perc_smaller, "p-value:", p_smaller
	print "Wilcoxon for equal", perc_equal, "p-value:", p_equal
	print "Wilcoxon for larger", perc_larger, "p-value:", p_larger
	diff_full_inac.append( perc_smaller )
	diff_full_inac.append( perc_equal )
	diff_full_inac.append( perc_larger )

	if csvfile:
		csvrow = ",".join(folder_nums)+",("+ str(smaller)+ ", " + str(equal)+ ", " + str(larger)+ "), "
		print csvrow
		csvfile.write( csvrow )

	if not plotAll:
		# single figure creation
		print "Starting plotting ..."
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()
		barlist = ax.bar([0.5,1.5,2.5], diff_full_inac, width=0.8)
		barlist[0].set_color('brown')
		barlist[1].set_color('darkgreen')
		barlist[2].set_color('blue')
		ax.errorbar(0.9, diff_full_inac[0], sem_full_inac[0], color='brown', capsize=20, capthick=3, elinewidth=3 )
		ax.errorbar(1.9, diff_full_inac[1], sem_full_inac[1], color='darkgreen', capsize=20, capthick=3, elinewidth=3 )
		ax.errorbar(2.9, diff_full_inac[2], sem_full_inac[2], color='blue', capsize=20, capthick=3, elinewidth=3 )
		ax.plot([0,4], [0,0], 'k-') # horizontal 0 line
		ax.set_ylim([-60,60])
		ax.set_yticks([-60, -40, -20, 0., 20, 40, 60])
		ax.set_yticklabels([-60, -40, -20, 0, 20, 40, 60])
		ax.set_xlim([0,4])
		ax.set_xticks([.9,1.9,2.9])
		ax.set_xticklabels(['small', 'equal', 'larger'])
		ax.set_ylabel("Response change (%)")
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		plt.tight_layout()
		# plt.show()
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_box"+str(box)+"only_bars.png", dpi=300, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()

	else:
		# subplot figure creation
		print 'rowplots', rowplots
		print "Starting plotting ..."
		fig, axes = plt.subplots(nrows=2, ncols=rowplots+1, figsize=(3*rowplots, 5), sharey=False)
		# print axes.shape
		axes[0,0].set_ylabel("Response change (%)")

		barlist = axes[0,0].bar([0.5,1.5,2.5], diff_full_inac, width=0.8)
		barlist[0].set_color('brown')
		barlist[1].set_color('darkgreen')
		barlist[2].set_color('blue')
		axes[0,0].errorbar(0.9, diff_full_inac[0], sem_full_inac[0], color='brown')
		axes[0,0].errorbar(1.9, diff_full_inac[1], sem_full_inac[1], color='darkgreen')
		axes[0,0].errorbar(2.9, diff_full_inac[2], sem_full_inac[2], color='blue')
		axes[0,0].plot([0,4], [0,0], 'k-') # horizontal 0 line

		# Plotting tuning curves
		x_full = tc_dict1[0].values()[0][0]
		x_inac = tc_dict2[0].values()[0][0]
		# each cell couple 
		axes[0,1].set_ylabel("Response (spikes/sec)", fontsize=10)
		for j,nid in enumerate(neurons_full[changing_idxs]):
			# print col,j,nid
			if len(neurons_full[changing_idxs])>1: # case with just one neuron in the group
				y_full = closed_values[:,j]
				y_inac = open_values[:,j]
			else:
				y_full = closed_values
				y_inac = open_values

			axes[0,j+1].plot(x_full, y_full, linewidth=2, color='b')
			axes[0,j+1].plot(x_inac, y_inac, linewidth=2, color='r')
			axes[0,j+1].set_title(str(nid), fontsize=10)
			axes[0,j+1].set_xscale("log")

		fig.subplots_adjust(hspace=0.4)
		# fig.suptitle("All recorded cells grouped by circular distance", size='xx-large')
		fig.text(0.5, 0.04, 'cells', ha='center', va='center')
		fig.text(0.06, 0.5, 'ranges', ha='center', va='center', rotation='vertical')
		for ax in axes.flatten():
			ax.set_ylim([0,60])
			ax.set_xticks(sizes)
			ax.set_xticklabels([0.1, '', '', '', '', 1, '', 2, 4, 6])
			# ax.set_xticklabels([0.1, '', '', '', '', '', '', '', '', '', '', 1, '', '', 2, '', '', '', 4, '', 6])

		axes[0,0].set_ylim([-60,60])
		axes[0,0].set_yticks([-60, -40, -20, 0., 20, 40, 60])
		axes[0,0].set_yticklabels([-60, -40, -20, 0, 20, 40, 60])
		axes[0,0].set_xlim([0,4])
		axes[0,0].set_xticks([.9,1.9,2.9])
		axes[0,0].set_xticklabels(['small', 'equal', 'larger'])
		axes[0,0].spines['right'].set_visible(False)
		axes[0,0].spines['top'].set_visible(False)
		axes[0,0].spines['bottom'].set_visible(False)

		# plt.show()
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_box"+str(box)+".png", dpi=150, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()




def mean_confidence_interval(data, confidence=0.95):
	import scipy.stats
	a = 1.0*numpy.array(data)
	n = len(a)
	m, se = numpy.mean(a, axis=0), scipy.stats.sem(a, axis=0)
	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
	return m, m-h, m+h




def variability( sheet, folder, stimulus, stimulus_parameter ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	# dsv = param_filter_query(data_store,st_name=stimulus,sheet_name=sheet)
	# dsv.print_content(full_recordings=False)

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()

	if sheet=='V1_Exc_L4' and stimulus_parameter=='orientation':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < 0.1)[0]]
		print "# of V1 cells having orientation 0:", len(l4_exc_or_many)
		spike_ids = list(l4_exc_or_many)

	if sheet=='V1_Exc_L4' and stimulus_parameter=='radius':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < 0.1)[0]]
		position_V1 = data_store.get_neuron_postions()['V1_Exc_L4']
		V1_sheet_ids = data_store.get_sheet_indexes(sheet_name='V1_Exc_L4',neuron_ids=l4_exc_or_many)
		radius_V1_ids = select_ids_by_position(position_V1, V1_sheet_ids, box=[[-.5,-.5],[.5,.5]])
		radius_V1_ids = data_store.get_sheet_ids(sheet_name='V1_Exc_L4',indexes=radius_V1_ids)
		print "# of V1 cells within radius range having orientation 0:", len(radius_V1_ids)
		spike_ids = radius_V1_ids

	# Raster + Histogram
	# RasterPlot(
	# 	param_filter_query(data_store,st_name=stimulus,sheet_name=sheet),
	# 	ParameterSet({'sheet_name' : sheet, 'neurons' : spike_ids, 'trial_averaged_histogram': True, 'spontaneous' : False}),
	# 	fig_param={'dpi' : 100,'figsize': (200,100)},
	# 	plot_file_name=folder+"/HistRaster_"+sheet+"_"+stimulus+".png"
	# ).plot({'SpikeRasterPlot.group_trials':True})

	stimuli, means, variances = get_per_neuron_sliding_window_spike_count( data_store, sheet, stimulus, stimulus_parameter, 100.0, 2000.0, window=20.0, step=10.0, neurons=spike_ids )
	print "means:", means.shape
	print "variances:", variances.shape

	# # OVERALL CELL COUNT
	# # as in AndolinaJonesWangSillito2007 
	# color = 'black'
	# fanos = numpy.mean(variances, axis=1) / numpy.mean(means, axis=1)
	# print "fanos shape: ",fanos.shape
	# for i,ff in enumerate(fanos):
	# 	print "max ff: ", numpy.amax(ff) 
	# 	print "min ff: ", numpy.amin(ff) 
	# 	counts, bin_edges = numpy.histogram( ff, bins=10 )
	# 	print counts
	# 	bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
	# 	# Plot
	# 	fig = plt.figure(1)
	# 	plt.errorbar(bin_centres, counts, fmt='o', color=color)
	# 	# Gaussian fit
	# 	mean = numpy.mean(ff)
	# 	variance = numpy.var(ff)
	# 	sigma = numpy.sqrt(variance)
	# 	x = numpy.linspace(min(ff), max(ff), 300)
	# 	dx = bin_edges[1] - bin_edges[0]
	# 	scale = len(ff)*dx
	# 	plt.xlim([0,3])
	# 	plt.ylim([0,160])
	# 	plt.plot(x, mlab.normpdf(x, mean, sigma)*scale, color=color, linewidth=2 )
	# 	plt.xlabel("Fano Factor")
	# 	plt.ylabel("Number of Cells")
	# 	plt.savefig( folder+"/TrialConditionsAveraged_FanoHist_"+stimulus_parameter+"_"+sheet+"_"+str(i)+".png", dpi=200 )
	# 	plt.close()

	# TIME COURSE
	# fig,ax = plt.subplots(nrows=len(stimuli), ncols=means.shape[1], figsize=(70, 20), sharey=False, sharex=False)
	# fig.tight_layout()

	for i,s in enumerate(stimuli):
		print "stimulus:",s

		csvfile = open(folder+"/TrialAveragedMeanVariance_"+sheet+"_"+stimulus_parameter+"_"+str(s)+".csv", 'w')

		# each column is a different time bin
		for t in range(means.shape[1]):
			# # print "\nTime bin:", t
			# # each row is a different stimulus
			# # print means[i][t].shape 
			# # print variances[i][t].shape
			# ax[i,t].scatter( means[i][t], variances[i][t], marker="o", facecolor="k", edgecolor="w" )

			# x0,x1 = ax[i,t].get_xlim()
			# y0,y1 = ax[i,t].get_ylim()
			# # to make it squared
			# if x1 >= y1:
			# 	y1 = x1
			# else:
			# 	x1 = y1

			# ax[i,t].tick_params(axis='both', which='both', labelsize=8)
			# ax[i,t].set_xlim( (x0,x1) )
			# ax[i,t].set_ylim( (y0,y1) )
			# ax[i,t].set_aspect( abs(x1-x0)/abs(y1-y0) )
			# # add diagonal
			# ax[i,t].plot( [x0,x1], [y0,y1], linestyle='--', color="orange" )

			# replace zeros with very small positive values (arbitrary solution to avoid polyfit crash)
			if not numpy.any(means[i][t]):
				means[i][t][ means[i][t]==0. ] = 0.000000001
			if not numpy.any(variances[i][t]):
				variances[i][t][ variances[i][t]==0. ] = 0.000000001

			# add regression line, whose slope is the Fano Factor
			k,b = numpy.polyfit( means[i][t], variances[i][t], 1)
			# x = numpy.arange(x0, x1)
			# ax[i,t].plot(x, k*x+b, 'k-')
			# ax[i,t].set_title( "Fano:{:.2f}".format(k), fontsize=8 )

			if csvfile:
				csvfile.write( "{:.3f}\n".format(k) )

			# # text
			# ax[i,t].set_xlabel( "window "+str(t), fontsize=8 )
			# ax[i,t].set_ylabel( "{:.2f} \nCount variance".format(s), fontsize=8 )

		csvfile.close()

	# plt.savefig( folder+"/TrialAveragedMeanVariance_"+stimulus_parameter+"_"+sheet+".png", dpi=200, transparent=True )
	# fig.clf()
	# plt.close()
	# # garbage
	# gc.collect()




def fano_comparison_timecourse( closed_files, open_files, sheet, folder, ylim, add_closed_files=[], add_open_files=[] ):
	closed_data = []
	for i,c in enumerate(closed_files):
		closed_data.append( numpy.genfromtxt(c, delimiter='\n') )
		if len(add_closed_files)>1:
			closed_data[i] = closed_data[i] + numpy.genfromtxt(add_closed_files[i], delimiter='\n') 

	# print closed_data
	open_data = []
	for i,o in enumerate(open_files):
		open_data.append( numpy.genfromtxt(o, delimiter='\n') )
		if len(add_open_files)>1:
			open_data[i] = open_data[i] + numpy.genfromtxt(add_open_files[i], delimiter='\n') 
	# print open_data

	# PLOTTING
	x = numpy.arange( len(closed_data[0]) )
	matplotlib.rcParams.update({'font.size':22})

	for i,(c,o) in enumerate(zip(closed_data,open_data)): 
		print "stimulus", i

		# replace wrong polyfit with interpolated values
		err = numpy.argwhere(c>20) # indices of too big values
		c[err] = numpy.nan
		nans, idsf = numpy.isnan(c), lambda z: z.nonzero()[0] # logical indices of too big values, function with signature indices to converto logical indices to equivalent indices
		c[nans] = numpy.interp( idsf(nans), idsf(~nans), c[~nans] )

		err = numpy.argwhere(o>20) # indices of too big values
		o[err] = numpy.nan
		nans, idsf = numpy.isnan(o), lambda z: z.nonzero()[0] # logical indices of too big values, function with signature indices to converto logical indices to equivalent indices
		o[nans] = numpy.interp( idsf(nans), idsf(~nans), o[~nans] )

		std_c = numpy.std(c, ddof=1) 
		err_max_c = c + std_c
		err_min_c = c - std_c
		std_o = numpy.std(o, ddof=1) 
		err_max_o = o + std_o
		err_min_o = o - std_o

		fig,ax = plt.subplots()
		ax.plot((0, len(closed_data[0])), (0,0), color="black")
		ax.plot( x, c, linewidth=2, color='blue' )
		ax.fill_between(x, err_max_c, err_min_c, color="blue", alpha=0.3)
		ax.plot( x, o, linewidth=2, color="cyan" )
		ax.fill_between(x, err_max_o, err_min_o, color="cyan", alpha=0.3)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.set_xlabel("Time bin (10 ms)")
		ax.set_ylabel("Fano Factor")
		plt.tight_layout()
		plt.ylim(ylim)
		plt.savefig( folder+"/fano_comparison"+str(sheet)+"_"+str(i)+".png", dpi=300, transparent=True )
		plt.close()




def trial_averaged_corrected_xcorrelation( sheet, folder, stimulus, start, end, bin=10.0, xlabel="", ylabel="" ):
	# following Brody1998a: "Correlations Without Synchrony"
	# shuffe-corrected cross-correlogram
	from scipy import signal

	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=False)

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
	positions = data_store.get_neuron_postions()[sheet]
	neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)
	print "neurons", len(neurons)

	# compute binned spiketrains per cell per trial
	SpikeCount( 
		param_filter_query(data_store, sheet_name=sheet, st_name=stimulus), 
		ParameterSet({'bin_length' : bin }) 
	).analyse()

	dsvS = param_filter_query( data_store, identifier='AnalogSignalList', sheet_name=sheet, st_name=stimulus, analysis_algorithm=['SpikeCount'] )
	# dsvS.print_content(full_recordings=False)

	# get S1 binned spiketrains
	box1 = [[-3,-.5],[-1.,.5]]
	ids1 = select_ids_by_position(positions, sheet_ids, box=box1)
	neurons1 = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)
	s1 = []
	for idd in sorted(neurons1):
		# print idd
		s1.append( [ads.get_asl_by_id(idd).magnitude for ads in dsvS.get_analysis_result()] ) # 3D (cells, counts, trials)
	S1 = numpy.array(s1)
	P1 = numpy.mean( numpy.array(s1), axis=1 )
	datapoints = P1.shape[1] # actual number of points to show after convolution, centered on its middle point
	print "datapoints", datapoints
	print "S1",S1.shape
	print "P1",P1.shape
	# print P1

	# get S2 binned spiketrains
	box2 = [[1.,-.5],[3.,.5]]
	ids2 = select_ids_by_position(positions, sheet_ids, box=box2)
	neurons2 = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids2)
	s2 = []
	for idd in sorted(neurons2):
		s2.append( [ads.get_asl_by_id(idd).magnitude for ads in dsvS.get_analysis_result()] )
	S2 = numpy.array(s2)
	P2 = numpy.mean( numpy.array(s2), axis=1 )
	print "S2",S2.shape
	print "P2",P2.shape
	# print S2

	# take the smallest number of ids
	num_ids = len(S1)
	if num_ids > len(S2):
		num_ids = len(S2)
	# print num_ids

	# correlation of one-vs-one cell 
	c = []
	k = []
	for i in range(num_ids):
		# cross-correlations per cell, trials
		# "full": The output is the full discrete linear cross-correlation of the inputs. (Default)
		# "fill": pad input arrays with fillvalue. (default)
		# "fillvalue": Value to fill pad input arrays with. Default is 0.
		c.append( signal.correlate2d(S1[i], S2[i], mode='full', boundary='fill', fillvalue=0) )
		# Shuffle corrector (aka K)
		k.append( numpy.correlate(P1[i], P2[i], "full") )

	C = numpy.array(c)
	K = numpy.array(k)
	print "C = S1*S2 ", C.shape
	print "K = P1*P2 ", K.shape
	# print K

	# the covariogram is the average of single-trials correlations minus the correlation of trial-averaged PSTHs
	R = numpy.mean( C, axis=1 )
	print "R = <S1*S2> ", R.shape

	V = R-K
	print "V = R - K", V.shape
	# print V
	# Vmean = numpy.mean( R-K, axis=0)

	timescale_side = int(datapoints/2)

	# Significance interval: (quotation from Brody1998a)
	# Significant departures of V from zero indicate that the two cells
	# were not independent, regardless of what distributions that S1 and S2 were
	# drawn from. Estimating the significance of departures of V from 0 requires
	# some assumptions. For the null hypothesis, it will be assumed that S1 is
	# independent of S2, different trials of S1 are independent of each other, and
	# different bins within each trial of S1 are independent of each other (similar
	# assumptions for the trials and bins of S2 will also be made). If P and sigma2
	# are the mean and variance of S over trials r and N trials is the number of
	# trials in the experiment, then the variance in the null hypothesis for V is (2.4)
	Vmean, err_max, err_min = mean_confidence_interval(V)
	print "Vmean", Vmean.shape
	print "err_max", err_max.shape
	print "err_min", err_min.shape

	# Plotting
	x = range(len(Vmean[timescale_side:datapoints+timescale_side]))
	fig,ax = plt.subplots()
	ax.plot(x, Vmean[timescale_side:datapoints+timescale_side], color="black", linewidth=3)
	ax.fill_between(x, err_max[timescale_side:datapoints+timescale_side], err_min[timescale_side:datapoints+timescale_side], color='grey', alpha=0.3)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_ylim([-10,40])
	ax.set_xlim([0, datapoints])
	ax.set_xticks([0, timescale_side, datapoints])
	ax.set_xticklabels([timescale_side*-1,0,timescale_side])
	ax.set_title( sheet )
	ax.set_xlabel( xlabel )
	ax.set_ylabel( ylabel )
	plt.savefig( folder+"/Covariogram_"+stimulus+"_"+sheet+".png", dpi=200, transparent=True )
	fig.clf()
	plt.close()
	gc.collect()

	for i in range(num_ids):
		# Plotting
		fig,ax = plt.subplots()
		ax.plot(range(len(V[i][timescale_side:datapoints+timescale_side])), V[i][timescale_side:datapoints+timescale_side], color="black", linewidth=3)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		ax.set_ylim([-10,40])
		ax.set_xlim([0, datapoints])
		ax.set_xticks([0, timescale_side, datapoints])
		ax.set_xticklabels([timescale_side*-1,0,timescale_side])
		ax.set_title( sheet )
		ax.set_xlabel( xlabel )
		ax.set_ylabel( ylabel )
		plt.savefig( folder+"/Covariogram_"+stimulus+"_"+sheet+"_"+str(i)+".png", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		gc.collect()




def end_inhibition_barplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None, csvfile=None ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=True)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False  )
	print rates.shape # (stimuli, cells)
	print stimuli
	width = 1.
	ind = numpy.arange(10)

	# END-INHIBITION as in MurphySillito1987 and AlittoUsrey2008:
	# "The responses of the cell with corticofugal feedback are totally suppressed at bar lenghts of 2deg and above, 
	#  and those of cell lacking feedback are reduced up to 40% at bar lenghts of 8deg and above."

	# 1. find the peak response at large sizes
	peaks = numpy.amax(rates, axis=0) # as in AlittoUsrey2008
	# print peaks

	# 2. compute average response at large sizes
	plateaus = numpy.mean( rates[5:], axis=0) 
	# print plateaus

	# 3. compute the difference from peak 
	# ends = (1-plateaus)/peaks # as in AlittoUsrey2008
	ends = (peaks-plateaus)/peaks # as in MurphySillito1987
	print ends

	# 4. group cells by end-inhibition
	hist, edges = numpy.histogram( ends, bins=10, range=(0.0,1.0) )
	rawmean = numpy.mean(ends)
	print rawmean
	mean = (rawmean*10)-1.
	print mean
	print hist

	if csvfile:
		folder_nums = re.findall(r'\d+', folder)
		print folder_nums
		csvrow = ",".join(folder_nums)+",("+str(rawmean)+"), "
		print csvrow
		csvfile.write( csvrow )

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		data_mean = data_list[0] -width/2 # positioning
		data_list = data_list[1:]
		print data_mean
		print data_list

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	if closed:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='blue', edgecolor='blue')
		ax.plot((mean, mean), (0,150), 'b--', linewidth=2)
	else:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='cyan', edgecolor='cyan')
		ax.plot((mean, mean), (0,150), 'c--', linewidth=2)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([ind[0]-width/2, ind[-1], 0, 150])
	ax.set_xticks(ind) 
	ax.set_xticklabels(('0.1','','','','0.5','','','','','1.0'))
	if data: # in front of the synthetic
		if closed:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='black', edgecolor='black')
			ax.plot((data_mean, data_mean), (0,150), 'k--', linewidth=2)
		else:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='grey', edgecolor='grey')
			ax.plot((data_mean, data_mean), (0,150), '--', linewidth=2, color='grey')
	plt.tight_layout()
	plt.savefig( folder+"/suppression_index_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()


from scipy.optimize import curve_fit
def NakaRushton(c, n, Rmax, c50, m):
	return Rmax * (c**n / (c**n + c50**n)) + m
def cumulative_distribution_C50_curve( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", data="", data_color="" ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False )
	print rates.shape # (stimuli, cells)
	# print rates
	# print stimuli

	# Naka-Rushton fit to find the c50 of each cell
	c50 = []
	for i,r in enumerate(numpy.transpose(rates)):
		Rmax = numpy.amax(r) + (numpy.amax(r)/100*10) # 
		# popt, pcov = curve_fit(NakaRushton, numpy.asarray(stimuli), r, maxfev=10000000 ) # workaround for scipy < 0.17
		popt, pcov = curve_fit(NakaRushton, numpy.asarray(stimuli), r, method='trf', bounds=((-numpy.inf,.0,.0,-numpy.inf),(numpy.inf,Rmax,100.,numpy.inf)) ) 
		c50.append( popt[2] ) # c50 fit
		# print popt
		# plt.plot(stimuli, r, 'b-', label='data')
		# plt.plot(stimuli, NakaRushton(r, *popt), 'r-', label='fit')
		# plt.savefig( folder+"/NakaRushton_fit_"+str(sheet)+"_"+str(i)+".png", dpi=100 )
		# plt.close()
	c50s = numpy.array(c50) 
	print c50s

	# count how many c50s are for each stimulus variation
	hist, bin_edges = numpy.histogram(c50s, bins=len(stimuli), density=True)
	print "histogram", hist, bin_edges
	# cumulative sum representation
	cumulative = numpy.cumsum(hist)
	cumulative = cumulative / numpy.amax(cumulative)
	print "cumulative", cumulative

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		print data_list

	# PLOTTING
	x = [0.0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60]
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	if data:
		ax.plot( x, data_list, color=data_color, linewidth=2 )
	ax.plot( x, cumulative, color=color, linewidth=2 )
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.xlim([0,.6])
	plt.ylim([0-.05, 1+.05])
	plt.tight_layout()
	plt.savefig( folder+"/cumulative_distribution_C50_"+str(sheet)+".png", dpi=300, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def orientation_bias_barplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False )
	print rates.shape # (stimuli, cells)
	print stimuli

	width = 1.
	ind = numpy.arange(7)

	# ORIENTATION BIAS as in VidyasagarUrbas1982:
	# For each cell
	# 1. find the peak response (optimal orientation response)
	peaks = numpy.amax(rates, axis=0)
	# print peaks
	# 2. find the index of peak response (stimulus preferred orientation)
	preferred = numpy.argmax(rates, axis=0)
	# print preferred
	# 3. find the response to opposite stimulus orientation
	opposite = (preferred + len(stimuli)/2) % len(stimuli)
	# print opposite
	cell_indexes = numpy.arange(len(opposite))
	trough = []
	for c in cell_indexes:
		trough.append( rates[opposite[c]][c] )
	# print trough
	# 4. compute the bias ratio
	bias = peaks/trough
	print bias
	# 5. group cells by orientation bias
	hist, edges = numpy.histogram( bias, bins=7, range=(1.0,7.0) )
	mean = numpy.mean(bias) -width # -1 because the ticks start at 0
	print mean
	print hist

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		data_mean = data_list[0] -width # positioning
		data_list = data_list[1:]
		print data_mean
		print data_list

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	if closed:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='blue', edgecolor='blue')
		ax.plot((mean, mean), (0,250), 'b--', linewidth=2)
	else:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='cyan', edgecolor='cyan')
		ax.plot((mean, mean), (0,250), 'c--', linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([ind[0]-width/2, ind[-1], 0, 250])
	ax.set_xticks(ind) 
	ax.set_xticklabels(('1','2','3','4','5','6','7'))
	if data: # in front of the synthetic
		if closed:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='black', edgecolor='black')
			ax.plot((data_mean, data_mean), (0,250), 'k--', linewidth=2)
		else:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='grey', edgecolor='grey')
			ax.plot((data_mean, data_mean), (0,250), '--', linewidth=2, color='grey')
	plt.tight_layout()
	plt.savefig( folder+"/orientation_bias_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], box=False, data=None, data_curve=True ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons = []
	if box:
		spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		box1 = [[-.5,-.5],[.5,.5]]
		ids1 = select_ids_by_position(positions, sheet_ids, box=box1)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	if sheet=='V1_Exc_L4' and parameter=='orientation':
		spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in spike_ids]) < 0.1)[0]]
		neurons = list(l4_exc_or_many)

	print "neurons:", len(neurons)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=False ) 
	# print rates

	# compute per-trial mean rate over cells
	mean_rates = numpy.mean(rates, axis=1) 
	# print "Ex. collapsed_mean_rates: ", mean_rates.shape
	# print "Ex. collapsed_mean_rates: ", mean_rates
	std_rates = numpy.std(rates, axis=1, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	# print "Ex. collapsed_std_rates: ", std_rates

	# print "stimuli: ", stimuli
	print "final means and stds: ", mean_rates, std_rates
	# print sorted( zip(stimuli, mean_rates, std_rates) )
	final_sorted = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates, std_rates) ) ) ]

	if percentile:
		firing_max = numpy.amax( final_sorted[1] )
		final_sorted[1] = final_sorted[1] / firing_max * 100

	# Plotting tuning curve
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()

	if data:
		data_list = numpy.genfromtxt(data, delimiter=',', filling_values=None)
		print data_list.shape
		if data_curve:
			# bootstrap taking the data as limits for uniform random sampling (10 samples)
			data_rates = []
			for cp in data_list:
				data_rates.append( numpy.random.uniform(low=cp[0], high=cp[1], size=(10)) )
			data_rates = numpy.array(data_rates)
			# means as usual
			data_mean_rates = numpy.mean(data_rates, axis=1) 
			# print data_mean_rates
			if percentile:
				firing_max = numpy.amax( data_mean_rates )
				data_mean_rates = data_mean_rates / firing_max * 100
			data_std_rates = numpy.std(data_rates, axis=1, ddof=1) 
			# print data_std_rates
			ax.plot( stimuli, data_mean_rates, color='black', label='data' )
			data_err_max = data_mean_rates + data_std_rates
			data_err_min = data_mean_rates - data_std_rates
			ax.fill_between(stimuli, data_err_max, data_err_min, color='black', alpha=0.6)
		else:
			print stimuli, data_list[:,0], data_list[:,1]
			ax.scatter(stimuli, data_list[:,0], marker="o", s=80, facecolor="black", alpha=0.6, edgecolor="white")
			ax.scatter(stimuli, data_list[:,1], marker="D", s=80, facecolor="black", alpha=0.6, edgecolor="white")

	ax.plot( final_sorted[0], final_sorted[1], color=color, label=sheet )
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	if useXlog:
		ax.set_xscale("log", nonposx='clip')
	if useYlog:
		ax.set_yscale("log", nonposy='clip')

	err_max = final_sorted[1] + final_sorted[2]
	err_min = final_sorted[1] - final_sorted[2]
	ax.fill_between(final_sorted[0], err_max, err_min, color=color, alpha=0.3)

	if len(ylim)>1:
		ax.set_ylim(ylim)

	if percentile:
		ax.set_ylim([0,100+numpy.amax(final_sorted[2])+10])

	# text
	ax.set_xlabel( xlabel )
	if percentile:
		ylabel = "Percentile " + ylabel
		sheet = str(sheet) + "_percentile"
	ax.set_ylabel( ylabel )
	# ax.legend( loc="lower right", shadow=False )
	plt.tight_layout()
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+stimulus+"_"+parameter+"_"+str(sheet)+".png", dpi=200, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def pairwise_scatterplot( sheet, folder_full, folder_inactive, stimulus, stimulus_band, parameter, start, end, xlabel="", ylabel="", withRegression=True, withCorrCoef=True, withCentroid=False, withPassIndex=False, withPeakIndex=False, withHighIndex=False, reference_band=3, xlim=[], ylim=[], data_full="", data_inac="", data_marker="D" ):
	print "folder_full: ",folder_full
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print "folder_inactive: ",folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	# # Verify index identity
	# spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	# sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids1)
	# print sheet_ids1
	# spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	# sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids2)
	# print sheet_ids2

	x_full_rates, stimuli_full = get_per_neuron_spike_count( data_store_full, stimulus, sheet, start, end, parameter, spikecount=False )
	x_inac_rates, stimuli_inac = get_per_neuron_spike_count( data_store_inac, stimulus, sheet, start, end, parameter, spikecount=False )
	# x_full_rates, stimuli_full = get_per_neuron_spike_count( data_store_full, stimulus, sheet, start, end, parameter, spikecount=True )
	# x_inac_rates, stimuli_inac = get_per_neuron_spike_count( data_store_inac, stimulus, sheet, start, end, parameter, spikecount=True )

	x_full = x_full_rates[stimulus_band]
	x_inac = x_inac_rates[stimulus_band]
	# print x_full.shape

	if withPassIndex:
		reference_full = x_full_rates[reference_band]
		reference_inac = x_inac_rates[reference_band]
		x_full = x_full / reference_full
		x_inac = x_inac / reference_inac
		print "SEM", scipy.stats.sem(x_full)
		print "SEM", scipy.stats.sem(x_inac)

	if withPeakIndex:
		index_full = numpy.argmax(x_full_rates, axis=0).astype(int)
		index_inac = numpy.argmax(x_inac_rates, axis=0).astype(int)
		print "SEM", scipy.stats.sem(index_full)
		print "SEM", scipy.stats.sem(index_inac)
		x_full = numpy.take( stimuli_full, index_full )
		x_inac = numpy.take( stimuli_inac, index_inac )

	if withHighIndex:
		index_full = numpy.argmax(x_full_rates[stimulus_band:,], axis=0).astype(int)
		index_inac = numpy.argmax(x_inac_rates[stimulus_band:,], axis=0).astype(int)
		print "SEM", scipy.stats.sem(index_full)
		print "SEM", scipy.stats.sem(index_inac)
		x_full = numpy.take( stimuli_full, index_full )
		x_inac = numpy.take( stimuli_inac, index_inac )

	print x_full
	print x_inac
	# read external data to plot as well
	if data_full and data_inac:
		data_full_list = numpy.genfromtxt(data_full, delimiter='\n')
		print data_full_list
		data_inac_list = numpy.genfromtxt(data_inac, delimiter='\n')
		print data_inac_list

	# PLOTTING
	fig,ax = plt.subplots()
	x0,x1 = ax.get_xlim()
	y0,y1 = ax.get_ylim()

	# to make it squared
	if x1 >= y1:
		y1 = x1
	else:
		x1 = y1

	if withPassIndex:
		x0 = y0 = 0.
		x1 = y1 = 1.

	ax.set_xlim( (x0,x1) )
	ax.set_ylim( (y0,y1) )
	if len(xlim)>1:
		ax.set_xlim( xlim )
		x0,x1 = ax.get_xlim()
	if len(ylim)>1:
		ax.set_ylim( ylim )
		y0,y1 = ax.get_ylim()

	ax.set_aspect( abs(x1-x0)/abs(y1-y0) )
	# add diagonal
	ax.plot( [x0,x1], [y0,y1], linestyle='--', color="k" )
	if data_full and data_inac:
		ax.scatter( data_full_list, data_inac_list, marker=data_marker, s=60, facecolor="k", edgecolor="k", label=sheet )

	ax.scatter( x_full, x_inac, marker="o", s=80, facecolor="blue", edgecolor="black", label=sheet )

	if withRegression:
		if data_full and data_inac:
			m,b = numpy.polyfit(data_full_list,data_inac_list, 1)
			print "data fitted line:", m, "x +", b
			x = numpy.arange(x0, x1)
			ax.plot(x, m*x+b, linestyle='-', color="k")

		m,b = numpy.polyfit(x_full,x_inac, 1)
		print "fitted line:", m, "x +", b
		x = numpy.arange(x0, x1)
		ax.plot(x, m*x+b, linestyle='-', color="blue")

	if withCorrCoef:
		# add correlation coefficient
		corr = numpy.corrcoef(x_full,x_inac)
		sheet = str(sheet) + " r=" + '{:.3f}'.format(corr[0][1])

	if withCentroid:
		cx = x_full.sum()/len(x_full)
		cy = x_inac.sum()/len(x_inac)
		print "cx", cx, "cy", cy
		ax.plot(cx, cy, 'b+', markersize=12, markeredgewidth=3)
		if data_full and data_inac:
			cx = data_full_list.sum()/len(data_full_list)
			cy = data_inac_list.sum()/len(data_inac_list)
			print "dcx", cx, "dcy", cy
			ax.plot(cx, cy, 'k+', markersize=12, markeredgewidth=3)

	# from scipy.stats import chi2_contingency
	# obs = [ x_inac, x_full ]
	# chi2, p, dof, ex = chi2_contingency( obs, correction=False )
	# print "Scatter chi2:", chi2, "p:", p

	# text
	ax.set_title( sheet )
	ax.set_xlabel( xlabel )
	ax.set_ylabel( ylabel )
	ax.legend( loc="lower right", shadow=False, scatterpoints=1 )
	# plt.show()
	plt.savefig( folder_inactive+"/TrialAveragedPairwiseScatter_"+parameter+"_"+str(sheet)+".png", dpi=200, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def pairwise_response_reduction( sheet, folder_full, folder_inactive, stimulus, parameter, start, end, percentage=False, xlabel="", ylabel="" ):
	print "folder_full: ",folder_full
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print "folder_inactive: ",folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	x_full_rates, stimuli_full = get_per_neuron_spike_count( data_store_full, stimulus, sheet, start, end, parameter, neurons=[], spikecount=False ) 
	x_inac_rates, stimuli_inac = get_per_neuron_spike_count( data_store_inac, stimulus, sheet, start, end, parameter, neurons=[], spikecount=False ) 
	# print x_full_rates.shape
	# print x_inac_rates.shape

	print "1. Test of normality for full (mean p-value must be >>0):", numpy.mean(scipy.stats.normaltest(x_full_rates)[1])
	print "1. Test of normality for inac (mean p-value must be >>0):", numpy.mean(scipy.stats.normaltest(x_inac_rates)[1])

	x_full_std = x_full_rates.std(axis=1)
	x_inac_std = x_inac_rates.std(axis=1)
	print "2. Difference between full and inac std (test whether ANOVA will be meaningful, all values have to be ~0):", x_full_std - x_inac_std

	increased = 0.
	nochange = 0.
	decreased = 0.
	diff = x_full_rates - x_inac_rates
	for i in range(0,diff.shape[1]):
		# ANOVA significance test for open vs closed loop conditions
		# Null-hypothesis: "the rates in the closed- and open-loop conditions are equal"
		# full_inac_diff = scipy.stats.f_oneway(x_full_rates[:,i], x_inac_rates[:,i])
		full_inac_diff = scipy.stats.f_oneway(numpy.sqrt(x_full_rates[:,i]), numpy.sqrt(x_inac_rates[:,i]))
		print full_inac_diff
		# Custom test for open vs closed loop conditions
		# Low threshold, on the majority of tested frequencies
		if   numpy.sum( diff[:,i] > 0.5 ) > 4: increased += 1
		elif numpy.sum( diff[:,i] < -0.5 ) > 4: decreased += 1
		else: nochange += 1
	increased /= diff.shape[1] 
	nochange /= diff.shape[1]
	decreased /= diff.shape[1]
	increased *= 100.
	nochange *= 100.
	decreased *= 100.
	print "increased", increased, "%"
	print "nochange", nochange, "%"
	print "decreased", decreased, "%"

	x_full = x_full_rates.mean(axis=1)
	x_inac = x_inac_rates.mean(axis=1)
	print x_full.shape

	x_full_max = x_full.max()
	x_inac_max = x_full.max()
	x_full = x_full/x_full_max *100
	x_inac = x_inac/x_inac_max *100

	reduction = numpy.zeros( len(stimuli_full) )
	# compute the percentage difference at each group
	for i,(full,inac) in enumerate(zip(x_full, x_inac)):
		print i, full, inac, full-inac
		reduction[i] = full-inac
	print reduction

	# PLOTTING
	# width = 0.35 # spatial
	width = 0.9
	ind = numpy.arange(len(reduction))
	fig, ax = plt.subplots()

	barlist = ax.bar(numpy.arange(8),[increased,44,0,nochange,36,0,decreased,20], width=0.99) # empty column to separate groups
	barlist[0].set_color('blue')
	barlist[1].set_color('black')
	barlist[3].set_color('blue')
	barlist[4].set_color('black')
	barlist[6].set_color('blue')
	barlist[7].set_color('black')
	ax.set_xticklabels(['','increased','','','no change','','','decreased'])
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)
	ax.set_yticklabels(['0','','10','','20','','30','','40',''])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_ylabel("Number of cells", fontsize=18)

	ax2 = fig.add_axes([.7, .7, .2, .2])
	ax2.bar( ind, reduction, width=width, facecolor='blue', edgecolor='blue')
	ax2.set_xlabel(xlabel)
	ax2.set_ylabel(ylabel)
	# ax2.spines['right'].set_visible(False)
	# ax2.spines['top'].set_visible(False)
	# ax2.spines['bottom'].set_visible(False)
	ax2.set_xticklabels( ['.05', '.2', '1.2', '3.', '6.4', '8', '12', '30'] )
	ax2.axis([0, 8, -1, 1])
	for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
		label.set_fontsize(9)

	plt.savefig( folder_inactive+"/response_reduction_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_conductance_tuning_curve( sheet, folder, stimulus, parameter, ticks, percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.] ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )
	print "analog_ids (pre): ",analog_ids

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name=sheet)[0]
		l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < 0.5)[0]]
		analog_ids = l4_exc_or_many
		print "# of V1 cells having orientation close to 0:", len(analog_ids)

		if parameter=='radius':
			position_V1 = data_store.get_neuron_postions()[sheet]
			print position_V1
			V1_sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=analog_ids)
			print V1_sheet_ids
			# radius_V1_ids = select_ids_by_position(position_V1, V1_sheet_ids, radius=[0,3])
			radius_V1_ids = select_ids_by_position(position_V1, V1_sheet_ids, box=[[-.5,-.5],[.5,.5]])
			radius_V1_ids = data_store.get_sheet_ids(sheet_name=sheet,indexes=radius_V1_ids)
			print "# of V1 cells within radius range having orientation close to 0:", len(radius_V1_ids)
			analog_ids = radius_V1_ids

	print "analog_ids (post): ",analog_ids
	num_ticks = len( ticks )
	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		# key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).radius 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	trials = len(segs) / num_ticks
	print "trials:",trials

	pop_gsyn_e = []
	pop_gsyn_i = []
	for n,idd in enumerate(analog_ids):
		print "idd", idd
		full_gsyn_es = [s.get_esyn(idd) for s in segs]
		full_gsyn_is = [s.get_isyn(idd) for s in segs]
		# print "len full_gsyn_e/i", len(full_gsyn_es) # 61 = 1 spontaneous + 6 trial * 10 num_ticks
		# print "shape gsyn_e/i", full_gsyn_es[0].shape
		# mean input over trials
		mean_full_gsyn_e = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0])) # init
		mean_full_gsyn_i = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0]))
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
		for s in range(num_ticks):
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] / trials
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] / trials

		pop_gsyn_e.append(mean_full_gsyn_e)
		pop_gsyn_i.append(mean_full_gsyn_i)


	pop_e = numpy.array(pop_gsyn_e)
	pop_i = numpy.array(pop_gsyn_i)

	# mean and std over cells
	mean_pop_e = numpy.mean(pop_e, axis=(2,0) )
	mean_pop_i = numpy.mean(pop_i, axis=(2,0) ) 
	std_pop_e = numpy.std(pop_e, axis=(2,0), ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	std_pop_i = numpy.std(pop_i, axis=(2,0), ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points

	final_sorted_e = [ numpy.array(list(e)) for e in zip( *sorted( zip(ticks, mean_pop_e, std_pop_e) ) ) ]
	final_sorted_i = [ numpy.array(list(e)) for e in zip( *sorted( zip(ticks, mean_pop_i, std_pop_i) ) ) ]

	if percentile:
		firing_max = numpy.amax( final_sorted_e[1] )
		final_sorted_e[1] = final_sorted_e[1] / firing_max * 100
		firing_max = numpy.amax( final_sorted_i[1] )
		final_sorted_i[1] = final_sorted_i[1] / firing_max * 100

	# Plotting tuning curve
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()

	err_max = final_sorted_i[1] + final_sorted_i[2]
	max_i = numpy.amax(err_max)
	err_min = final_sorted_i[1] - final_sorted_i[2]
	ax.fill_between(final_sorted_i[0], err_max, err_min, color='blue', alpha=0.3)
	ax.plot( final_sorted_i[0], final_sorted_i[1], color='blue', linewidth=2 )

	err_max = final_sorted_e[1] + final_sorted_e[2]
	max_e = numpy.amax(err_max)
	err_min = final_sorted_e[1] - final_sorted_e[2]
	ax.fill_between(final_sorted_e[0], err_max, err_min, color='red', alpha=0.3)
	ax.plot( final_sorted_e[0], final_sorted_e[1], color='red', linewidth=2 )

	if not percentile:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim([0, max_i if max_i > max_e else max_e])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.xaxis.set_ticks(ticks, ticks)
	ax.yaxis.set_ticks_position('left')

	if useXlog:
		ax.set_xscale("log", nonposx='clip')
	if useYlog:
		ax.set_yscale("log", nonposy='clip')

	# text
	ax.set_xlabel( parameter )
	ax.set_ylabel( "Conductance change (%)" )
	plt.tight_layout()
	plt.savefig( folder+"/TrialAveragedConductances_"+sheet+"_"+parameter+"_pop.png", dpi=200, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def data_significance(closed_file, open_file, testtype, removefirst=False):
	closed_data = numpy.genfromtxt(closed_file, delimiter='\n')
	open_data = numpy.genfromtxt(open_file, delimiter='\n')
	if removefirst:
		closed_data = closed_data[1:]
		open_data = open_data[1:]
	# print closed_data
	# print open_data

	print "\n----------- Testing pre-requisite for significance tests:"
	closed_alpha, closed_normality = scipy.stats.normaltest(closed_data)
	print "1a. Test of normality for closed", closed_alpha," (mean p-value must be >>0):", closed_normality
	open_normality = scipy.stats.normaltest(open_data)[1]
	print "1b. Test of normality for open (mean p-value must be >>0):", open_normality
	closed_skewness = scipy.stats.skew(closed_data)
	closed_skewtest = scipy.stats.skewtest(closed_data)[1]
	print "2a. Test of skewness for closed (must be ~0):", closed_skewness, "with skewtest p-value=", closed_skewtest
	open_skewness = scipy.stats.skew(open_data)
	open_skewtest = scipy.stats.skewtest(open_data)[1]
	print "2b. Test of skewness for open (must be ~0):", open_skewness, "with skewtest p-value=", open_skewtest
	var_diff = abs(closed_data.var() - open_data.var())
	print "3. Difference between closed and open variance (must be ~0):", var_diff


	if testtype == 't-test':
		print "\n----------- t-test"

		if (closed_normality<0.1 and open_normality<0.1) or (closed_skewtest<0.05 and open_skewtest<0.05 and len(closed_data)>50 and len(open_data)>50): # permissive
			equal_var = var_diff<0.1 # permissive limit
			if not equal_var:
				print "Welch's t-test is performed instead of Student's due to inequality of variances."
			st, p = scipy.stats.ttest_ind( closed_data, open_data, equal_var=equal_var ) # if equal_var==False Welch is performed
			print "z-score:", st, "p-value:", p
		else:
			print "Test of normality has not been passed (and skewness is not compensated by the number of samples), therefore t-test cannot be applied. Chi-squared will be performed instead."
			chi2, p = scipy.stats.chisquare( open_data, closed_data )
			print "z-score:", chi2, "p-value:", p


	if testtype == 'anova':
		print "\n----------- ANOVA"
		# homoscedasticity
		if ((closed_normality>0.1 and open_normality>0.1) or (closed_skewtest>0.05 and open_skewtest>0.05 and len(closed_data)>50 and len(open_data)>50)) and var_diff<0.1: # permissive
			st, p = scipy.stats.f_oneway(closed_data, open_data)
			print "plain z-score:", st, "p-value:", p
			st, p = scipy.stats.f_oneway(numpy.sqrt(closed_data), numpy.sqrt(open_data))
			print "squared z-score:", st, "p-value:", p
		else:
			print "Test of normality has not been passed (and skewness is not compensated by the number of samples), therefore ANOVA cannot be applied. Kruskal-Wallis will be performed instead."
			st, p = scipy.stats.kruskal(closed_data, open_data)
			print "z-score:", st, "p-value:", p




def normalize(a, axis=-1, order=2):
	l2 = numpy.atleast_1d( numpy.linalg.norm(a, order, axis) )
	l2[l2==0] = 1
	return a/ numpy.expand_dims(l2, axis)




def comparison_tuning_map(directory, xvalues, yvalues, ticks):

	filenames = [ x for x in glob.glob(directory+"/*.csv") ]
	print filenames

	colors = numpy.zeros( (len(xvalues),len(yvalues)) )
	alpha = numpy.zeros( (len(xvalues),len(yvalues)) )
	for name in filenames:
		print name
		mapname = os.path.splitext(name)[0]+'.png'
		print mapname

		# cycle over lines
		with open(name,'r') as csv:
			for i,line in enumerate(csv): 
				print line
				print eval(line)
				xvalue = eval(line)[0]
				yvalue = eval(line)[1]
				s = eval(line)[2]
				print xvalue, yvalue, s
				fit = numpy.polyfit([0,1,2], s, 1)
				if numpy.amin(s) < -10.: # tolerance on the smallest value
					fit = [0., 0.]
				if fit[0] < 0.:
					fit = [0., 0.]
				print s, fit
				colors[xvalues.index(xvalue)][yvalues.index(yvalue)] = fit[0] # if fit[0]>0. else 0. # slope
				alpha[xvalues.index(xvalue)][yvalues.index(yvalue)] = fit[1] # in

		print colors
		# alpha = numpy.absolute( normalize(alpha) )
		# alpha = normalize(alpha)
		print alpha

		plt.figure()
		ca = plt.imshow(colors, interpolation='nearest', cmap='coolwarm')
		# ca = plt.contourf(colors, cmap='coolwarm')
		cbara = plt.colorbar(ca, ticks=[numpy.amin(colors), 0, numpy.amax(colors)])
		cbara.set_label('Regression Slope')
		# cb = plt.contour(alpha, cmap='brg')
		# cbarb = plt.colorbar(cb, ticks=[numpy.amin(alpha), 0, numpy.amax(alpha)])
		# print cbarb.set_ticklabels([numpy.amin(alpha), 0, numpy.amax(alpha)])
		# cbarb.set_label('Regression Intercept')
		plt.xticks(ticks, xvalues)
		plt.yticks(ticks, yvalues)
		plt.xlabel('V1-PGN arborization radius')
		plt.ylabel('PGN-LGN arborization radius')
		plt.savefig( mapname, dpi=300, transparent=True )
		plt.close()
		# plt.show()




###################################################
# Execution

full_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_luminance_open_____",

	# "Deliverable/ThalamoCorticalModel_data_contrast_closed_____",
	"Deliverable/ThalamoCorticalModel_data_contrast_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_LGNonly_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "Deliverable/ThalamoCorticalModel_data_size_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_size_open_____",
	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_nonoverlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____old",
	# "Deliverable/ThalamoCorticalModel_data_size_LGNonly_____",

	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_open_____",

	# "ThalamoCorticalModel_data_xcorr_open_____1", # just one trial
	# "ThalamoCorticalModel_data_xcorr_open_____2deg", # 2 trials
	# "ThalamoCorticalModel_data_xcorr_closed_____2deg", # 2 trials

	# "Deliverable/CombinationParamSearch_LGN_PGN",
	# "CombinationParamSearch_large_closed",
	# "CombinationParamSearch_more_focused_closed_nonoverlapping",

	# "ThalamoCorticalModel_data_size_closed_nonoverlapping_____",
	# "ThalamoCorticalModel_data_size_closed_overlapping_____",

	# "CombinationParamSearch_intact_nonoverlapping",
	]

inac_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "Deliverable/ThalamoCorticalModel_data_size_open_____",

	# "CombinationParamSearch_large_nonoverlapping",
	# "CombinationParamSearch_more_focused_nonoverlapping",

	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_nonoverlapping_____",
	# "ThalamoCorticalModel_data_size_nonoverlapping_____",
	# "ThalamoCorticalModel_data_size_overlapping_____",

	# "CombinationParamSearch_altered_nonoverlapping",
	]



# sheets = ['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']
# sheets = ['X_ON', 'X_OFF', 'PGN']
# sheets = [ ['X_ON', 'X_OFF'], 'PGN']
# sheets = [ ['X_ON', 'X_OFF'] ]
# sheets = [ 'X_ON', 'X_OFF', ['X_ON', 'X_OFF'] ]
# sheets = ['X_ON', 'X_OFF', 'V1_Exc_L4']
sheets = ['X_ON', 'X_OFF']
# sheets = ['X_ON']
# sheets = ['X_OFF'] 
# sheets = ['PGN']
# sheets = ['V1_Exc_L4'] 


# ONLY for comparison parameter search
if False: 
	# How the choice of smaller, equal, and larger is made:
	# - Smaller: [ smaller0 : smaller1 ]
	# - Equal:   [  equal0  :  equal1  ]
	# - Larger:  [  larger0 : larger1  ]

	#            0     1     2     3     4     5     6     7     8     9
	sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46]

	#                                         |  smaller    |       equal        |                    |   larger
	#          0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19
	# sizes = [0.125, 0.152, 0.186, 0.226, 0.276, 0.337, 0.412, 0.502, 0.613, 0.748, 0.912, 1.113, 1.358, 1.657, 2.021, 2.466, 3.008, 3.670, 4.477, 5.462]
	# Ssmaller = 5
	# Sequal   = 7
	# SequalStop  = 10
	# Slarger  = 13

	# # CHOICE OF STIMULI GROUPS
	# Ismaller = [0,3]
	# Iequal   = [4,6]
	# Ilarger  = [6,8] # NON
	# Ilarger  = [7,10] # OVER

	box = [[-.5,-.5],[.5,.5]] # close to the overlapping
	# box = [[-.5,.0],[.5,.5]] # close to the overlapping
	# box = [[-.5,.0],[.5,.1]] # far from the overlapping

	if len(inac_list):
		csvfile = open(inac_list[0]+"/barsizevalues_"+str(sheets[0])+"_box"+str(box)+".csv", 'w')
	else:
		csvfile = open(full_list[0]+"/endinhibitionindex_"+str(sheets[0])+".csv", 'w')

	for i,l in enumerate(full_list):
		# for parameter search
		full = [ l+"/"+f for f in sorted(os.listdir(l)) if os.path.isdir(os.path.join(l, f)) ]
		if len(inac_list):
			large = [ inac_list[i]+"/"+f for f in sorted(os.listdir(inac_list[i])) if os.path.isdir(os.path.join(inac_list[i], f)) ]

		for i,f in enumerate(full):
			print i,f

			color = "black"
			if "open" in f:
				color = "cyan"
			if "closed" in f:
				color = "blue"
			if "Kimura" in f:
				color = "gold"
			if "LGNonly" in f:
				color = "yellow"

			for s in sheets:

				if "open" in f and 'PGN' in s:
					color = "lime"
				if "closed" in f and 'PGN' in s:
					color = "darkgreen"

				print color

				if len(inac_list):
					size_tuning_comparison( 
						sheet=s, 
						folder_full=f, 
						folder_inactive=large[i],
						stimulus="DriftingSinusoidalGratingDisk",
						parameter='radius',
						reference_position=[[0.0], [0.0], [0.0]],
						sizes = sizes,
						box = box,
						csvfile = csvfile,
						plotAll = False # plot all barplots per folder?
					)

				else:
					end_inhibition_barplot( 
						# sheet=['X_ON', 'X_OFF'], 
						# sheet=['X_ON'], 
						sheet=s, 
						folder=f, 
						stimulus="DriftingSinusoidalGratingDisk",
						parameter='radius',
						start=100., 
						end=1000., 
						xlabel="Index of end-inhibition",
						ylabel="Number of cells",
						closed=False,
						# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_open.csv",
						# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_7D.csv",
						# closed=True,
						# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_closed.csv",
						csvfile = csvfile,
					)
					trial_averaged_tuning_curve_errorbar( 
						# sheet=['X_ON', 'X_OFF'], 
						sheet=s, 
						folder=f, 
						stimulus='DriftingSinusoidalGratingDisk',
						parameter="radius",
						start=100., 
						end=2000., 
						xlabel="radius", 
						ylabel="firing rate (sp/s)", 
						color=color, 
						useXlog=False, 
						useYlog=False, 
						percentile=False, #True,
						ylim=[0,50],
						box=False,
						# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
						data_curve=False,
					)

				csvfile.write("\n")

	# plot map
	csvfile.close()

else:

	for i,f in enumerate(full_list):
		print i,f

		color = "black"
		if "open" in f:
			color = "cyan"
		if "closed" in f:
			color = "blue"
		if "Kimura" in f:
			color = "gold"
		if "LGNonly" in f:
			color = "yellow"

		for s in sheets:

			if "open" in f and 'PGN' in s:
				color = "lime"
			if "closed" in f and 'PGN' in s:
				color = "darkgreen"

			print color

			# LUMINANCE
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='Null',
			# 	parameter="background_luminance",
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="contrast", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color="black", 
			# 	ylim=[0.,10.],
			# 	percentile=False,
			# 	useXlog=True 
			# )

			# CONTRAST
			# trial_averaged_tuning_curve_errorbar( 
			# 	# sheet=['X_ON', 'X_OFF'], 
			# 	# sheet=['X_ON'], 
			# 	# sheet=['X_OFF'], 
			# 	# sheet=['PGN'], 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="contrast",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Contrast", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color,
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=True, #False 
			# 	ylim=[0,120], #[0,35]
			# )
			cumulative_distribution_C50_curve( 
				# sheet=['X_ON', 'X_OFF'], 
				sheet=s, 
				folder=f, 
				stimulus="FullfieldDriftingSinusoidalGrating",
				parameter='contrast',
				start=100., 
				end=10000., 
				xlabel="C$_{50}$",
				ylabel="Percentile",
				color=color,
				# data="/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_closed.csv",
				# data_color="black",
				data="/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_open.csv",
				data_color="grey",
			)

			# # TEMPORAL
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="temporal_frequency",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Temporal frequency", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color="black", 
			# 	useXlog=True, 
			# 	useYlog=True, 
			# 	percentile=False,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/DerringtonLennie1982_6A_2d.csv",
			# )

			# # SPATIAL
			# trial_averaged_tuning_curve_errorbar( 
			# 	# sheet=['X_ON', 'X_OFF'], 
			# 	# sheet=['X_ON'], 
			# 	# sheet=['X_OFF'], 
			# 	# sheet=['PGN'], 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="Spatial frequency", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, #"gold", #"darkgreen", "lime"
			# 	useXlog=True, 
			# 	useYlog=False, 
			# 	percentile=False, #False 
			# )

			# SIZE
			# Ex: ThalamoCorticalModel_data_size_V1_full_____
			# Ex: ThalamoCorticalModel_data_size_open_____
			# end_inhibition_barplot( 
			# 	# sheet=['X_ON', 'X_OFF'], 
			# 	# sheet=['X_ON'], 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	xlabel="Index of end-inhibition",
			# 	ylabel="Number of cells",
			# 	closed=False,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_open.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_7D.csv",
			# 	# closed=True,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_closed.csv",
			# )
			# trial_averaged_tuning_curve_errorbar( 
			# 	# sheet=['X_ON', 'X_OFF'], 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="radius", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False, #True,
			# 	ylim=[0,50],
			# 	box=False,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
			# 	data_curve=False,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	#       0      1     2     3     4     5     6     7     8     9
			# 	ticks=[0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
			# 	percentile=True,
			# 	ylim=[0,120]
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius'
			# )

			# # #ORIENTATION
			# # Ex: ThalamoCorticalModel_data_orientation_V1_full_____
			# # Ex: ThalamoCorticalModel_data_orientation_open_____
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	# color="black", 
			# 	# color="red", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	ylim=[0,50]
			# )
			# orientation_bias_barplot( 
			# 	sheet=['X_ON', 'X_OFF'], 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='orientation',
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="Orientation bias",
			# 	ylabel="Number of cells",
			# 	closed=False,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_open.csv",
			# 	# closed=True,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_closed.csv",
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter='orientation',
			# 	ticks=[0.0, 0.314, 0.628, 0.942, 1.256, 1.570, 1.884, 2.199, 2.513, 2.827],
			# 	percentile=True,
			# 	ylim=[0,110]
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating', # 'InternalStimulus',
			# 	stimulus_parameter='orientation' # 'duration',
			# )

			# #CROSS-CORRELATION
			# trial_averaged_corrected_xcorrelation( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	# stimulus='FlashingSquares',
			# 	stimulus='FullfieldDriftingSquareGrating',
			# 	start=100., 
			# 	end=1000., 
			# 	bin=20.0,
			# 	xlabel="time (ms)", 
			# 	ylabel="counts per bin"
			# )

			# PAIRWISE
			for j,l in enumerate(inac_list):
				print j

				# # ONGOING ACTIVITY
				# #Ex: ThalamoCorticalModel_data_luminance_closed_____ vs ThalamoCorticalModel_data_luminance_open_____
				# pairwise_scatterplot( 
				# 	# sheet=s,
				# 	sheet=['X_ON', 'X_OFF'], # s,
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="Null",
				# 	stimulus_band=6, # 1 cd/m2 as in WaleszczykBekiszWrobel2005
				# 	parameter='background_luminance',
				# 	start=100., 
				# 	end=2000., 
				# 	xlabel="closed-loop ongoing activity (spikes/s)",
				# 	ylabel="open-loop ongoing activity (spikes/s)",
				# 	withRegression=True,
				# 	withCorrCoef=True,
				# 	withCentroid=True,
				# 	xlim=[0,40],
				# 	ylim=[0,40],
				# 	data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_closed.csv",
				# 	data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_open.csv",
				#	data_marker = "D",
				# )

				# # SPATIAL FREQUENCY
				# # Ex: ThalamoCorticalModel_data_spatial_V1_full_____ vs ThalamoCorticalModel_data_spatial_Kimura_____
				# pairwise_scatterplot( 
				# 	# sheet=['X_ON', 'X_OFF'], 
				# 	# sheet=['X_ON'], 
				# 	# sheet=['X_OFF'], 
				# 	# sheet=['PGN'], 
				# 	sheet=s, 
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="FullfieldDriftingSinusoidalGrating",
				# 	parameter='spatial_frequency',
				# 	start=100., 
				# 	end=10000., 
				# 	xlabel="Control",
				# 	ylabel="Cortex Inactivated",
				# 	withRegression=False,
				# 	withCorrCoef=False,
				# 	withCentroid=True,
				# 	data_marker = "o",
				# 	# LOW-PASS
				# 	stimulus_band=1,
				# 	withPassIndex=True,
				# 	reference_band=3,
				# 	# data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_closed.csv",
				# 	# data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_open.csv",
				# 	# # PEAK
				# 	# stimulus_band=None,
				# 	# reference_band=None,
				# 	# withPeakIndex=True,
				# 	# data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2B_closed.csv",
				# 	# data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2B_open.csv",
				# 	# # HIGH-CUTOFF
				# 	# stimulus_band=6,
				# 	# reference_band=None,
				# 	# withHighIndex=True,
				# 	# data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2C_closed.csv",
				# 	# data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2C_open.csv",
				# )

				# # TEMPORAL
				# pairwise_response_reduction( 
				# 	sheet=['X_ON', 'X_OFF'], 
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="FullfieldDriftingSinusoidalGrating",
				# 	parameter='temporal_frequency',
				# 	start=100., 
				# 	end=10000., 
				# 	percentage=True,
				# 	xlabel="", #Temporal frequency", 
				# 	ylabel="", #Response change (%)", 
				# )

				# # COMPARISON SIZE TUNING
				#            0     1     2     3     4     5     6     7     8     9
				# sizes = [0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46]
				# box = [[-.6, -.6],[.6,.6]] # all
				# # box = [[-.6, .0],[.6,.6]] # non-over
				# size_tuning_comparison( 
				# 	sheet=s, 
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="DriftingSinusoidalGratingDisk",
				# 	parameter='radius',
				# 	reference_position=[[0.0], [0.0], [0.0]],
				# 	sizes = sizes,
				# 	box = box,
				# 	plotAll = False # plot barplot and all tuning curves?
				# 	# plotAll = True # plot barplot and all tuning curves?
				# )




# STATISTICAL SIGNIFICANCE

# # ONGOING ACTIVITY (WaleszczykBekiszWrobel2005)
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_open.csv",
# 	'anova'
# )
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/ongoing_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/ongoing_open.csv",
# 	'anova'
# )

# # CONTRAST (LiYeSongYangZhou2011)
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_open.csv",
# 	't-test'
# )
# data_significance( # synthetic
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/contrast_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/contrast_open.csv",
# 	't-test'
# )
# data_significance( # synthetic cells
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/contrast_closed_cells.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/contrast_open_cells.csv",
# 	't-test'
# )


# # SPATIAL (KimuraShimegiHaraOkamotoSato2013)
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_open.csv",
# 	'anova'
# )
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/spatial_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/spatial_open.csv",
# 	'anova'
# )


# # SIZE (MurphySillito1987)
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_open.csv",
# 	't-test',
#	removefirst=True
# )
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/size_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/size_open.csv",
# 	't-test'
# )
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/size_closed_peaks.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/size_open_peaks.csv",
# 	't-test'
# )


# # ORIENTATION ()
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_open.csv",
# 	't-test',
# 	removefirst=True
# )
# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/orientation_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/orientation_open.csv",
# 	't-test'
# )




closed_files = [ 
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.82743338823.csv",
    ]

closed_ON_files = [ 
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.125.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.163713961769.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.214418090225.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.280825880206.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.367800939327.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.481713191357.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.630905399949.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.826304180217.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.08222024776.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.41739651414.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.85638078982.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_2.43132362923.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_3.18433298947.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_4.17055815439.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_ON_orientation_2.82743338823.csv",
	]

closed_OFF_files = [ 
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.125.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.163713961769.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.214418090225.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.280825880206.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.367800939327.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.481713191357.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.630905399949.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.826304180217.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.08222024776.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.41739651414.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.85638078982.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_2.43132362923.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_3.18433298947.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_4.17055815439.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_X_OFF_orientation_2.82743338823.csv",
	]


open_files = [ 
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.125.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.163713961769.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.214418090225.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.280825880206.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.367800939327.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.481713191357.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.630905399949.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.826304180217.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08222024776.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41739651414.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.85638078982.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43132362923.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18433298947.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17055815439.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.82743338823.csv",
    ]

open_ON_files = [ 
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.125.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.163713961769.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.214418090225.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.280825880206.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.367800939327.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.481713191357.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.630905399949.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.826304180217.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.08222024776.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.41739651414.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.85638078982.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_2.43132362923.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_3.18433298947.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_4.17055815439.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_ON_orientation_2.82743338823.csv",
	]

open_OFF_files = [ 
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.125.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.163713961769.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.214418090225.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.280825880206.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.367800939327.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.481713191357.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.630905399949.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.826304180217.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.08222024776.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.41739651414.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.85638078982.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_2.43132362923.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_3.18433298947.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_4.17055815439.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_0.0.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_0.314159265359.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_0.628318530718.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_0.942477796077.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_1.25663706144.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_1.57079632679.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_1.88495559215.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_2.19911485751.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_2.51327412287.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_X_OFF_orientation_2.82743338823.csv",
	]

# fano_comparison_timecourse( closed_files, open_files, sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____", [-3., 15.] ) # cortex
# fano_comparison_timecourse( closed_ON_files, open_ON_files, sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____", [-2., 5.], closed_OFF_files, open_OFF_files ) # LGN

# fano_comparison_timecourse( closed_files, open_files, sheets, "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____", [-0.5, 2.] )
# fano_comparison_timecourse( closed_ON_files, open_ON_files, sheets, "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____", [-0.5, 2.], closed_OFF_files, open_OFF_files ) # LGN

###############################
# directory = "CombinationParamSearch_more_focused_nonoverlapping"
# xvalues = [70, 80, 90, 100, 110]
# yvalues = [130, 140, 150, 160, 170]
# ticks = [0,1,2,3,4]


# directory = "CombinationParamSearch_large_nonoverlapping"
# xvalues = [30, 50, 70, 90]
# yvalues = [150, 200, 250, 300]
# ticks = [0,1,2,3]

# comparison_tuning_map(directory, xvalues, yvalues, ticks)
