# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import os
import ast
import re
import glob
import json

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
from mozaik.analysis.TrialAveragedFiringRateCutout import TrialAveragedFiringRateCutout
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore

from mozaik.tools.mozaik_parametrized import colapse

import neo


# #from: https://stackoverflow.com/questions/35990467/fit-two-gaussians-to-a-histogram-from-one-set-of-data-python
def gauss(x,mu,sigma,A):
	return A * numpy.exp( -(x-mu)**2 / (2*sigma**2) )
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
	return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def bimodal_fit(x, y, sd):
	from scipy.optimize import curve_fit
	x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
	params,_ = curve_fit( gauss, x, y, 
		bounds=((20.,  0.,     0.),  (60., 100., 100.)), 
		p0=(30.,  5.,  20.),
		maxfev=10000000,
	) 
	# params,_ = curve_fit( bimodal, x, y, 
			# method='lm', 
			# method='trf', 
			# method='dogbox', 
		# 	#        mu1, sigma1, A1,  mu2, sigma2, A2 	
		# 	bounds=((0.,  0.,     0.,  20.,  0.,     0.),  (10., 100., 500., 60., 100., 100.)), 
		# 	# p0=(3.,  0.3,  50.,  30.,  5.,  1.),
		# 	sigma=sd,
		# 	maxfev=10000000,
		# ) 
	return params



def select_ids_by_position(positions, sheet_ids, radius=[0,0], box=[], reverse=False):
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
				# print a[0], "   ", a[1]
				selected_ids.append(i[0])
				distances.append(0.0)
		else:
			# print a # from origin
			l = numpy.linalg.norm([[0.],[0.],[0.]]-a)

			# print "distance",l
			if abs(l)>min_radius and abs(l)<max_radius:
				# print "taken"
				selected_ids.append(i[0])
				distances.append(l)

	# sort by distance
	# print len(selected_ids)
	# print distances
	return [x for (y,x) in sorted(zip(distances,selected_ids), key=lambda pair:pair[0], reverse=reverse)]




def get_per_neuron_window_spike_count( datastore, sheet, stimulus, stimulus_parameter, start, stop, window=50.0, neurons=[], nulltime=False ):

	if not len(neurons)>1:
		spike_ids = param_filter_query(datastore, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = datastore.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		neurons = datastore.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)
	print "neurons", len(neurons)

	SpikeCount( 
		param_filter_query( datastore, sheet_name=sheet, st_name=stimulus ), 
		ParameterSet({'bin_length':window, 'neurons':list(neurons), 'null':nulltime}) 
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
		print sh
		dsv = param_filter_query( datastore, identifier='PerNeuronValue', sheet_name=sh, st_name=stimulus )
		# dsv.print_content(full_recordings=False)
		pnvs = [ dsv.get_analysis_result() ]
		# print pnvs
		# get stimuli from PerNeuronValues
		st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs[-1]]

		if len(neurons)<1:
			neurons = param_filter_query(datastore, sheet_name=sh, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
			sheet_ids = datastore.get_sheet_indexes(sheet_name=sh, neuron_ids=neurons)
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




def select_by_box(data_store, sheet, ids, box):
	positions = data_store.get_neuron_postions()[sheet]
	sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=ids)
	box_ids = select_ids_by_position(positions, sheet_ids, box=box)
	return data_store.get_sheet_ids(sheet_name=sheet,indexes=box_ids)




def select_by_orientation(data_store, sheet, ids, preferred=True):
	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	cell_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = sheet)[0]
	if preferred:
		sel_cell_or = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(cell_or.get_value_by_id(i),0.,numpy.pi) for i in ids]) < .1)[0]]
	else:
		sel_cell_or = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(cell_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in ids]) < .1)[0]]
	# print "Selected cells:", len(sel_cell_or)
	return list(sel_cell_or)




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
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_box"+str(box)+"only_bars.svg", dpi=300, transparent=True )
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
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_box"+str(box)+".svg", dpi=150, transparent=True )
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




def activity_ratio( sheet, folder, stimulus, stimulus_parameter, arborization_diameter=10, addon="", color="black" ):
	import ast
	matplotlib.rcParams.update({'font.size':22})
	print "folder: ",folder
	print "sheet: ",sheet
	print "addon: ", addon
	print 
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	# Spontaneous activity
	print "Spontaneous firing rate ..."
	TrialAveragedFiringRate(
		param_filter_query(data_store, sheet_name=sheet, st_direct_stimulation_name="None", st_name='InternalStimulus'),
		ParameterSet({})
	).analyse()
	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_direct_stimulation_name="None", st_name='InternalStimulus').get_segments()[0].get_stored_spike_train_ids()
	print "Total neurons:", len(spike_ids)
	spont = numpy.array( param_filter_query(data_store, st_name='InternalStimulus', sheet_name=sheet, analysis_algorithm=['TrialAveragedFiringRate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids) )
	mean_spontaneous_rate = numpy.mean(spont)
	std_spontaneous_rate = numpy.std(spont)
	print "Mean spontaneous firing rate:", mean_spontaneous_rate, " std:", std_spontaneous_rate

	# Find all cells with rate above mean spontaneous
	print 
	print "Evoked firing rate ..."
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )
	TrialAveragedFiringRate(
		param_filter_query(data_store, sheet_name=sheet, st_name=stimulus),
		ParameterSet({})
	).analyse()
	NeuronAnnotationsToPerNeuronValues(data_store, ParameterSet({})).analyse()
	above_spontaneous_ids = {}
	above_spontaneous_perc = {}
	for dsa in data_store.get_analysis_result(identifier='PerNeuronValue', value_name='Firing rate', sheet_name=sheet):
		param = ast.literal_eval( dsa.stimulus_id )[stimulus_parameter]
		above_spontaneous_ids["{:.2f}".format(param)] = numpy.array(spike_ids)[numpy.nonzero(numpy.array([dsa.get_value_by_id(i) for i in spike_ids]) > mean_spontaneous_rate)[0]]
		above_spontaneous_perc["{:.2f}".format(param)] = (float(len(above_spontaneous_ids["{:.2f}".format(param)]))/len(spike_ids) ) * 100.
		# print above_spontaneous_ids

	# Plot barplot over sizes
	x = range(len(above_spontaneous_perc))
	plt.bar(x, sorted(above_spontaneous_perc.values()), align='center', width=0.8, color=color)
	plt.xticks(x, sorted(above_spontaneous_perc.keys()), rotation='vertical')
	plt.tight_layout()
	plt.savefig( folder+"/Activity_ratio_"+str(sheet)+"_"+stimulus_parameter+"_"+addon+".png", dpi=200, transparent=True )
	plt.close()
	gc.collect()

	# Plot scatterplot positions+arborization
	positions_x = []
	positions_y = []
	for key in sorted(above_spontaneous_ids):
		print key, above_spontaneous_ids[key]
		if len(above_spontaneous_ids[key]) > 0:
			sheet_ids = numpy.array( data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=above_spontaneous_ids[key]) ).flatten()
			positions_x = numpy.take(data_store.get_neuron_postions()[sheet][0], sheet_ids)
			positions_y = numpy.take(data_store.get_neuron_postions()[sheet][1], sheet_ids)
		plt.scatter( positions_x, positions_y, c=color, marker="o", s=arborization_diameter, alpha=0.3 )
		plt.ylim([-2., 2.])
		plt.xlim([-2., 2.])
		plt.tight_layout()
		plt.savefig( folder+"/Activity_Map_"+str(sheet)+"_"+stimulus_parameter+"_"+key+"_"+addon+".png", dpi=200, transparent=True )
		plt.close()
		gc.collect()




def jpsth( sheet1, sheet2, folder, stimulus, stimulus_parameter, box1=None, box2=None, preferred1=True, preferred2=True, addon="", color="black", trials=6 ):
	import ast
	import random
	import copy
	from matplotlib.ticker import NullFormatter
	from mozaik.analysis.helper_functions import psth
	from mozaik.tools.mozaik_parametrized import MozaikParametrized

	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	spike_ids1 = param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 1:", len(spike_ids1)
	spike_ids2 = param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 2:", len(spike_ids2)

	if sheet1=='V1_Exc_L4':
		spike_ids1 = select_by_orientation(data_store, sheet1, spike_ids1, preferred=preferred1)

	if sheet2=='V1_Exc_L4':
		spike_ids2 = select_by_orientation(data_store, sheet2, spike_ids2, preferred=preferred2)

	if box1:
		spike_ids1 = select_by_box(data_store, sheet1, spike_ids1, box=box1)

	if box2:
		spike_ids2 = select_by_box(data_store, sheet2, spike_ids2, box=box2)

	print "Selected neurons 1:", len(spike_ids1)
	print spike_ids1
	print "Selected neurons 2:", len(spike_ids2)
	print spike_ids2

	# collect the spiketrains of each cell, for each trial, for each stimulus
	# st1 = {
	#   'stim' = {
	#      'id' = [] # spiktrains trials
	#   }
	# }

	# Spiktrains are collected unrespective of their trial
	# the resulting matrices are more like shift predictors than rawJPSTH (which assumes identity of trial numbers)

	# to have the real rawJPSTH and the shift predictor to be subtracted, we can add one dimension
	# or, better, 
	# - preallocate each cell's spiketrain array (content of the 2D dict) with the number of trials
	# - store the spiktrain in the right trial order
	# - retrieve them in order for the rawJPSTH
	# - from random import shuffle, and shuffle each cell's spiketrain list for the shiftJPSTH
	# - compute the correctedJPSTH

	mean_coincidence = {}
	std_coincidence = {}
	temporal_frequency = 2.

	print "Collecting spiketrains of spike_ids1 into dictionary ..."
	dsv1 = queries.param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus)
	st1 = {}
	for st, seg in zip([MozaikParametrized.idd(s) for s in dsv1.get_stimuli()], dsv1.get_segments()):
		print st, seg
		stim = ast.literal_eval(str(st))[stimulus_parameter]
		temporal_frequency = ast.literal_eval(str(st))['temporal_frequency']
		trial = ast.literal_eval(str(st))['trial']
		if not "{:.3f}".format(stim) in st1:
			st1["{:.3f}".format(stim)] = {}
			mean_coincidence["{:.3f}".format(stim)] = 0
			std_coincidence["{:.3f}".format(stim)] = 0
		for idd in spike_ids1:
			if not str(idd) in st1["{:.3f}".format(stim)]:
				# st1["{:.3f}".format(stim)][str(idd)] = []
				st1["{:.3f}".format(stim)][str(idd)] = [ [] for _ in range(trials) ]
			# st1["{:.3f}".format(stim)][str(idd)].append( seg.get_spiketrain(idd) )
			st1["{:.3f}".format(stim)][str(idd)][trial] = seg.get_spiketrain(idd)

	print "Collecting spiketrains of spike_ids2 into dictionary ..."
	dsv2 = queries.param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus)
	st2 = {}
	for st, seg in zip([MozaikParametrized.idd(s) for s in dsv2.get_stimuli()], dsv2.get_segments()):
		stim = ast.literal_eval(str(st))[stimulus_parameter]
		trial = ast.literal_eval(str(st))['trial']
		if not "{:.3f}".format(stim) in st2:
			st2["{:.3f}".format(stim)] = {}
		for idd in spike_ids2:
			if not str(idd) in st2["{:.3f}".format(stim)]:
				# st2["{:.3f}".format(stim)][str(idd)] = []
				st2["{:.3f}".format(stim)][str(idd)] = [ [] for _ in range(trials) ]
			# st2["{:.3f}".format(stim)][str(idd)].append( seg.get_spiketrain(idd) )
			st2["{:.3f}".format(stim)][str(idd)][trial] = seg.get_spiketrain(idd)

	# print "st1 == st2 ?", st1 == st2
	# st1 and st2 now contain cell-wise, stimulus-wise, trial-wise ordered spiketrains

	# plotting formats
	nullfmt = NullFormatter()
	# axes
	left, width = 0.01, 0.4
	bottom, height = 0.01, 0.4
	bottom_s = left_s = left + 0.2 + 0.04
	bottom_h = left_h = left_s + width + 0.04
	rect_jpsth = [left_s, bottom_s, left_h, bottom_h]
	rect_histx = [left_s, bottom, left_h, bottom+0.2]
	rect_histy = [left, bottom_s, left+0.2, bottom_h]
	rect_cbar = [left_s, bottom_h+0.04, left_h, bottom_h+0.04]

	# Compute JPSTH (as in mulab.physiol.upenn.edu/jpst.html)
	duration = 1029.+1
	bin_length = 5. # ms
	nbins = int(round(duration/bin_length)+1)
	shift = int(duration/temporal_frequency)

	# rawJPSTH = numpy.zeros( (nbins,nbins) )
	# shiftJPSTH = numpy.zeros( (nbins,nbins) )
	# for each stimulus
	for stim in st1:

		# Raw JPSTH: ordered trials
		rawJPSTH = numpy.zeros( (nbins,nbins) )
		print stim
		for cell1 in st1[stim]:
			# JPSTH = numpy.zeros( (nbins,nbins) )
			print "\t",cell1
			for cell2 in st2[stim]:
				print "\t\t",cell2
				for trial1 in st1[stim][cell1]:
					# print "\t\t\t", len(trial1)
					for trial2 in st2[stim][cell2]:
						# print "\t\t\t\t", len(trial2)
						for spiketime1 in trial1:
							x = round(spiketime1/bin_length)
							for spiketime2 in trial2:
								y = round(spiketime2/bin_length)
								rawJPSTH[x][y] = rawJPSTH[x][y] +1
				# plot cell-to-cell
				# print JPSTH
				# plt.figure()
				# plt.imshow( JPSTH, interpolation='nearest', cmap='coolwarm', origin='lower' )
				# plt.savefig( folder+"/JPSTH_"+str(cell1)+"_"+str(cell2)+"_"+stimulus_parameter+stim+"_"+addon+".png", dpi=300, transparent=True )
				# plt.close()
				# gc.collect()
		# print rawJPSTH

		# shiftJPSTH: shifted trials
		shiftJPSTH = numpy.zeros( (nbins,nbins) )
		print stim
		for cell1 in st1[stim]:
			# JPSTH = numpy.zeros( (nbins,nbins) )
			print "\t",cell1
			for cell2 in st2[stim]:
				print "\t\t",cell2
				for trial1 in st1[stim][cell1]:
					# print "\t\t\t", len(trial1)
					for trial2 in st2[stim][cell2]:
						# print "\t\t\t\t", len(trial2)
						for spiketime1 in trial1:
							x = round(spiketime1/bin_length)
							for spiketime2 in trial2:
								if spiketime2.size > 1: 
									# manual roll
									spiketime2 = numpy.array([t.magnitude for t in spiketime2]) + shift # add one stimulus cycle interval to each spike
									spiketime2_over = spiketime2[spiketime2>duration] - duration # put in the front those beyond the recording duration
									spiketime2 = numpy.append(spiketime2_over, numpy.delete(spiketime2[spiketime2>duration])) # rolled spiketime2
									# as before
									y = round(spiketime2/bin_length)
									shiftJPSTH[x][y] = shiftJPSTH[x][y] +1

		print "rawJPSTH == shiftJPSTH ?", numpy.array_equal(rawJPSTH,shiftJPSTH)

		# correctedJPSTH = rawJPSTH / trials
		correctedJPSTH = (rawJPSTH - shiftJPSTH).clip(min=0)
		correctedJPSTH = correctedJPSTH / ( rawJPSTH.sum(axis=0).std() * rawJPSTH.sum(axis=1).std() )

		# Plot JPSTH
		fig = plt.figure(1, figsize=(8,8))
		axJPSTH = plt.axes(rect_jpsth)
		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)
		# ax1 = plt.subplot(121)
		# axJPSTH = ax1.axes(rect_jpsth)
		# axHistx = ax1.axes(rect_histx)
		# axHisty = ax1.axes(rect_histy)
		# ax2 = plt.subplot(122)
		# axDPSTH = ax2.axes(rect_jdiag)
		# no labels
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)
		# data
		cax = axJPSTH.imshow( correctedJPSTH, interpolation='none', cmap='coolwarm', origin='lower' )
		axHistx.bar(range(nbins), correctedJPSTH.sum(axis=0), width=1.0, align='edge', facecolor=color, edgecolor=color)
		axHisty.barh(range(nbins), correctedJPSTH.sum(axis=1), align='edge', facecolor=color, edgecolor=color)
		# graphics
		axHistx.set_xlim( (0.,nbins) )
		axHistx.spines['right'].set_visible(False)
		axHistx.spines['bottom'].set_visible(False)
		axHisty.set_ylim( (0.,nbins) )
		axHisty.spines['left'].set_visible(False)
		axHisty.spines['top'].set_visible(False)
		axHisty.set_xlim( axHisty.get_xlim()[::-1] )
		axHistx.set_ylim( axHistx.get_ylim()[::-1] )
		for label in axJPSTH.get_ymajorticklabels():
			label.set_rotation(90)
		for label in axHisty.get_xmajorticklabels():
			label.set_rotation(90)
		# plt.tight_layout()
		# fig.colorbar(cax, orientation="horizontal")
		plt.savefig( folder+"/correctedJPSTH_"+str(sheet1)+"_"+addon+"_"+stimulus_parameter+stim+"_bin"+str(bin_length)+".png", dpi=300, transparent=True )
		plt.close()
		gc.collect()

		# coincidence histogram
		diag = numpy.diag( numpy.fliplr(correctedJPSTH) )
		mean_coincidence[stim] = numpy.mean(diag)
		std_coincidence[stim] = numpy.std(diag)

		# Plot Coincidence Histogram
		# print diag
		# from matplotlib.transforms import Affine2D
		# import mpl_toolkits.axisartist.floating_axes as floating_axes
		fig = plt.figure(1, figsize=(4,8))
		# tr = Affine2D().scale(2, 1).rotate_deg(45)
		# grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(0,len(diag),0,max(diag)))
		# ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
		# fig.add_subplot(ax1)
		# aux_ax = ax1.get_aux_axes(tr)
		# grid_helper.grid_finder.grid_locator1._nbins = 4
		# grid_helper.grid_finder.grid_locator2._nbins = 4
		# # spatial_corr = numpy.trace( numpy.fliplr(rawJPSTH), offset=0 )
		# aux_ax.bar(range(len(diag)), diag, width=1.0, align='edge', facecolor=color, edgecolor=color)
		plt.bar(range(len(diag)), diag, width=1.0, align='edge', facecolor=color, edgecolor=color)
		# plt.ylim( (0,) )
		plt.savefig( folder+"/correctedJPSTH_"+str(sheet1)+"_"+addon+"_"+stimulus_parameter+stim+"_bin"+str(bin_length)+"_diag.png", dpi=300, transparent=True )
		plt.close()
		gc.collect()

	mean_c = numpy.array([value for key,value in sorted(mean_coincidence.items())])
	std_c = numpy.array([value for key,value in sorted(std_coincidence.items())])
	fig = plt.figure()
	print range(len(mean_c))
	print mean_c
	print std_c
	plt.plot(range(len(mean_c)), mean_c, linewidth=3, color=color, linestyle='-' )
	err_max = mean_c + std_c
	err_min = mean_c - std_c
	plt.fill_between(range(len(mean_c)), err_max, err_min, color=color, alpha=0.3)
	# plt.ylim( (0,20) )
	plt.savefig( folder+"/correctedJPSTH_"+str(sheet1)+"_"+addon+"_"+stimulus_parameter+"_summary_coincidence_histogram.png", dpi=300, transparent=True )
	plt.close()
	gc.collect()




def spectrum( sheet, folder, stimulus, stimulus_parameter, addon="", color="black", box=False, radius=None, preferred=True, ylim=[0.,200.] ):
	import ast
	from matplotlib.ticker import NullFormatter
	from scipy import signal

	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(spike_ids)

	if sheet=='V1_Exc_L4':
		spike_ids = select_by_orientation(data_store, sheet, spike_ids, preferred=preferred)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(spike_ids)
	if len(spike_ids) < 1:
		return

	# compute PSTH per neuron, per stimulus, per trial
	PSTH( param_filter_query(data_store, sheet_name=sheet, st_name=stimulus), ParameterSet({'bin_length':2.0, 'neurons':list(spike_ids) }) ).analyse()
	dsv = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus, analysis_algorithm='PSTH')
	asls = dsv.get_analysis_result( sheet_name=sheet )

	# store the mean PSTH (over the selected neurons) in a dict
	dt = 0.002 # sampling interval, PyNN default time is in ms, 'bin_length': 2.0
	fs = 1.029/dt # 
	nfft = 4 #int(fs*dt) # closest to 514 number of samples at 
	noverlap = 2 #int((fs*dt)/2.)
	# print nfft, noverlap
	psths = {}
	spectra = {}
	for asl in asls:
		# print asl
		stim = ast.literal_eval(str(asl.stimulus_id))[stimulus_parameter]
		trial = ast.literal_eval(str(asl.stimulus_id))['trial']
		# print stim, trial

		psth = []
		for idd in set(asl.ids).intersection(spike_ids): # get only the slected neurons
			psth.append( asl.get_asl_by_id(idd) )
		psth = numpy.mean(psth, axis=0) # mean over neurons

		if not "{:.3f}".format(stim) in psths:
			psths["{:.3f}".format(stim)] = []
		psths["{:.3f}".format(stim)].append( psth )


	powers = []
	for stim, ps in psths.items():
		print stim

		# mps = numpy.mean(ps, axis=0)
		# # trial-averaged psth spectrogram
		# plt.figure()
		# plt.specgram( mps, Fs=fs, vmin=-40, vmax=50)#, NFFT=nfft, noverlap=noverlap )
		# plt.tight_layout()
		# plt.savefig( folder+"/spectrogram_"+str(sheet)+"_"+stim+"_"+addon+".png", dpi=300, transparent=True )
		# plt.close()
		# gc.collect()

		# spectrum
		spectra = []
		for psth in ps:
			spectra.append( numpy.abs(numpy.fft.fft(psth))**2 )
			freqs = numpy.fft.fftfreq(psth.size, dt)

		power = numpy.mean(spectra, axis=0)
		
		powers.append( power )

	# 	plt.figure()
	# 	idx = numpy.argsort(freqs) # last one is as all the others
	# 	plt.plot( freqs[idx], power[idx], linewidth=3, color=color, linestyle='-' )
	# 	plt.ylim([0., 8000000.])
	# 	plt.xlim([0., 150.])
	# 	plt.tight_layout()
	# 	plt.savefig( folder+"/spectrum_"+str(sheet)+"_"+stim+"_"+addon+".png", dpi=300, transparent=True )
	# 	plt.close()
	# 	gc.collect()

	# 	# # find peaks
	# 	# peakind = signal.find_peaks_cwt(power, [1,2,3,4,5,6,7,8] )
	# 	# # print peakind, power[peakind]
	# 	# # amount of power in each band (computed from peaks)
	# 	# hist = []
	# 	# bands = [0, 4, 7, 13, 30, 100]
	# 	# for i in range(len(bands)-1):
	# 	# 	# print bands[i], bands[i+1]
	# 	# 	# sum the powers of peaks
	# 	# 	idxs = numpy.logical_and( peakind>=bands[i], peakind<=bands[i+1] )
	# 	# 	raw = numpy.sum( power[ idxs ] )
	# 	# 	band = raw / len(nidxs)
	# 	# 	hist.append( band )
	# 	# hist = hist / numpy.max(hist)
	# 	# # plotting
	# 	# plt.figure()
	# 	# plt.bar(range(5), hist, width=0.35, color=color)
	# 	# # plt.ylim([0., 200000000.])
	# 	# plt.tight_layout()
	# 	# plt.savefig( folder+"/hist_spectrum_"+str(sheet)+"_"+stim+"_"+addon+".png", dpi=300, transparent=True )
	# 	# plt.close()
	# 	# gc.collect()

	powers = numpy.array(powers)
	# print powers.shape

	powers = numpy.mean(powers, axis=0)
	idx = numpy.argsort(freqs) # last one is as all the others
	plt.figure()
	idx = numpy.argsort(freqs) # last one is as all the others
	plt.plot( freqs[idx], powers[idx], linewidth=3, color=color, linestyle='-' )
	plt.ylim([0., 8000000.])
	plt.xlim([0., 150.])
	plt.tight_layout()
	plt.savefig( folder+"/avg_spectrum_"+str(sheet)+"_"+stim+"_"+addon+".png", dpi=300, transparent=True )
	plt.savefig( folder+"/avg_spectrum_"+str(sheet)+"_"+stim+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()




def isi( sheet, folder, stimulus, stimulus_parameter, addon="", color="black", box=False, radius=False, opposite=False, ylim=[0.,5.] ):
	import ast
	from scipy.stats import norm
	matplotlib.rcParams.update({'font.size':22})

	print "folder: ",folder
	print addon
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )
	dsv = queries.param_filter_query(data_store, sheet_name=sheet, st_name=stimulus)
	segments = dsv.get_segments()
	spike_ids = dsv.get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons :", len(spike_ids)

	if radius or box:
		if sheet=='V1_Exc_L4':
			NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
			l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')[0]
			if opposite:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
			else:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
			print "# of V1 cells range having orientation 0:", len(l4_exc_or_many)
			spike_ids = list(l4_exc_or_many)

		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons :", len(spike_ids)

	segs, stids = colapse(dsv.get_segments(), dsv.get_stimuli(), parameter_list=[], allow_non_identical_objects=True ) # ALL spiketrains from all trials
	print len(segs), len(stids)

	nbins = 100
	isis = []
	cvs = []
	for st,seg in zip(stids,segs):
		stim = ast.literal_eval(str(st))[stimulus_parameter]

		isi = seg[0].isi(neuron_id=spike_ids)
		cv = seg[0].cv_isi(neuron_id=spike_ids)

		for i,c in zip(isi,cv):
			if stim > 0.6:
			 isis.append( numpy.nan_to_num( numpy.histogram( i, range=(0, nbins), bins=nbins )[0] ) )
			 if c==None: c = 0.
			 cvs.append( numpy.nan_to_num(c) )

	# plot
	x = range(nbins)
	# print numpy.array(isis).shape
	hisi = numpy.mean(isis, axis=0)
	avg_cv = numpy.mean(cvs)
	std_cv = numpy.std(cvs)
	# print hisi.shape

	frame = plt.gca()
	# Move left and bottom spines outward by 10 points
	frame.axes.spines['left'].set_position(('outward', 10))
	frame.axes.spines['bottom'].set_position(('outward', 10))
	# Hide the right and top spines
	frame.axes.spines['right'].set_visible(False)
	frame.axes.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	frame.axes.yaxis.set_ticks_position('left')
	frame.axes.xaxis.set_ticks_position('bottom')
	plt.title("{:.2f}".format(avg_cv)+" +- "+"{:.2f}".format(std_cv))
	plt.tight_layout()
	plt.ylim(ylim)
	plt.plot( x, hisi, linewidth=3, color=color) 
	plt.savefig( folder+"/ISI_"+str(sheet)+"_"+stimulus_parameter+"_x"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/ISI_"+str(sheet)+"_"+stimulus_parameter+"_x"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()




def windowed_isi( sheet, folder, stimulus, stimulus_parameter, addon="", color="black", box=False, opposite=False, ylim=[0.,200.] ):
	import ast
	from scipy.stats import norm
	matplotlib.rcParams.update({'font.size':22})

	print "folder: ",folder
	print addon
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )
	dsv = queries.param_filter_query(data_store, sheet_name=sheet, st_name=stimulus)
	segments = dsv.get_segments()
	spike_ids = dsv.get_segments()[0].get_stored_spike_train_ids()

	if box:
		if sheet=='V1_Exc_L4':
			NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
			l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')[0]
			if opposite:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
			else:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
			print "# of V1 cells range having orientation 0:", len(l4_exc_or_many)
			spike_ids = list(l4_exc_or_many)

		positions = data_store.get_neuron_postions()[sheet]
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
		box_ids = select_ids_by_position(positions, sheet_ids, box=box)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet,indexes=box_ids)

	sheet_ids = numpy.array( data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids) ).flatten()
	print "Recorded neurons :", len(spike_ids), len(sheet_ids)
	positions_x = numpy.take(data_store.get_neuron_postions()[sheet][0], sheet_ids)
	positions_y = numpy.take(data_store.get_neuron_postions()[sheet][1], sheet_ids)
	# print len(positions_x)

	segs, stids = colapse(dsv.get_segments(), dsv.get_stimuli(), parameter_list=[], allow_non_identical_objects=True) # ALL spiketrains from all trials
	# segs, stids = colapse(dsv.get_segments(), dsv.get_stimuli(), parameter_list=['trial'], allow_non_identical_objects=True) # colapse() randomly picks from the colapsed dimension?
	print len(segs), len(stids)

	# ISI
	# Each segs element contains spike_ids spiketrains of one trial
	# segs = [
	#   <SpikeTrain(array([  42.9,  367.7,  408.8,  866.1,  890.1]) * ms, [0.0 ms, 1029.0 ms])>  ... annotation: recorded cell, trial
	#   <SpikeTrain(array([  42.9,  367.7,  408.8,  866.1,  890.1]) * ms, [0.0 ms, 1029.0 ms])>  ... annotation: recorded cell, trial
	# ]
	# So we are going to compute the windowed ISI in this way:
	# iteration over segs
	#   windowed iteration over each seg
	#     store the isi_hist for the chunk isi_chunk
	#     when a seg has the same stim annotation, its isi_hist gets added to the same isi_chunk
	# isis_chunk has therefore a structure:
	# {
	# 	'stim': {
	#     'start_stop': [ 12,23,45,78, ... isi_hist #nbins ],
	#     'start_stop': [ 12,23,45,78, ... isi_hist #nbins ],
	#     '#  windows': [ 12,23,45,78, ... isi_hist #nbins ],
	#   }
	#   ...
	# 	'#stim': [...], 
	# }
	# for each seg
	# we iterate over start-to-stop chunks until the end
	# we compute the isi(numpy.diff) for each chunk 
	# we take the hist(nbins) and save it in isis_chunks['stim']['start_stop']
	# for the next seg, we add the hist result onto the previous

	nbins = 100
	# isis = []
	# d_isi_cvisi_fit = {}
	isis = {}
	# compute chunking
	trials = 6.
	total = 1*147*7 # from experiments.py
	t_start = 0
	t_stop = 0
	window = 300
	step = 150
	slides = int(total/step)
	print "... Computing ISI, CVisi"

	# init dict
	for i,seg in enumerate(segs):
		stim = ast.literal_eval( seg[0].annotations['stimulus'] )[stimulus_parameter]
		isis["{:.3f}".format(stim)] = {}
		for i in range(0,slides):
			t_start = step*i
			t_stop = t_start+window
			isis["{:.3f}".format(stim)][str(t_start)+'_'+str(t_stop)] = numpy.zeros(nbins)
	# print isis

	# actual computation
	for i,seg in enumerate(segs):
		stim = ast.literal_eval( seg[0].annotations['stimulus'] )[stimulus_parameter]
		print i, stim
		sts = numpy.array( seg[0].get_spiketrain(spike_ids) )
		# print sts.shape #, sts # one spiketrain per recorded neuron 
		for i in range(0,slides):
			t_start = step*i
			t_stop = t_start+window
			# print t_start, "< t <", t_stop
			_isi = [list(numpy.diff( st[(st >= t_start) & (st <= t_stop)])) for st in sts]
			isi = [item for sublist in _isi for item in sublist]
			# print isi
			_isis = isis["{:.3f}".format(stim)][str(t_start)+'_'+str(t_stop)]
			isis["{:.3f}".format(stim)][str(t_start)+'_'+str(t_stop)] = _isis + numpy.nan_to_num( numpy.histogram( isi, range=(0, nbins), bins=nbins )[0] )
			# cv_isi = numpy.std(isi)/numpy.mean(isi)
			# isis["{:.3f}".format(stim)][str(t_start)+'_'+str(t_stop)+'_cv'] = cv_isi

	# plotting
	for stim,windows in sorted(isis.items()):
		print stim
		for start_stop, hisi in sorted(windows.items()):
			# print start_stop
			# print max(hisi)
			x = range(nbins)
			frame = plt.gca()
			frame.axison = False
			frame.axes.get_xaxis().set_visible(False)
			frame.axes.get_yaxis().set_visible(False)
			plt.tight_layout()
					# plt.hist( hisi, bins=x, histtype=u'step', normed=True, linewidth=3, color=color )
			plt.ylim([0., 60.])
			plt.hist( hisi, bins=x, histtype=u'step', normed=False, linewidth=3, color=color )
			plt.savefig( folder+"/ISI_"+str(sheet)+"_"+stimulus_parameter+stim+"_win"+str(start_stop)+"_"+addon+".png", dpi=200, transparent=True )
			plt.close()
			gc.collect()




def phase_synchrony( sheet, folder, stimulus, stimulus_parameter, periods=[], addon="", color="black", box=False, opposite=False, percentage=False, fit='', ylim=[0.,200.] ):
	import ast
	from scipy.stats import norm
	matplotlib.rcParams.update({'font.size':22})

	print "folder: ",folder
	print addon
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )
	dsv = queries.param_filter_query(data_store, sheet_name=sheet, st_name=stimulus)
	segments = dsv.get_segments()
	spike_ids = dsv.get_segments()[0].get_stored_spike_train_ids()

	if box:
		if sheet=='V1_Exc_L4':
			NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
			l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')[0]
			if opposite:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
			else:
				l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
			print "# of V1 cells range having orientation 0:", len(l4_exc_or_many)
			spike_ids = list(l4_exc_or_many)

		positions = data_store.get_neuron_postions()[sheet]
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids)
		box_ids = select_ids_by_position(positions, sheet_ids, box=box)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet,indexes=box_ids)

	sheet_ids = numpy.array( data_store.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids) ).flatten()
	print "Recorded neurons :", len(spike_ids), len(sheet_ids)
	positions_x = numpy.take(data_store.get_neuron_postions()[sheet][0], sheet_ids)
	positions_y = numpy.take(data_store.get_neuron_postions()[sheet][1], sheet_ids)
	# print len(positions_x)

	# segs, stids = colapse(dsv.get_segments(), dsv.get_stimuli(), parameter_list=[], allow_non_identical_objects=True)
	segs, stids = colapse(dsv.get_segments(), dsv.get_stimuli(), parameter_list=['trial'], allow_non_identical_objects=True) # colapse() randomly picks from the colapsed dimension?
	print len(segs), len(stids)

	total = 1*147*7 # from experiments.py
	t_start = 0.
	t_stop = 0.
	window = 200.0
	step = 50.0
	slides = int(total/step)
	json_data = {}
	for i in range(0,slides):
		t_start = step*i
		t_stop = t_start+window
		print t_start, "< x <", t_stop

		vss = []
		for j,seg in enumerate(segs):
			print j #, seg[0].annotations

			vs = []
			sts = seg[0].get_spiketrain( spike_ids )
			# print len(sts)
			# print sts
			for j,st in enumerate(sts):
				# print st.annotations
				# print st.annotations['source_id']
				# print st.t_start, st.t_stop
				# if numpy.any()
				selected_st = st[numpy.where(numpy.logical_and(st>=t_start, st<=t_stop))]
				vs.append( scipy.signal.vectorstrength(selected_st, periods) )
				# print vs
			vss.append(vs)

		# PLOTTING
		vectorstrengths = numpy.nan_to_num( numpy.array(vss) )[:,:,0]
		vectorphases = numpy.nan_to_num( numpy.array(vss) )[:,:,1]
		print "vectorstrengths: ",vectorstrengths.shape
		# print "vectorstrengths: ",vectorstrengths[10][100]
		colors = vectorstrengths.argmax(axis=2)
		print "colors: ", colors.shape
		print "colors: ", colors

		##MAP
		cmap = plt.cm.jet
		cmaplist = [cmap(j) for j in range(cmap.N)]
		cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
		bounds = numpy.linspace(0, len(periods)-1, len(periods))
		norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		dat = {}
		for j,s in enumerate(stids):
			dat["{:.3f}".format(s.radius)] = numpy.histogram(colors[j], range(len(periods)), density=True)[0].tolist()
			# print s.radius
			for x,y,c in zip(positions_x, positions_y, colors[j]):
				plt.scatter( x, y, c=c, marker="o", edgecolors='none', cmap=cmap, norm=norm )
			cbar = plt.colorbar()
			plt.tight_layout()
			plt.savefig( folder+"/PhaseSynchronyMap_"+str(sheet)+"_"+"{:.3f}".format(s.radius)+"_"+addon+"_{:.0f}".format(t_start)+"_"+"{:.0f}".format(t_stop)+".png", dpi=200, transparent=True )
			plt.close()
			# garbage
			gc.collect()
		json_data[i] = dat

	with open(folder+"/PhaseSynchronyData_"+str(sheet)+"_"+addon+".txt", "w") as outfile:
		json.dump(json_data, outfile, indent=4)
		outfile.close()



				
def phase_synchrony_summary(sheet, folder, json_file, ylim=[], data_labels=False, addon=""):
	json_data = json.load( open(folder+"/"+json_file) )
	# print json_closed.keys()
	json_data = {int(k):v for k,v in json_data.items()} # atoi for json keys
	# print json_closed.keys()
	stimuli = numpy.array( [ [k for k,_ in sorted(d.items())] for _,d in json_data.items() ] )[0]
	# print stimuli
	# data = numpy.array( [ [v for _,v in sorted(d.items())] for _,d in json_data.items() ] ) # sorted array
	data = numpy.swapaxes( numpy.array( [ [v for _,v in sorted(d.items())] for _,d in json_data.items() ] ),0,1) # sorted and swapped array
	print(data.shape)
	# print(data[0])

	matplotlib.rcParams.update({'font.size':22})
	cmap = plt.cm.jet
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	bounds = numpy.linspace(0, data.shape[2]-1, data.shape[2])
	norm = matplotlib.colors.Normalize(vmin=0, vmax=bounds[-1])
	scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
	if not data_labels:
		data_labels = range(len(data))
	x = range(20)
	for i,d in enumerate(data):
		fig, axs = plt.subplots()
		d = numpy.swapaxes(d,0,1)
		# print i,d
		for c,l in enumerate(d):
			colorVal = scalarMap.to_rgba(c)
			axs.plot( x, l, linewidth=2, color=colorVal, label=data_labels[c] )
			std_l = numpy.std(l, ddof=1) 
			err_max_l = l + std_l
			err_min_l = l - std_l
			print "mean(std) open:", numpy.mean(std_l), "(", numpy.mean(std_l), ")"
			axs.fill_between(x, err_max_l, err_min_l, color=colorVal, alpha=0.3)

		axs.set_ylim(ylim)
		axs.set_aspect(aspect=8, adjustable='box')
		axs.spines['right'].set_visible(False)
		axs.spines['top'].set_visible(False)
		handles,labels = axs.get_legend_handles_labels()
		axs.legend(handles, labels, loc='upper right')
		axs.set_xlabel("Time (ms)")
		axs.set_ylabel("Phase Synchrony")
		axs.set_xticklabels(('0','300','600','900','1200'))
		lgd = axs.legend(bbox_to_anchor=(1.4, 1.1))
		plt.tight_layout()
		plt.savefig( folder+"/phase_comparison"+str(sheet)+"_"+addon+"_"+str(stimuli[i])+".png", dpi=300, transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight' )
		plt.close()




def spontaneous(sheet, folder, ylim=[], color="black", box=None, addon="", opposite=False):
	print "folder: ",folder
	print addon
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	TrialAveragedFiringRate(
		param_filter_query(data_store, st_direct_stimulation_name="None", st_name='InternalStimulus'),
		ParameterSet({})
	).analyse()

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_direct_stimulation_name="None", st_name='InternalStimulus').get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(spike_ids)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		if opposite:
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
		else:
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
		print "# of V1 cells range having orientation 0:", len(l4_exc_or_many)
		spike_ids = list(l4_exc_or_many)

	spont = numpy.array( param_filter_query(data_store, st_name='InternalStimulus', sheet_name=sheet, analysis_algorithm=['TrialAveragedFiringRate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids) )
	# spont = spont[ spont > 0.0 ] # to have actually spiking units

	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	ax.bar([0.], [numpy.mean(spont)], color=color, width=0.8)
	ax.errorbar(0.4, numpy.mean(spont), numpy.std(spont), lw=2, capsize=5, capthick=2, color=color)
	ax.set_xticklabels(['Spontaneous'])
	ax.set_ylabel("Response (spikes/s)")
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.set_ylim( ylim )
	plt.tight_layout()
	# plt.show()
	plt.savefig( folder+"/SpontaneousActivity_"+sheet+".png", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def correlation( sheet1, sheet2, folder, stimulus, stimulus_parameter, box1=None, box2=None, radius1=None, radius2=None, preferred1=True, preferred2=True, sizes=[], addon="", color="black" ):
	import ast
	from matplotlib.ticker import NullFormatter
	from scipy import signal

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	spike_ids1 = param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 1:", len(spike_ids1)
	spike_ids2 = param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 2:", len(spike_ids2)

	if sheet1=='V1_Exc_L4':
		spike_ids1 = select_by_orientation(data_store, sheet1, spike_ids1, preferred=preferred1)

	if sheet2=='V1_Exc_L4':
		spike_ids2 = select_by_orientation(data_store, sheet2, spike_ids2, preferred=preferred2)

	print "Oriented neurons 1:", len(spike_ids1)
	print "Oriented neurons 2:", len(spike_ids2)

	# if box1:
	# 	spike_ids1 = select_by_box(data_store, sheet1, spike_ids1, box=box1)

	# if box2:
	# 	spike_ids2 = select_by_box(data_store, sheet2, spike_ids2, box=box2)

	if radius1 or box1:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet1, neuron_ids=spike_ids1)
		positions = data_store.get_neuron_postions()[sheet1]
		if box1:
			ids = select_ids_by_position(positions, sheet_ids1, box=box1)
		if radius1:
			ids = select_ids_by_position(positions, sheet_ids1, radius=radius1)
		spike_ids1 = data_store.get_sheet_ids(sheet_name=sheet1, indexes=ids)

	if radius2 or box2:
		sheet_ids2 = data_store.get_sheet_indexes(sheet_name=sheet2, neuron_ids=spike_ids2)
		positions = data_store.get_neuron_postions()[sheet2]
		if box2:
			ids = select_ids_by_position(positions, sheet_ids2, box=box2)
		if radius2:
			ids = select_ids_by_position(positions, sheet_ids2, radius=radius2)
		spike_ids2 = data_store.get_sheet_ids(sheet_name=sheet2, indexes=ids)

	print "Selected neurons 1:", len(spike_ids1)
	print "Selected neurons 2:", len(spike_ids2)

	# compute PSTH per neuron, per stimulus, per trial
	bin = 2.0 
	tf = 2. # temporal freq (Hz)
	PSTH( param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus), ParameterSet({'bin_length':bin, 'neurons':list(spike_ids1) }) ).analyse()
	dsv1 = param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus, analysis_algorithm='PSTH')
	asls1 = dsv1.get_analysis_result( sheet_name=sheet1 )

	PSTH( param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus), ParameterSet({'bin_length':bin, 'neurons':list(spike_ids2) }) ).analyse()
	dsv2 = param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus, analysis_algorithm='PSTH')
	asls2 = dsv2.get_analysis_result( sheet_name=sheet2 )

	# store the mean PSTH (over the selected neurons) in a dict
	shift_predictors = {}
	correlations = {}
	psths1 = {}
	for asl in asls1:
		stim = ast.literal_eval(str(asl.stimulus_id))[stimulus_parameter]
		psth = []
		# for idd in set(asl.ids).intersection(spike_ids1): # get only the slected neurons
		for idd in spike_ids1: # get only the slected neurons
			psth.append( asl.get_asl_by_id(idd) )
		psth = numpy.mean(psth, axis=0) # mean over neurons
		if not "{:.3f}".format(stim) in psths1:
			psths1["{:.3f}".format(stim)] = []
			correlations["{:.3f}".format(stim)] = []
			shift_predictors["{:.3f}".format(stim)] = []
		psths1["{:.3f}".format(stim)].append( psth )

	psths2 = {}
	for asl in asls2:
		stim = ast.literal_eval(str(asl.stimulus_id))[stimulus_parameter]
		psth = []
		# for idd in set(asl.ids).intersection(spike_ids2): # get only the slected neurons
		for idd in spike_ids2:
			psth.append( asl.get_asl_by_id(idd) )
		psth = numpy.mean(psth, axis=0) # mean over neurons
		if not "{:.3f}".format(stim) in psths2:
			psths2["{:.3f}".format(stim)] = []
		psths2["{:.3f}".format(stim)].append( psth )

	# create all combinations
	psths1items = psths1.items()
	psths2items = psths2.items()
	# print "psths1items:", len(psths1items)
	# print "psths2items:", len(psths2items)
	combinations = [(x,y) for x in psths1items for y in psths2items]
	# print "combinations:", len(combinations)

	# perform correlation for each combination
	# for signal1, signal2 in zip( sorted(psths1.items()), sorted(psths2.items()) ):
	for signal1, signal2 in combinations:
		# print signal1[1][0].shape, signal2[1][0].shape
		# print numpy.array(signal1[1]).mean(axis=0).shape, numpy.array(signal2[1]).mean(axis=0).shape
		# shift_predictors[signal1[0]].append( numpy.correlate(numpy.array(signal1[1]).mean(axis=0), numpy.array(signal2[1]).mean(axis=0), "same") )
		for s1, s2 in zip(signal1[1], signal2[1]):
			correlations[signal1[0]].append( numpy.correlate(s1, s2, "same") )
			shift_predictors[signal1[0]].append( numpy.correlate(s1, numpy.roll(s2, int(len(s2)/tf)), "same" ) )

	# # CROSS CORRELATION COEFFICIENT
	# corrcoef = numpy.corrcoef( signal1, signal2 )
	# print "cross-correlation coefficient: ", corrcoef

	# # LINEAR REGRESSION
	# txtfile = open(folder+"/linear_regression_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".txt", 'w')
	# txtfile.write( "Selected neurons 1: {:.1f}\n".format(len(spike_ids1)) )
	# txtfile.write( "Selected neurons 2: {:.1f}\n".format(len(spike_ids2)) )
	# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( signal1, signal2 )
	# print "1-2: ", r_value**2, p_value, std_err
	# txtfile.write( "r: {:.4f}\n".format(corrcoef[0][1]) )
	# txtfile.write( "R2: {:.4f}\n".format(r_value**2) )
	# txtfile.write( "p:  {:.1E}\n".format(p_value) )
	# txtfile.write( "er: {:.2f}\n".format(std_err) )
	# txtfile.close()
	summary = {}
	xcraw = []
	final = []
	shpre = []
	for corr, shift in zip( sorted(correlations.items()), sorted(shift_predictors.items()) ):
		# print corr[0], corr[1], len(corr[1])
		summary[corr[0]] = []
		for i,(c,s) in enumerate(zip(corr[1],shift[1])):
			# print len(c), len(shift[1][0])
			xcraw.append( c )
			final.append( c-s )
			shpre.append( s )
			summary[corr[0]].append( c-s ) 
			# # plot
			x = range(len(c))
			# fig = plt.figure()
			# plt.plot( x, c, color="green", linewidth=1 )
			# plt.plot( x, c, color="green", linewidth=1 )
			# plt.plot( x, s, color="red", linewidth=1 )
			# plt.plot( x, c-s, color=color, linewidth=3 )
			# plt.axvline( x=int(len(c)/2.), color="black", linewidth=1 )
			# plt.ylim([-1000000, 3000000])
			# plt.tight_layout()
			# plt.savefig( folder+"/xcorr_rawshift_"+str(sheet1)+"_"+str(sheet2)+"_"+corr[0]+"_"+str(i)+"_"+addon+".png", dpi=300, transparent=True )
			# fig.clf()
			# plt.close()
			# gc.collect()

	xcraw = numpy.array(xcraw).mean(axis=0)
	final = numpy.array(final).mean(axis=0)
	shpre = numpy.array(shpre).mean(axis=0)
	fig = plt.figure()
	# full
	x = range(-len(x)/2, len(x)/2)
	plt.plot( x, xcraw, color='black', linewidth=2 )
	plt.plot( x, shpre, color='red', linewidth=2 )
	plt.plot( x, final, color=color, linewidth=3 )
	plt.axvline( x=0.0, color="black", linewidth=1 )
	# plt.axvline( x=int(len(final)/2.), color="black", linewidth=1 )
	# plt.ylim([-1500, 5000])
	plt.tight_layout()
	plt.savefig( folder+"/xcorr_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".png", dpi=300, transparent=True )
	plt.savefig( folder+"/xcorr_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	gc.collect()
	# zoom
	x = range(-50, 50)
	mid = int(len(final)/2.)
	plt.plot( x, xcraw[mid-50:mid+50], color='black', linewidth=2 )
	plt.plot( x, shpre[mid-50:mid+50], color='red', linewidth=2 )
	plt.plot( x, final[mid-50:mid+50], color=color, linewidth=3 )
	plt.axvline( x=0.0, color="black", linewidth=1 )
	# plt.ylim([-1500, 5000])
	plt.tight_layout()
	plt.savefig( folder+"/xcorr_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+"_zoom.png", dpi=300, transparent=True )
	plt.savefig( folder+"/xcorr_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+"_zoom.svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	gc.collect()

	print len(summary), summary.keys()
	maxi = []
	std = []
	for k in sorted(summary.keys()):
		maxi.append(numpy.array(summary[k]).max())
		std.append(numpy.array(summary[k]).std())
	maxi = numpy.array(maxi)
	# maxi = (maxi - maxi.mean()) / maxi.std() # normalization
	std = numpy.array(std)
	# std = (std - std.mean()) / std.std()

	fig = plt.figure()
	print "xcorr tuning:"
	print maxi 
	print std
	x = sizes #[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711]
	plt.plot( x, maxi, color=color, linewidth=3 )
	err_max = maxi + std
	err_min = maxi - std
	plt.fill_between( x, err_max, err_min, color=color, alpha=0.3 )
	plt.ylim([0, 200000])
	# plt.xscale('log')
	plt.tight_layout()
	plt.savefig( folder+"/xcorr_summary_max_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".png", dpi=300, transparent=True )
	plt.savefig( folder+"/xcorr_summary_max_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	gc.collect()
	
	# # AUTOCORRELATION
	# matplotlib.rcParams.update({'font.size':22})
	# fig = plt.figure()
	# ax1 = fig.add_subplot(211)
	# ax1.acorr( signal1, usevlines=True, normed=True, maxlags=200, linewidth=2, color=color )
	# ax1.spines['left'].set_visible(False)
	# ax1.spines['right'].set_visible(False)
	# ax1.spines['top'].set_visible(False)
	# ax1.spines['bottom'].set_visible(False)
	# ax2 = fig.add_subplot(212, sharex=ax1)
	# ax2.acorr( signal2, usevlines=True, normed=True, maxlags=200, linewidth=2, color=color )
	# ax2.spines['left'].set_visible(False)
	# ax2.spines['right'].set_visible(False)
	# ax2.spines['top'].set_visible(False)
	# ax2.spines['bottom'].set_visible(False)
	# plt.tight_layout()
	# plt.savefig( folder+"/autocorrelation_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".png", dpi=300, transparent=True )
	# fig.clf()
	# plt.close()
	# gc.collect()




def variability( sheet, folder, stimulus, stimulus_parameter, box=None, radius=None, addon="", opposite=False, nulltime=False ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	namef = ""
	if "feedforward" in folder:
		namef = "feedforward"
	if "open" in folder:
		namef = "open"
	if "closed" in folder:
		namef = "closed"
	if "Kimura" in folder:
		namef = "Kimura"
	if "LGNonly" in folder:
		namef = "LGNonly"
	if nulltime:
		namef = namef+"_nulltime"

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(spike_ids)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		if opposite:
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
		else:
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
		print "# of V1 cells range having orientation 0:", len(l4_exc_or_many)
		spike_ids = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(spike_ids)

	###############################################
	# sliding window variability
	discard = 130.0 # 900 ms remain
	total = (1*147*7)+1 # hardcoded, taken from experiments.py 
	window = 20.0
	step = 5.0
	slides = int(window/step)
	stimuli = []
	means = None
	variances = None
	for i in range(0,slides):
		start = discard+(step*i)
		s, m, v = get_per_neuron_window_spike_count( data_store, sheet, stimulus, stimulus_parameter, start, total, window=window, neurons=spike_ids, nulltime=nulltime )
		if i==0:
			stimuli = s
			means = m
			variances = v
		else:
			means = means + m
			variances = variances + v
	means = means/slides
	variances = variances/slides

	# ################## only for spontaneous
	# sp_means = numpy.mean(means, axis=2)
	# sp0_means = numpy.ones(sp_means.shape) * 0.0000000001
	# # print sp_means.shape, sp0_means.shape
	# means = numpy.swapaxes( numpy.array( [numpy.append(sp0_means, sp_means, axis=0)] ), 1,2)
	# variances = numpy.swapaxes( numpy.array( [numpy.append(sp0_means, numpy.ones(sp_means.shape)*numpy.std(means), axis=0 )] ), 1,2)
	# ################## only for spontaneous

	# print "means:", means.shape, means
	# print "variances:", variances.shape, variances
	###############################################

	# TIME COURSE
	print len(stimuli)
	# fig,ax = plt.subplots(nrows=len(stimuli)+1, ncols=means.shape[1], figsize=(70, 20), sharey=False, sharex=False)
	# fig.tight_layout()

	for i,s in enumerate(stimuli):
		print "stimulus:",s

		csvfile = open(folder+"/TrialAveragedMeanVariance_"+sheet+"_"+stimulus_parameter+"_"+"{:.2f}".format(s)+"_"+addon+"_"+namef+".csv", 'w')

		# each column is a different time bin
		for t in range(means.shape[1]):
			# print "\nTime bin:", t
			# each row is a different stimulus
			# print means[i][t].shape 
			# print variances[i][t].shape

			# replace zeros with very small positive values (arbitrary solution to avoid polyfit crash)
			means[i][t][ means[i][t]==0. ] = 0.000000001
			variances[i][t][ variances[i][t]==0. ] = 0.000000001
			means[i][t][ means[i][t]==float('Inf') ] = 100.0
			variances[i][t][ variances[i][t]==float('Inf') ] = 100.0

			# Churchland et al. 2010
			# # add regression line, whose slope is the Fano Factor
			# print [means[i][t]], [variances[i][t]]
			# k,b = numpy.polyfit( means[i][t], variances[i][t], 1)

			# Classic Fano Factor
			k = variances[i][t] / abs(means[i][t])
			print k

			# # Plotting
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

			# x = numpy.arange(x0, x1)
			# ax[i,t].plot(x, k*x+b, 'k-')
			# ax[i,t].set_title( "Fano:{:.2f}".format(k), fontsize=8 )
			# # text
			# ax[i,t].set_xlabel( "window "+str(t), fontsize=8 )
			# ax[i,t].set_ylabel( "{:.2f} \nCount variance".format(s), fontsize=8 )

			if csvfile:
				for f in k:
					csvfile.write( "{:.3f}\n".format(f) )

		if csvfile:
			csvfile.close()

	# plt.savefig( folder+"/TrialAveragedMeanVariance_"+stimulus_parameter+"_"+sheet+".png", dpi=200, transparent=True )
	# plt.savefig( folder+"/TrialAveragedMeanVariance_"+stimulus_parameter+"_"+sheet+".svg", dpi=200, transparent=True )
	# fig.clf()
	# plt.close()
	# # garbage
	# gc.collect()




def fano_comparison_timecourse( sheet, folder, closed_files, open_files, ylim=[], add_closed_files=[], add_open_files=[], averaged=False, sliced=[], addon="" ):
	from matplotlib import gridspec
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

	# interpolate nan values
	for i,(c,o) in enumerate(zip(closed_data,open_data)): 
		# print "stimulus", i

		# replace wrong polyfit with interpolated values
		err = numpy.argwhere(c>20) # indices of too big values
		c[err] = numpy.nan
		nans, idsf = numpy.isnan(c), lambda z: z.nonzero()[0] # logical indices of too big values, function with signature indices to converto logical indices to equivalent indices
		# c[nans] = numpy.interp( idsf(nans), idsf(~nans), c[~nans] )
		closed_data[i][nans] = numpy.interp( idsf(nans), idsf(~nans), c[~nans] )

		err = numpy.argwhere(o>20) # indices of too big values
		o[err] = numpy.nan
		nans, idsf = numpy.isnan(o), lambda z: z.nonzero()[0] # logical indices of too big values, function with signature indices to converto logical indices to equivalent indices
		# o[nans] = numpy.interp( idsf(nans), idsf(~nans), o[~nans] )
		open_data[i][nans] = numpy.interp( idsf(nans), idsf(~nans), o[~nans] )

	# PLOTTING
	x = numpy.arange( len(closed_data[0]) )
	matplotlib.rcParams.update({'font.size':22})
	s = numpy.sin(0.6*x + numpy.pi*1.3) # hadcoded, taken from experiments.py

	if averaged:

		c = numpy.mean(closed_data[slice(*sliced)], axis=0)
		o = numpy.mean(open_data[slice(*sliced)], axis=0)
		print c.shape

		std_c = numpy.std(c, ddof=1) 
		err_max_c = c + std_c
		err_min_c = c - std_c
		std_o = numpy.std(o, ddof=1) 
		err_max_o = o + std_o
		err_min_o = o - std_o

		st, p = scipy.stats.ttest_ind( numpy.clip(c,.0, None), numpy.clip(o,.0, None), equal_var=False )
		print "significance (t):", p
		chi2, p = scipy.stats.chisquare( numpy.clip(o,.0, None), numpy.clip(c,.0, None) )
		print "significance (chi):", p
		print "mean(std) open:", numpy.mean(numpy.clip(o,.0, None)), "(", numpy.mean(std_o), ")"
		print "mean(std) closed:", numpy.mean(numpy.clip(c,.0, None)), "(", numpy.mean(std_c), ")" 

		fig, (ax1, ax2) = plt.subplots(2, sharex=True)
		ax1.plot((0, len(closed_data[0])), (0,0), color="black")
		ax1.plot( x, o, linewidth=2, color="cyan" )
		ax1.fill_between(x, err_max_o, err_min_o, color="cyan", alpha=0.3)
		ax1.plot( x, c, linewidth=2, color='blue' )
		ax1.fill_between(x, err_max_c, err_min_c, color="blue", alpha=0.3)

		# plot input
		# ax2.plot( x, s, linewidth=2, color='black')

		ax1.set_ylim(ylim)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		# ax2.set_ylim(ylim)
		# ax2.spines['right'].set_visible(False)
		# ax2.spines['top'].set_visible(False)
		# ax2.spines['bottom'].set_visible(False)
		# ax2.set_xlabel("Time (ms)")
		# ax1.set_ylabel("Fano Factor")
		# fig.subplots_adjust(hspace=0)
		plt.tight_layout()
		plt.savefig( folder+"/fano_comparison"+str(sheet)+"_"+addon+"_mean"+str(sliced)+".png", dpi=300, transparent=True )
		plt.savefig( folder+"/fano_comparison"+str(sheet)+"_"+addon+"_mean"+str(sliced)+".svg", dpi=300, transparent=True )
		plt.close()

	else:
		for i,(c,o) in enumerate(zip(closed_data,open_data)): 
			print "stimulus", i

			std_c = numpy.std(c, ddof=1) 
			err_max_c = c + std_c
			err_min_c = c - std_c
			std_o = numpy.std(o, ddof=1) 
			err_max_o = o + std_o
			err_min_o = o - std_o

			fig, (ax1, ax2) = plt.subplots(2, sharex=True)
			ax1.plot((0, len(closed_data[0])), (0,0), color="black")
			ax1.plot( x, c, linewidth=2, color='blue' )
			ax1.fill_between(x, err_max_c, err_min_c, color="blue", alpha=0.3)
			ax1.plot( x, o, linewidth=2, color="cyan" )
			ax1.fill_between(x, err_max_o, err_min_o, color="cyan", alpha=0.3)

			# plot input
			ax2.plot( x, s, linewidth=2, color='black')

			ax1.set_ylim(ylim)
			ax1.spines['right'].set_visible(False)
			ax1.spines['top'].set_visible(False)
			ax1.spines['bottom'].set_visible(False)
			ax2.set_ylim(ylim)
			ax2.spines['right'].set_visible(False)
			ax2.spines['top'].set_visible(False)
			ax2.spines['bottom'].set_visible(False)
			ax2.set_xlabel("Time (ms)")
			ax1.set_ylabel("Fano Factor")
			# fig.subplots_adjust(hspace=0)
			plt.tight_layout()
			plt.savefig( folder+"/fano_comparison"+str(sheet)+"_"+str(i)+"_"+addon+".png", dpi=300, transparent=True )
			plt.savefig( folder+"/fano_comparison"+str(sheet)+"_"+str(i)+"_"+addon+".svg", dpi=300, transparent=True )
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




def end_inhibition_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None, opposite=False, box=None, radius=None, addon="" ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=True)

	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(neurons)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons]) < .1)[0]]
		neurons = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(neurons)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=False  )
	# print rates.shape # (stimuli, cells)
	# print stimuli

	# END-INHIBITION as in MurphySillito1987 and AlittoUsrey2008:
	# "The responses of the cell with corticofugal feedback are totally suppressed at bar lenghts of 2deg and above, 
	#  and those of cell lacking feedback are reduced up to 40% at bar lenghts of 8deg and above."
	# 1. find the peak response at large sizes
	peaks = numpy.amax(rates, axis=0) # as in AlittoUsrey2008
	# 2. compute average response at large sizes
	plateaus = numpy.mean( rates[5:], axis=0) 
	# 3. compute the difference from peak 
	ends = (peaks-plateaus)/peaks # as in MurphySillito1987
	print ends

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		data_mean = data_list[0]/10.
		data_list = data_list[1:]
		# https://mathoverflow.net/questions/14481/calculate-percentiles-from-a-histogram
		#1. stack the columns of the histogram on top of each other 
		tot_freq = data_list.sum()
		sum_data_list = numpy.cumsum(data_list, dtype=float) # cumulative sum for searching purposes
		#2. mark the point at a fraction p of the full height from the bottom
		val_q1 = tot_freq/4 # the 1st quartile is at 1/4 from the bottom
		val_q2 = val_q1*3 # the 3rd quartile is at 3/4 from the bottom
		#3. get the column corresponding to the quartile
		q1 = (numpy.abs(sum_data_list - val_q1)).argmin() / 10.# quartile in the index of end-inhibition
		q2 = (numpy.abs(sum_data_list - val_q2)).argmin() / 10.# quartile in the index of end-inhibition
		whislo = (numpy.where(data_list>1)[0][0]) / 10.
		whishi = (numpy.where(data_list>1)[0][-1]) / 10.
		bxp = {}
		bxp["label"] = '2' # not required
		bxp["mean"] = data_mean # not required
		bxp["med"] = data_mean #numpy.median(data_list)
		bxp["q1"] = q1
		bxp["q3"] = q2
		bxp["whislo"] = whislo
		bxp["whishi"] = whishi
		bxp["fliers"] = [] # required if showfliers=True
		# print bxp

	# PLOTTING
	c = 'cyan'
	cdata = 'grey'
	if closed:
		c = 'blue'
		cdata = 'black'
	matplotlib.rcParams.update({'font.size':22})
	fig = plt.figure()
	box1 = plt.boxplot( ends, positions=[1], notch=False, patch_artist=False, showfliers=False )
	for item in ['boxes', 'whiskers', 'medians', 'caps']:
		plt.setp(box1[item], color=c, linewidth=2, linestyle='solid')

	if data: # in front of the synthetic
		box2 = plt.boxplot( data_list, positions=[2], notch=False, patch_artist=False, showfliers=False)
		box2['caps'][0].set_ydata([bxp["whislo"], bxp["whislo"]])
		box2['caps'][1].set_ydata([bxp["whishi"], bxp["whishi"]])
		box2['whiskers'][0].set_ydata([bxp["whislo"], bxp["q1"]])
		box2['whiskers'][1].set_ydata([bxp["q3"], bxp["whishi"]])
		box2['boxes'][0].set_ydata([bxp["q1"],bxp["q1"],bxp["q3"],bxp["q3"],bxp["q1"]])
		box2['medians'][0].set_ydata([bxp["med"], bxp["med"]])
		# other styles
		for item in ['boxes', 'whiskers', 'medians', 'caps']:
			plt.setp(box2[item], color=cdata, linewidth=2, linestyle='solid')

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(0.5,3)
	plt.ylim(0.,1.)
	ax = plt.axes()
	ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(True)
	plt.tight_layout()
	plt.savefig( folder+"/suppression_index_box_"+str(sheet)+"_"+addon+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def end_inhibition_barplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None, csvfile=None ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=True)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False  )
	print rates.shape # (stimuli, cells)
	print stimuli
	width = 1.
	ind = numpy.arange(11)
	# ind = numpy.arange(10)

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
	ends = (peaks-plateaus)/peaks # as in MurphySillito1987
	print ends

	# 4. group cells by end-inhibition
	hist, edges = numpy.histogram( ends, bins=11, range=(0.0,1.0) )
	rawmean = numpy.mean(ends)
	print "rawmean:",rawmean
	mean = (rawmean*11)
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
		data_mean = data_list[0] +width/2 # positioning
		data_list = data_list[1:]
		print data_mean
		print data_list

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	if closed:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='blue', edgecolor='blue')
		ax.plot((mean, mean), (0,160), 'b--', linewidth=2)
	else:
		barlist = ax.bar(ind, hist, align='center', width=width, facecolor='cyan', edgecolor='cyan')
		ax.plot((mean, mean), (0,160), 'c--', linewidth=2)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([ind[0]-width/2, ind[-1], 0, 160])
	ax.set_xticks(ind) 
	ax.set_xticklabels(('0','','','','','0.5','','','','','1.0'))
	if data: # in front of the synthetic
		if closed:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='black', edgecolor='black')
			ax.plot((data_mean, data_mean), (0,160), 'k--', linewidth=2)
		else:
			datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='grey', edgecolor='grey')
			# datalist = ax.bar(ind, data_list, align='center', width=width, facecolor='none', edgecolor='grey', linewidth=2)
			ax.plot((data_mean, data_mean), (0,160), '--', linewidth=2, color='grey')
	plt.tight_layout()
	plt.savefig( folder+"/suppression_index_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def NakaRushton(c, n, Rmax, c50, m):
	return Rmax * (c**n / (c**n + c50**n)) + m
from scipy.optimize import curve_fit
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
	print sheet
	for i,r in enumerate(numpy.transpose(rates)):
		# bounds and guesses
		Rmax = numpy.amax(r) # 
		Rmax_up = Rmax + ((numpy.amax(r)/100)*10) # 
		m = numpy.amin(r) # 
		m_down = m - ((m/100)*10) # 
		# popt, pcov = curve_fit( NakaRushton, numpy.asarray(stimuli), r, maxfev=10000000 ) # workaround for scipy < 0.17
		popt, pcov = curve_fit( NakaRushton, numpy.asarray(stimuli), 
			r, 
			method='trf', 
			bounds=((3., Rmax, 20., m_down), (numpy.inf, Rmax_up, 50., m)), 
			p0=(3., Rmax_up, 30., m), 
		) 
		c50.append( popt[2] ) # c50 fit
		# print popt
		# plt.plot(stimuli, r, 'b-', label='data')
		# plt.plot(stimuli, NakaRushton(stimuli, *popt), 'r-', label='fit')
		# plt.savefig( folder+"/NakaRushton_fit_"+str(sheet)+"_"+str(i)+".png", dpi=100 )
		# plt.close()
	c50s = numpy.array(c50) 
	print c50s

	# count how many c50s are for each stimulus variation
	hist, bin_edges = numpy.histogram(c50s, bins=len(stimuli), range=(.0,60.) )
	print "histogram", hist, bin_edges
	# cumulative sum representation
	cumulative = numpy.cumsum(hist)
	cumulative = cumulative.astype(float) / numpy.amax(cumulative)
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




def orientation_selectivity_index_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", color_data="grey", data=None ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False )
	print rates.shape # (stimuli, cells)
	print stimuli

	# ORIENTATION SELECTIVITY INDEX as in Suematsu et al 2012:
	osi = numpy.zeros(rates.shape[1])
	den_osi = numpy.zeros(rates.shape[1])
	sin_osi = numpy.zeros(rates.shape[1])
	cos_osi = numpy.zeros(rates.shape[1])
	print osi.shape
	# For each stimulus orientation theta:
	for theta, r in zip(stimuli, rates):
		print theta#, r
		for i,R in enumerate(r):
			den_osi[i] = den_osi[i] + R
			sin_osi[i] = sin_osi[i] + (R * numpy.sin(2*theta))
			cos_osi[i] = cos_osi[i] + (R * numpy.cos(2*theta))
		for i,_ in enumerate(osi):
			osi[i] = numpy.sqrt( sin_osi[i]**2 + cos_osi[i]**2 ) / den_osi[i]
	print osi

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		# data_mean = data_list[0]
		# data_list = data_list[1:]
		# # https://mathoverflow.net/questions/14481/calculate-percentiles-from-a-histogram
		# #1. stack the columns of the histogram on top of each other 
		# tot_freq = data_list.sum()
		# sum_data_list = numpy.cumsum(data_list, dtype=float) # cumulative sum for searching purposes
		# # print tot_freq, sum_data_list
		# #2. mark the point at a fraction p of the full height from the bottom
		# val_q1 = tot_freq/4. # the 1st quartile is at 1/4 from the bottom
		# val_q2 = val_q1*3. # the 3rd quartile is at 3/4 from the bottom
		# # print val_q1, val_q2
		# #3. get the column corresponding to the quartile
		# # print numpy.abs(sum_data_list - val_q1)
		# # print numpy.abs(sum_data_list - val_q2)
		# q1 = numpy.abs(sum_data_list - val_q1) / 100 # quartile 
		# q2 = numpy.abs(sum_data_list - val_q2) / 100 # quartile 
		# whislo = (numpy.where(data_list>1)[0][0]) /10.
		# whishi = (numpy.where(data_list>1)[0][-1]) /10.
		# bxp = {}
		# bxp["label"] = '2' # not required
		# bxp["mean"] = data_mean # not required
		# bxp["med"] = data_mean #numpy.median(data_list)
		# bxp["q1"] = q1[0]
		# bxp["q3"] = q2[0]
		# bxp["whislo"] = whislo
		# bxp["whishi"] = whishi
		# bxp["fliers"] = [] # required if showfliers=True
		# print bxp

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	fig = plt.figure()
	box1 = plt.boxplot( osi, positions=[1], notch=False, patch_artist=False, showfliers=False )
	for item in ['boxes', 'whiskers', 'medians', 'caps']:
		plt.setp(box1[item], color=color, linewidth=2, linestyle='solid')

	if data: # in front of the synthetic
		box2 = plt.boxplot( data_list, positions=[2], notch=False, patch_artist=False, showfliers=False)
		# box2['caps'][0].set_ydata([bxp["whislo"], bxp["whislo"]])
		# box2['caps'][1].set_ydata([bxp["whishi"], bxp["whishi"]])
		# box2['whiskers'][0].set_ydata([bxp["whislo"], bxp["q1"]])
		# box2['whiskers'][1].set_ydata([bxp["q3"], bxp["whishi"]])
		# box2['boxes'][0].set_ydata([bxp["q1"],bxp["q1"],bxp["q3"],bxp["q3"],bxp["q1"]])
		# box2['medians'][0].set_ydata([bxp["med"], bxp["med"]])
		# # other styles
		for item in ['boxes', 'whiskers', 'medians', 'caps']:
			plt.setp(box2[item], color=color_data, linewidth=2, linestyle='solid')

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(0.5,3)
	plt.ylim(0.,1.)
	ax = plt.axes()
	ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(True)
	plt.tight_layout()
	plt.savefig( folder+"/orientation_selectivity_index_box_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def orientation_bias_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, spikecount=False )
	print rates.shape # (stimuli, cells)
	print stimuli

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
	print "bias", bias
	bias[bias > 10.] = 1.
	print "bias", bias
	print "Mean", numpy.mean(bias), "Std", numpy.std(bias)

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		data_mean = data_list[0]
		data_list = data_list[1:]
		# https://mathoverflow.net/questions/14481/calculate-percentiles-from-a-histogram
		#1. stack the columns of the histogram on top of each other 
		tot_freq = data_list.sum()
		sum_data_list = numpy.cumsum(data_list, dtype=float) # cumulative sum for searching purposes
		#2. mark the point at a fraction p of the full height from the bottom
		val_q1 = tot_freq/4 # the 1st quartile is at 1/4 from the bottom
		val_q2 = val_q1*3 # the 3rd quartile is at 3/4 from the bottom
		#3. get the column corresponding to the quartile
		q1 = (numpy.abs(sum_data_list - val_q1)).argmin() # quartile 
		q2 = (numpy.abs(sum_data_list - val_q2)).argmin() # quartile 
		whislo = (numpy.where(data_list>1)[0][0])
		whishi = (numpy.where(data_list>1)[0][-1])
		bxp = {}
		bxp["label"] = '2' # not required
		bxp["mean"] = data_mean # not required
		bxp["med"] = data_mean #numpy.median(data_list)
		bxp["q1"] = q1+1
		bxp["q3"] = q2+1
		bxp["whislo"] = whislo
		bxp["whishi"] = whishi
		bxp["fliers"] = [] # required if showfliers=True
		# print bxp

	# PLOTTING
	c = 'cyan'
	cdata = 'grey'
	if closed:
		c = 'blue'
		cdata = 'black'
	matplotlib.rcParams.update({'font.size':22})
	fig = plt.figure()
	box1 = plt.boxplot( bias, positions=[1], notch=False, patch_artist=False, showfliers=False )
	for item in ['boxes', 'whiskers', 'medians', 'caps']:
		plt.setp(box1[item], color=c, linewidth=2, linestyle='solid')

	if data: # in front of the synthetic
		box2 = plt.boxplot( data_list, positions=[2], notch=False, patch_artist=False, showfliers=False)
		box2['caps'][0].set_ydata([bxp["whislo"], bxp["whislo"]])
		box2['caps'][1].set_ydata([bxp["whishi"], bxp["whishi"]])
		box2['whiskers'][0].set_ydata([bxp["whislo"], bxp["q1"]])
		box2['whiskers'][1].set_ydata([bxp["q3"], bxp["whishi"]])
		box2['boxes'][0].set_ydata([bxp["q1"],bxp["q1"],bxp["q3"],bxp["q3"],bxp["q1"]])
		box2['medians'][0].set_ydata([bxp["med"], bxp["med"]])
		# other styles
		for item in ['boxes', 'whiskers', 'medians', 'caps']:
			plt.setp(box2[item], color=cdata, linewidth=2, linestyle='solid')

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(0.5,3)
	plt.ylim(1.,4.)
	ax = plt.axes()
	ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(True)
	plt.tight_layout()
	plt.savefig( folder+"/orientation_bias_box_"+str(sheet)+".png", dpi=200, transparent=True )
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




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], opposite=False, box=None, radius=None, addon="", data=None, data_curve=True ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(neurons)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons]) < .1)[0]]
		neurons = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(neurons)

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

	if parameter == "orientation":
		# append first mean and std to the end to close the circle
		print len(final_sorted), final_sorted
		final_sorted[0] = numpy.append(final_sorted[0], 3.14)
		final_sorted[1] = numpy.append(final_sorted[1], final_sorted[1][0])
		final_sorted[2] = numpy.append(final_sorted[2], final_sorted[2][0])

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
			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( stimuli, data_mean_rates )
			print "Data Slope:", slope, intercept
			ax.plot( stimuli, data_mean_rates, color='black', label='data' )
			data_err_max = data_mean_rates + data_std_rates
			data_err_min = data_mean_rates - data_std_rates
			ax.fill_between(stimuli, data_err_max, data_err_min, color='black', alpha=0.6)
		else:
			print stimuli, data_list[:,0], data_list[:,1]
			ax.scatter(stimuli, data_list[:,0], marker="o", s=80, facecolor="black", alpha=0.6, edgecolor="white")
			ax.scatter(stimuli, data_list[:,1], marker="D", s=80, facecolor="black", alpha=0.6, edgecolor="white")

	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( final_sorted[0], final_sorted[1] )
	print "Model Slope:", slope, intercept
	ax.plot( final_sorted[0], final_sorted[1], color=color, label=sheet, linewidth=3 )
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
	dist = box if not radius else radius
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+parameter+"_"+str(sheet)+"_"+addon+"_"+str(dist)+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+parameter+"_"+str(sheet)+"_"+addon+"_"+str(dist)+".svg", dpi=300, transparent=True )
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
		ax.scatter( data_full_list, data_inac_list, marker=data_marker, s=60, facecolor="black", edgecolor="white", label=sheet )

	ax.scatter( x_full, x_inac, marker="o", s=80, facecolor="blue", edgecolor="white", label=sheet )

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
	# ax.legend( loc="lower right", shadow=False, scatterpoints=1 )
	# plt.show()
	matplotlib.rcParams.update({'font.size':22})
	plt.tight_layout()
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
	increased_sem = 0.
	nochange_sem = 0.
	decreased_sem = 0.
	diff = x_full_rates - x_inac_rates
	print diff
	diff_sem = scipy.stats.sem(diff)
	print diff_sem

	for i in range(0,diff.shape[1]):
		# ANOVA significance test for open vs closed loop conditions
		# Null-hypothesis: "the rates in the closed- and open-loop conditions are equal"
		# full_inac_diff = scipy.stats.f_oneway(x_full_rates[:,i], x_inac_rates[:,i])
		full_inac_diff = scipy.stats.f_oneway(numpy.sqrt(x_full_rates[:,i]), numpy.sqrt(x_inac_rates[:,i]))
		print full_inac_diff
		# Custom test for open vs closed loop conditions
		# Low threshold, on the majority of tested frequencies
		if   numpy.sum( diff[:,i] > 0.5 ) > 4: 
			increased += 1
			increased_sem += diff_sem[i]
		elif numpy.sum( diff[:,i] < -0.5 ) > 4: 
			decreased += 1
			decreased_sem += diff_sem[i]
		else: 
			nochange += 1
			nochange_sem += diff_sem[i]
	increased /= diff.shape[1] 
	nochange /= diff.shape[1]
	decreased /= diff.shape[1]
	increased_sem /= diff.shape[1] 
	nochange_sem /= diff.shape[1]
	decreased_sem /= diff.shape[1]
	increased *= 100.
	nochange *= 100.
	decreased *= 100.
	increased_sem *= 100.
	nochange_sem *= 100.
	decreased_sem *= 100.
	print "increased", increased, "% +-", increased_sem
	print "nochange", nochange, "% +-", nochange_sem
	print "decreased", decreased, "% +-", decreased_sem

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
	# print reduction

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	ind = numpy.arange(len(reduction))
	fig, ax = plt.subplots()

	barlist = ax.bar(numpy.arange(8),[increased,44,0,nochange,36,0,decreased,20], width=0.99) # empty column to separate groups
	barlist[0].set_color('blue')
	barlist[1].set_color('black')
	barlist[3].set_color('blue')
	barlist[4].set_color('black')
	barlist[6].set_color('blue')
	barlist[7].set_color('black')
	ax.errorbar(0.5, increased, yerr=increased_sem, color='blue', capsize=20, capthick=3, elinewidth=3 )
	ax.errorbar(3.5, nochange, yerr=nochange_sem, color='blue', capsize=20, capthick=3, elinewidth=3 )
	ax.errorbar(6.5, decreased, yerr=decreased_sem, color='blue', capsize=20, capthick=3, elinewidth=3 )
	ax.errorbar(1.5, 44, yerr=6.6, color='black', capsize=20, capthick=3, elinewidth=3 )
	ax.errorbar(4.5, 36, yerr=5.4, color='black', capsize=20, capthick=3, elinewidth=3 )
	ax.errorbar(7.5, 20, yerr=3., color='black', capsize=20, capthick=3, elinewidth=3 )

	ax.set_xticklabels(['','increased','','','no change','','','decreased'])
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontsize(18)
	# ax.set_yticklabels(['0','10','20','30','40',''])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_ylabel("Number of cells %", fontsize=18)

	# ax2 = fig.add_axes([.7, .7, .2, .2])
	# ax2.bar( ind, reduction, width=0.99, facecolor='blue', edgecolor='blue')
	# ax2.set_xlabel(xlabel)
	# ax2.set_ylabel(ylabel)
	# # ax2.spines['right'].set_visible(False)
	# # ax2.spines['top'].set_visible(False)
	# # ax2.spines['bottom'].set_visible(False)
	# ax2.set_xticklabels( ['.05', '.2', '1.2', '3.', '6.4', '8', '12', '30'] )
	# ax2.axis([0, 8, -1, 1])
	# for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	# 	label.set_fontsize(9)

	plt.tight_layout()
	plt.savefig( folder_inactive+"/response_reduction_"+str(sheet)+".png", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_conductance_tuning_curve( sheet, folder, stimulus, parameter, percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], box=[], addon="", inputoutputratio=False, dashed=False ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )
	print "analog_ids (pre): ",analog_ids

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )
	print "Recorded neurons:", len(analog_ids)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = sheet)[0]
		l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < 0.1)[0]]
		# l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in analog_ids]) < 0.9)[0]]
		analog_ids = list(l4_exc_or_many)

	if box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
		positions = data_store.get_neuron_postions()[sheet]
		ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		analog_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(analog_ids)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	ticks = set([])
	for x in segs:
		ticks.add( getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) )
	ticks = sorted(ticks)
	num_ticks = len( ticks )
	print ticks
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
	print "mean std exc:", mean_pop_e, std_pop_e
	print "mean std inh:", mean_pop_i, std_pop_i

	final_sorted_e = [ numpy.array(list(e)) for e in zip( *sorted( zip(ticks, mean_pop_e, std_pop_e) ) ) ]
	final_sorted_i = [ numpy.array(list(e)) for e in zip( *sorted( zip(ticks, mean_pop_i, std_pop_i) ) ) ]

	if parameter == "orientation":
		# append first mean and std to the end to close the circle
		final_sorted_e[0] = numpy.append(final_sorted_e[0], 3.14)
		final_sorted_e[1] = numpy.append(final_sorted_e[1], final_sorted_e[1][0])
		final_sorted_e[2] = numpy.append(final_sorted_e[2], final_sorted_e[2][0])
		final_sorted_i[0] = numpy.append(final_sorted_i[0], 3.14)
		final_sorted_i[1] = numpy.append(final_sorted_i[1], final_sorted_i[1][0])
		final_sorted_i[2] = numpy.append(final_sorted_i[2], final_sorted_i[2][0])

	if percentile:
		firing_max = numpy.amax( final_sorted_e[1] )
		final_sorted_e[1] = final_sorted_e[1] / firing_max * 100
		firing_max = numpy.amax( final_sorted_i[1] )
		final_sorted_i[1] = final_sorted_i[1] / firing_max * 100

	# Plotting tuning curve
	matplotlib.rcParams.update({'font.size':22})
	fig,ax = plt.subplots()
	alpha = 0.1
	linestyle = '-'
	if dashed:
		linestyle='--'

	err_max = final_sorted_i[1] + final_sorted_i[2]
	max_i = numpy.amax(err_max)
	err_min = final_sorted_i[1] - final_sorted_i[2]
	ax.plot( final_sorted_i[0], final_sorted_i[1], color='blue', linewidth=3, linestyle=linestyle )
	ax.fill_between(final_sorted_i[0], err_max, err_min, color='blue', alpha=alpha, linewidth=1)
	if dashed:
		ax.plot( final_sorted_i[0], err_max, color='blue', linewidth=1, linestyle=linestyle )
		ax.plot( final_sorted_i[0], err_min, color='blue', linewidth=1, linestyle=linestyle )

	err_max = final_sorted_e[1] + final_sorted_e[2]
	max_e = numpy.amax(err_max)
	err_min = final_sorted_e[1] - final_sorted_e[2]
	ax.plot( final_sorted_e[0], final_sorted_e[1], color='red', linewidth=3, linestyle=linestyle )
	ax.fill_between(final_sorted_e[0], err_max, err_min, color='red', alpha=alpha, linewidth=1)
	if dashed:
		ax.plot( final_sorted_e[0], err_max, color='red', linewidth=1, linestyle=linestyle )
		ax.plot( final_sorted_e[0], err_min, color='red', linewidth=1, linestyle=linestyle )

	if not percentile:
		ax.set_ylim(ylim)
		ax.set_ylabel( "Conductance (nS)" )
	else:
		ax.set_ylim([0, max_i if max_i > max_e else max_e])
		ax.set_ylabel( "Conductance change (%)" )

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
	plt.tight_layout()
	plt.savefig( folder+"/TrialAveragedConductances_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".png", dpi=200, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()

	if inputoutputratio:
		neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, 100., 1000., parameter, neurons=neurons, spikecount=False ) 
		# compute per-trial mean rate over cells
		mean_rates = numpy.mean(rates, axis=1) / numpy.max(rates)
		final_sorted = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates) ) ) ]
		ratio = final_sorted[1] / final_sorted_e[1]
		fig,ax = plt.subplots()
		ax.plot( final_sorted_e[1], final_sorted[1], color='cyan', linewidth=2 )
		# ax.plot( final_sorted_e[0], ratio, color='red', linewidth=2 )
		ax.set_xlim([.0,6.])
		ax.set_xlabel( "Input (nS)" )
		ax.set_ylim([.0,1.])
		ax.set_ylabel( "Spike probability" )
		plt.tight_layout()
		plt.savefig( folder+"/TrialAveragedInputOutputRatio_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".png", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()




def trial_averaged_conductance_timecourse( sheet, folder, stimulus, parameter, ticks, ylim=[0.,100.], box=[], addon="" ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )
	print "Recorded neurons:", len(analog_ids)

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = sheet)[0]
		l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < 0.2)[0]]
		# l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in analog_ids]) < 0.9)[0]]
		analog_ids = list(l4_exc_or_many)

	if box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
		positions = data_store.get_neuron_postions()[sheet]
		ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		analog_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(analog_ids)

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
		t_stop = 200.0 #float(full_gsyn_es[0].t_stop - sampling_period)
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
	# print pop_e.shape
	mean_pop_e = numpy.mean(pop_e, axis=0 )
	mean_pop_i = numpy.mean(pop_i, axis=0 ) 
	std_pop_e = numpy.std(pop_e, axis=0, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	std_pop_i = numpy.std(pop_i, axis=0, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	print "mean std:", mean_pop_e.shape, std_pop_e.shape
	# print "mean std exc:", mean_pop_e, std_pop_e
	# print "mean std inh:", mean_pop_i, std_pop_i

	for s in range(num_ticks):
		# for each stimulus plot the average conductance per cell over time
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()

		err_max = mean_pop_i[s] + std_pop_i[s]
		err_min = mean_pop_i[s] - std_pop_i[s]
		ax.fill_between(range(0,len(mean_pop_i[s])), err_max, err_min, color='blue', alpha=0.3)
		ax.plot( range(0,len(mean_pop_i[s])), mean_pop_i[s], color='blue', linewidth=2 )

		err_max = mean_pop_e[s] + std_pop_e[s]
		err_min = mean_pop_e[s] - std_pop_e[s]
		ax.fill_between(range(0,len(mean_pop_e[s])), err_max, err_min, color='red', alpha=0.3)
		ax.plot( range(0,len(mean_pop_e[s])), mean_pop_e[s], color='red', linewidth=2 )

		ax.set_ylim(ylim)
		ax.set_ylabel( "Conductance (nS)" )
		ax.set_xlabel( "Time (ms)" )

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.xaxis.set_ticks(ticks, ticks)
		ax.yaxis.set_ticks_position('left')

		# text
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseConductances_"+sheet+"_"+parameter+"_"+str(ticks[s])+".png", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()




def SpikeTriggeredAverage(sheet, folder, stimulus, parameter, ylim=[0.,100.], box=False, radius=False, opposite=False, addon="", color="black"):
	import ast
	from scipy import signal
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=False)

	# LFP
	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()
	print "LFP neurons:", len(neurons)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	ticks = set([])
	for x in segs:
		ticks.add( getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) )
	ticks = sorted(ticks)
	num_ticks = len( ticks )
	print ticks
	trials = len(segs) / num_ticks
	print "trials:",trials

	pop_vm = []
	pop_gsyn_e = []
	pop_gsyn_i = []
	for n,idd in enumerate(neurons):
		print "idd", idd
		full_vm = [s.get_vm(idd) for s in segs]
		full_gsyn_es = [s.get_esyn(idd) for s in segs]
		full_gsyn_is = [s.get_isyn(idd) for s in segs]
		# print "len full_gsyn_e/i", len(full_gsyn_es) # 61 = 1 spontaneous + 6 trial * 10 num_ticks
		# print "shape gsyn_e/i", full_gsyn_es[0].shape
		# mean input over trials
		mean_full_vm = numpy.zeros((num_ticks, full_vm[0].shape[0])) # init
		mean_full_gsyn_e = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0])) # init
		mean_full_gsyn_i = numpy.zeros((num_ticks, full_gsyn_es[0].shape[0]))
		# print "shape mean_full_gsyn_e/i", mean_full_gsyn_e.shape
		sampling_period = full_gsyn_es[0].sampling_period
		t_stop = float(full_gsyn_es[0].t_stop - sampling_period) # 200.0
		t_start = float(full_gsyn_es[0].t_start)
		time_axis = numpy.arange(0, len(full_gsyn_es[0]), 1) / float(len(full_gsyn_es[0])) * abs(t_start-t_stop) + t_start
		# sum by size
		t = 0
		for v,e,i in zip(full_vm, full_gsyn_es, full_gsyn_is):
			s = int(t/trials)
			v = v.rescale(mozaik.tools.units.mV) 
			e = e.rescale(mozaik.tools.units.nS) #e=e*1000
			i = i.rescale(mozaik.tools.units.nS) #i=i*1000
			mean_full_vm[s] = mean_full_vm[s] + numpy.array(v.tolist())
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] + numpy.array(e.tolist())
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] + numpy.array(i.tolist())
			t = t+1

		# average by trials
		for s in range(num_ticks):
			mean_full_vm[s] = mean_full_vm[s] / trials
			mean_full_gsyn_e[s] = mean_full_gsyn_e[s] / trials
			mean_full_gsyn_i[s] = mean_full_gsyn_i[s] / trials

		pop_vm.append(mean_full_vm)
		pop_gsyn_e.append(mean_full_gsyn_e)
		pop_gsyn_i.append(mean_full_gsyn_i)

	pop_v = numpy.array(pop_vm)
	pop_e = numpy.array(pop_gsyn_e)
	pop_i = numpy.array(pop_gsyn_i)

	# mean and std over cells
	# print pop_e.shape
	mean_pop_v = numpy.mean(pop_v, axis=0 )
	mean_pop_e = numpy.mean(pop_e, axis=0 )
	mean_pop_i = numpy.mean(pop_i, axis=0 ) 
	std_pop_v = numpy.std(pop_v, axis=0, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	std_pop_e = numpy.std(pop_e, axis=0, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	std_pop_i = numpy.std(pop_i, axis=0, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	# print "mean std:", mean_pop_v.shape, mean_pop_e.shape, std_pop_e.shape
	# print "mean std exc:", mean_pop_e, std_pop_e
	# print "mean std inh:", mean_pop_i, std_pop_i

	# We produce the current for each cell for this time interval, with the Ohm law:
	# I = g(V-E), where E is the equilibrium for exc, which usually is 0.0 (we can change it)
	# (and we also have to consider inhibitory condictances)
	i = (pop_e/1000)*pop_v + (pop_i/1000)*pop_v # NEST is in nS, PyNN is in uS
	# the LFP is the result of cells' currents
	avg_i = numpy.mean( i, axis=0 )
	sigma = 0.1 # [0.1, 0.01] # Dobiszewski_et_al2012.pdf
	lfp = (1/(4*numpy.pi*sigma)) * avg_i
	print "LFP:", lfp.shape
	# print lfp

	# for s in range(num_ticks):
	# 	# for each stimulus plot the average conductance per cell over time
	# 	matplotlib.rcParams.update({'font.size':22})
	# 	fig,ax = plt.subplots()

	# 	ax.plot( range(0,len(lfp[s])), lfp[s], color=color, linewidth=3 )

	# 	# ax.set_ylim(ylim)
	# 	ax.set_ylabel( "LFP (mV)" )
	# 	ax.set_xlabel( "Time (ms)" )

	# 	ax.spines['right'].set_visible(False)
	# 	ax.spines['top'].set_visible(False)
	# 	ax.xaxis.set_ticks_position('bottom')
	# 	ax.xaxis.set_ticks(ticks, ticks)
	# 	ax.yaxis.set_ticks_position('left')

	# 	# text
	# 	plt.tight_layout()
	# 	plt.savefig( folder+"/TimecourseLFP_"+sheet+"_"+parameter+"_"+str(ticks[s])+".png", dpi=200, transparent=True )
	# 	fig.clf()
	# 	plt.close()
	# 	# garbage
	# 	gc.collect()

	# SUA
	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()

	if sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons]) < .1)[0]]
		neurons = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids)

	print "SUA neurons:", len(neurons)
	if len(neurons) < 1:
		return

	print "Collecting spiketrains of selected neurons into dictionary ..."
	dsv1 = queries.param_filter_query(data_store, sheet_name=sheet, st_name=stimulus)
	sts = {}
	for st, seg in zip([MozaikParametrized.idd(s) for s in dsv1.get_stimuli()], dsv1.get_segments()):
		# print st, seg
		stim = ast.literal_eval(str(st))[parameter]
		trial = ast.literal_eval(str(st))['trial']
		if not "{:.3f}".format(stim) in sts:
			sts["{:.3f}".format(stim)] = {}
		for idd in neurons:
			if not str(idd) in sts["{:.3f}".format(stim)]:
				sts["{:.3f}".format(stim)][str(idd)] = [ [] for _ in range(trials) ]
			sts["{:.3f}".format(stim)][str(idd)][trial] = seg.get_spiketrain(idd)
	# print sts

	# STA
	print "Computing Spike Triggered Average"
	# Short segments of fixed duration are selected from the LFP, based on the timing of occurrence of spikes and then averaged on the number of spikes
	duration = 300 # ms
	STA = numpy.zeros( (len(sts), len(neurons), 2*duration) )
	print STA.shape
	for i,(st,neurons) in enumerate(sorted(sts.items())):
		# print st
		for j,(idd,spiketrains) in enumerate(neurons.items()):
			# print "\t", idd
			for spiketrain in spiketrains:
				# print spiketrain
				for time in spiketrain:
					start = max(0., time.item() - duration)
					end = min(spiketrain.t_stop.item(), time.item() + duration)
					# print "\t\t", start, time, end, "=", end-start
					segment = lfp[i][int(start):int(end)]
					if len(segment) < 2*duration:
						if start==0.0:
							segment = numpy.pad(segment, (duration-time.item()+1,0), 'constant', constant_values=(0.,0.))
						if end>=spiketrain.t_stop.item():
							segment = numpy.pad(segment, (0,time.item()-duration), 'constant', constant_values=(0.,0.))
					# print len(segment)
					STA[i][j] = STA[i][j] + segment[:2*duration]
			STA[i][j] = STA[i][j] / len(spiketrains)

	# print STA
	STA_tuning = []
	for i,(st,neurons) in enumerate(sorted(sts.items())):
		STA_tuning.append( STA[i].min() )
		sta_mean = STA[i].mean(axis=0) - STA[i].mean() # mean corrected
		# sta_std = STA[i].std(axis=0) - STA[i].std()
		fig = plt.figure()
		x = range(-duration, duration)
		plt.plot( x, sta_mean, color=color, linewidth=3 )
		# plt.fill_between(x, sta_mean-sta_std, sta_mean+sta_std, color=color, alpha=0.3)
		plt.tight_layout()
		plt.ylim([-4., 1.5])
		plt.savefig( folder+"/STA_"+str(sheet)+"_"+st+"_"+addon+".png", dpi=300, transparent=True )
		plt.savefig( folder+"/STA_"+str(sheet)+"_"+st+"_"+addon+".svg", dpi=300, transparent=True )
		fig.clf()
		plt.close()

	print STA_tuning
	fig = plt.figure()
	plt.plot( range(len(STA_tuning)), numpy.array(STA_tuning)*-1., color=color, linewidth=3 ) # reversed
	plt.ylim([0, 160])
	plt.tight_layout()
	plt.savefig( folder+"/STAtuning_"+str(sheet)+"_"+addon+".png", dpi=300, transparent=True )
	plt.savefig( folder+"/STAtuning_"+str(sheet)+"_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
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
	# "Deliverable/ThalamoCorticalModel_data_contrast_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_LGNonly_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "Thalamocortical_size_closed", # BIG
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____",
	"ThalamoCorticalModel_data_size_closed_____", # <<<<<<< ISO Coherence
	# "ThalamoCorticalModel_data_size_closed_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_open_____",
	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____old",
	# "Deliverable/ThalamoCorticalModel_data_size_nonoverlapping_____",

	# "Thalamocortical_size_feedforward", # BIG
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____",
	"ThalamoCorticalModel_data_size_feedforward_____", # <<<<<<< ISO Coherence
	# "Deliverable/ThalamoCorticalModel_data_size_LGNonly_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____nocorr",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____nocorr",

	# # Andrew's machine
	# "/data1/do/ThalamoCorticalModel_data_size_open_____",
	# "/data1/do/ThalamoCorticalModel_data_size_closed_many_____",
	# "/data1/do/ThalamoCorticalModel_data_size_nonoverlapping_many_____",
	# "/data1/do/ThalamoCorticalModel_data_size_overlapping_many_____",

	# "ThalamoCorticalModel_data_orientation_feedforward_____2",
	# "ThalamoCorticalModel_data_orientation_closed_____2",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_open_____",

	# "ThalamoCorticalModel_data_xcorr_open_____1", # just one trial
	# "ThalamoCorticalModel_data_xcorr_open_____2deg", # 2 trials
	# "ThalamoCorticalModel_data_xcorr_closed_____2deg", # 2 trials

	# "Deliverable/CombinationParamSearch_LGN_PGN_core",
	# "Deliverable/CombinationParamSearch_LGN_PGN_2",
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


addon = ""
# sheets = ['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']
# sheets = ['X_ON', 'X_OFF', 'PGN']
# sheets = [ ['X_ON', 'X_OFF'], 'PGN']
# sheets = [ ['X_ON', 'X_OFF'] ]
# sheets = [ 'X_ON', 'X_OFF', ['X_ON', 'X_OFF'] ]
# sheets = ['X_ON', 'X_OFF', 'V1_Exc_L4']
# sheets = ['X_ON', 'X_OFF']
# sheets = ['X_ON']
# sheets = ['X_OFF'] 
# sheets = ['PGN']
sheets = ['V1_Exc_L4'] 
# sheets = ['V1_Exc_L4', 'V1_Inh_L4'] 


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

		# color = "saddlebrown"
		color = "black"
		if "feedforward" in f:
			addon = "feedforward"
			closed = False
			color = "cyan"
			color_data = "grey"
			fit = "gamma"
			# color = "red"
			trials = 12
		if "open" in f:
			closed = False
			color = "cyan"
			color_data = "grey"
		if "closed" in f:
			addon = "closed"
			color = "blue"
			color_data = "black"
			closed = True
			fit = "bimodal"
			trials = 6
		if "Kimura" in f:
			color = "gold"
		if "LGNonly" in f:
			color = "yellow"

		for s in sheets:

			if 'PGN' in s:
				arborization = 300
				color = "darkgreen"
				if "open" in f or "feedforward" in f:
					color = "lime"
				if "closed" in f:
					color = "darkgreen"
			if 'X' in s: # X_ON X_OFF
				arborization = 150

			print color

			# SPONTANEOUS
			# spontaneous(
			# 	sheet=s, 
			# 	folder=f, 
			# 	color=color,
			# 	ylim=[0., 7.],
			# 	addon=addon
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='InternalStimulus',
			# 	stimulus_parameter='duration', # dummy parameter to make it work with colapse()
			# 	addon = addon,
			# 	opposite=False, # SAME
			# )

			# # LUMINANCE
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

			# # CONTRAST
			# trial_averaged_tuning_curve_errorbar( 
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
			# 	percentile=False, #False, # True,
			# 	ylim=[0,35], #[0,35], # [0,120],
			# )
			# cumulative_distribution_C50_curve( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='contrast',
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Half maximal contrast response (C$_{50}$)",
			# 	ylabel="Number of cells (%)",
			# 	color=color,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_closed.csv",
			# 	# data_color="black",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/LiYeSongYangZhou2011c_open.csv",
			# 	# data_color="grey",
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="contrast",
			# 	percentile=False,
			# 	ylim=[0,10],
			# 	useXlog=False, 
			# 	addon = "open",
			# 	inputoutputratio = True,
			# )
			# trial_averaged_conductance_timecourse( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="contrast",
			# 	ticks=[0.0, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.],
			# 	ylim=[0,20],
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	addon = "center",
			# 	# box = [[-.5,.0],[.5,.8]], # mixed surround (more likely to be influenced by the recorded thalamus)
			# 	# box = [[-.5,.5],[.5,1.]], # strict surround
			# 	# box = [[-0.1,.6],[.3,1.]], # surround
			# )

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
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="Spatial frequency", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=True, 
			# 	useYlog=False, 
			# 	percentile=False, #False 
			# 	ylim=[0.,100.],
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=True, 
			# 	addon = "",
			# )

			# SIZE
			# Ex: ThalamoCorticalModel_data_size_V1_full_____
			# Ex: ThalamoCorticalModel_data_size_open_____
			# end_inhibition_barplot( 
			# end_inhibition_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	xlabel="exp",
			# 	ylabel="Index of end-inhibition",
			# 	closed=closed,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_open.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_closed.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3E.csv", # closed drifting gratings
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3F.csv", # retinal drifting gratings
			# )
			# end_inhibition_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	closed=closed,
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# )
			# end_inhibition_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	closed=closed,
			# 	opposite=True, # to select cortical cells with OPPOSITE orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# )
			# end_inhibition_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	closed=closed,
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	# box = [[-.8,1.],[.8,2.5]], # surround
			# 	radius = [1.,4.], # surround
			# 	addon = "surround",
			# )
			# end_inhibition_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	closed=closed,
			# 	opposite=True, #
			# 	# box = [[-.8,1.],[.8,2.5]], # surround
			# 	radius = [1.,4.], # surround
			# 	addon = "surround",
			# )

			# trial_averaged_tuning_curve_errorbar( 
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
			# 	percentile=True,
			# 	ylim=[0,50],
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
			# 	# data_curve=False,
			# )
			# trial_averaged_tuning_curve_errorbar( 
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
			# 	# percentile=False, #True,
			# 	percentile=True,
			# 	ylim=[0,50],
			# 	opposite=True, # to select cortical cells with OPPOSITE orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# )
			# trial_averaged_tuning_curve_errorbar( 
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
			# 	# percentile=False, #True,
			# 	percentile=True,
			# 	ylim=[0,100],
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	# box = [[-.8,1.],[.8,2.5]], # surround
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround",
			# )
			# trial_averaged_tuning_curve_errorbar( 
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
			# 	# percentile=False, #True,
			# 	percentile=True,
			# 	ylim=[0,50],
			# 	opposite=True, #
			# 	# box = [[-.8,1.],[.8,2.5]], # surround
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround",
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=True, 
			# 	# percentile=True,
			# 	box = [[-.5,-.5],[.5,.5]], # center
			# 	# box = [[-.25,-.25],[.25,.25]], # center overlapping
			# 	# box = [[-.5,.0],[.5,.8]], # mixed surround (more likely to be influenced by the recorded thalamus)
			# 	# box = [[-.5,.5],[.5,1.]], # strict surround
			# 	# box = [[-0.1,.6],[.3,1.]], # surround
			# 	# box = [[-.5,.0],[.5,.5]], # surround nonoverlapping
			# 	# box = [[-.5,.2],[.2,.5]], # surround nonoverlapping
			# 	addon = "center",
			# )
			# trial_averaged_conductance_timecourse( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	#       0      1     2     3     4     5     6     7     8     9
			# 	ticks=[0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
			# 	ylim=[0,30],
			# 	box = [[-.5,-.5],[.5,.5]], # center
			# 	# box = [[-.25,-.25],[.25,.25]], # center overlapping
			# 	addon = "center",
			# 	# box = [[-.5,.0],[.5,.8]], # mixed surround (more likely to be influenced by the recorded thalamus)
			# 	# box = [[-.5,.5],[.5,1.]], # strict surround
			# 	# box = [[-0.1,.6],[.3,1.]], # surround
			# )

			# SpikeTriggeredAverage(
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	opposite=False, # ISO
			# 	radius = [.0,.7], # center
			# 	addon = addon + "_center",
			# 	color = color,
			# )
			SpikeTriggeredAverage(
				sheet=s, 
				folder=f, 
				stimulus='DriftingSinusoidalGratingDisk',
				parameter="radius",
				ylim=[0,50],
				opposite=True, # ORTHO
				radius = [.0,.7], # center
				addon = addon + "_center",
				color = color,
			)
			# SpikeTriggeredAverage(
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	opposite=False, # ISO
			# 	# box = [[-.8,1.],[.8,2.5]], # surround
			# 	radius = [1.,1.8], # surround
			# 	addon = addon + "_surround",
			# 	color = color,
			# )
			SpikeTriggeredAverage(
				sheet=s, 
				folder=f, 
				stimulus='DriftingSinusoidalGratingDisk',
				parameter="radius",
				ylim=[0,50],
				opposite=True, # ORTHO
				# box = [[-.8,1.],[.8,2.5]], # surround
				radius = [1.,1.8], # surround
				addon = addon + "_surround",
				color = color,
			)

			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	radius = [.0,.7], # center
			# 	addon = "center_same_",
			# 	opposite=False, # SAME
			# 	nulltime=True, # True for spontaneous activity before stimulus
			# 	# nulltime=False, # True for spontaneous activity before stimulus
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_same_",
			# 	opposite=False, # SAME
			# 	# nulltime=True, # True for spontaneous activity before stimulus
			# 	nulltime=False, # True for spontaneous activity before stimulus
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	radius = [.0,.7], # center
			# 	addon = "center_opposite_",
			# 	opposite=True, # OPPOSITE
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_opposite_",
			# 	opposite=True, # OPPOSITE
			# )

			# jpsth( # Thalamus 
			# 	sheet1='X_ON', 
			# 	sheet2='X_ON', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	box1=[[-.2,-.2],[.2,.2]], # center
			# 	box2=[[-.2,.3],[.2,.7]], # surround
			# 	addon="center_vs_surround_"+addon,
			# 	color=color,
			# 	trials=trials
			# )
			# jpsth( # Thalamus 
			# 	sheet1='X_OFF', 
			# 	sheet2='X_OFF', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	box1=[[-.2,-.2],[.2,.2]], # center
			# 	box2=[[-.2,.3],[.2,.7]], # surround
			# 	addon="center_vs_surround_"+addon,
			# 	color=color,
			# 	trials=trials
			# )
			# jpsth( # CORTICO-CORTICAL 
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	box1=[[-.5,-.5],[.5,.5]], # center
			# 	box2=[[-.5,.5],[.5,1.]], # surround
			# 	addon="center_vs_surround_"+addon,
			# 	color=color,
			# 	trials=trials
			# )
			# jpsth( # CORTICO-CORTICAL 
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	box1=[[-.5,-.5],[.5,.5]], # center
			# 	box2=[[-.5,-.5],[.5,.5]], # center
			# 	addon="center_vs_center_opposite_"+addon,
			# 	color=color,
			# 	trials=trials
			# )
			# jpsth( # CORTICO-CORTICAL 
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	box1=[[-.5,.5],[.5,1.]], # surround
			# 	box2=[[-.5,.5],[.5,1.]], # surround
			# 	addon="surround_vs_surround_opposite_"+addon,
			# 	color=color,
			# 	trials=trials
			# )

			# spectrum(
			# 	sheet='X_OFF', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,-.2],[.2,.2]], # center
			# 	radius = [.0,.2], # center
			# 	addon = "center_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='X_OFF', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,.3],[.2,.7]], # surround
			# 	radius = [.5,.8], # surround
			# 	addon = "surround_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='X_ON', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,-.2],[.2,.2]], # center
			# 	radius = [.0,.2], # center
			# 	addon = "center_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='X_ON', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,.3],[.2,.7]], # surround
			# 	radius = [.5,.8], # surround
			# 	addon = "surround_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,-.2],[.2,.2]], # center
			# 	radius = [.0,.2], # center
			# 	addon = "center_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box = [[-.2,.3],[.2,.7]], # surround
			# 	radius = [.5,.8], # surround
			# 	addon = "surround_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box=[[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center_same_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box=[[-.5,.5],[.5,1.]], # surround
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_same_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box=[[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = "center_opposite_"+addon,
			# 	preferred=False, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box=[[-.5,.5],[.5,1.]], # surround
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_opposite_"+addon,
			# 	preferred=False, # 
			# 	color = color,
			# )

			# correlation( 
			# 	sheet1='X_OFF', 
			# 	sheet2='X_OFF', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box1 = [[-.2,-.2],[.2,.2]], # center thalamus
			# 	# box2 = [[-.3,.3],[.3,.7]], # surround thalamus
			# 	radius1 = [.0,.2], # center
			# 	radius2 = [.5,.8], # surround
			# 	addon="center2surround_"+addon,
			# 	color=color,
			# )
			# correlation( 
			# 	sheet1='X_ON', 
			# 	sheet2='X_ON', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	# box1 = [[-.2,-.2],[.2,.2]], # center thalamus
			# 	# box2 = [[-.3,.3],[.3,.7]], # surround thalamus
			# 	radius1 = [.0,.2], # center
			# 	radius2 = [.5,.8], # surround
			# 	addon="center2surround_"+addon,
			# 	color=color,
			# )
			# correlation( # CORTICO-CORTICAL 
			# 	sheet1=s, 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [1.,1.8], # surround
			# 	preferred1=True, # SAME
			# 	preferred2=True, # SAME
			# 	addon="center2surround_iso_"+addon,
			# 	sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	color=color
			# )
			# correlation( # CORTICO-CORTICAL 
			# 	sheet1=s, 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [1.,1.8], # surround
			# 	preferred1=False, # OPPOSITE
			# 	preferred2=False, # OPPOSITE
			# 	addon="center2surround_ortho_"+addon,
			# 	sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711],
			# 	color=color
			# )


			# isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	# box = [[-.2,-.2],[.2,.2]], # center
			# 	# box = [[-.3,-.3],[.3,.3]], # center thalamus
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER V1
			# 	radius = [.0,.7], # center
			# 	opposite=False, # ISO
			# 	addon = "center_iso_"+addon,
			# 	ylim=[0.,1.]
			# )
			# isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	# box = [[-.3,.3],[.3,.7]], # surround thalamus
			# 	# box = [[-.2,.3],[.2,.7]], # surround
			# 	# box = [[-.5,.5],[.5,1.5]], # SURROUND V1
			# 	radius = [1.,1.8], # surround
			# 	opposite=False, # ISO
			# 	addon = "surround_iso_"+addon,
			# 	ylim=[0.,1.]
			# )
			# isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	# box = [[-.2,-.2],[.2,.2]], # center
			# 	# box = [[-.3,-.3],[.3,.3]], # center thalamus
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER V1
			# 	radius = [.0,.7], # center
			# 	opposite=True, # ORTHO
			# 	addon = "center_ortho_"+addon,
			# 	ylim=[0.,1.]
			# )
			# isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	# box = [[-.3,.3],[.3,.7]], # surround thalamus
			# 	# box = [[-.2,.3],[.2,.7]], # surround
			# 	# box = [[-.5,.5],[.5,1.5]], # SURROUND V1
			# 	radius = [1.,1.8], # surround
			# 	opposite=True, # ORTHO
			# 	addon = "surround_ortho_"+addon,
			# 	ylim=[0.,1.]
			# )

			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	box = [[-3.5,-3.5],[3.5,3.5]], # ALL
			# 	opposite=False, # SAME
			# 	addon = "all_same",
			# )
			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	opposite=False, # SAME
			# 	addon = "center_same",
			# )
			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	opposite=True, # OPPOSITE
			# 	addon = "center_opposite",
			# )
			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	color=color,
			# 	box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	opposite=False, # SAME
			# 	addon = "surround_same",
			# )

			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	       # 1K  56    32      24    16    8   Hz
			# 	periods=[1., 17.8, 31.25,  41.6, 62.5, 125.], 
			# 	color=color,
			# 	# compute_isi=False,
			# 	# compute_vectorstrength=True,
			# 	compute_isi=True,
			# 	compute_vectorstrength=False,
			# 	box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	opposite=True, # OPPOSITE
			# 	addon = "center_opposite",
			# 	fit=fit,
			# )
			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	       # 1K  56    32      24    16    8   Hz
			# 	periods=[1., 17.8, 31.25,  41.6, 62.5, 125.], 
			# 	color=color,
			# 	# compute_isi=False,
			# 	# compute_vectorstrength=True,
			# 	compute_isi=True,
			# 	compute_vectorstrength=False,
			# 	box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	opposite=False, # SAME
			# 	addon = "surround_same",
			# 	fit=fit,
			# )
			# windowed_isi( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	       # 1K  56    32      24    16    8   Hz
			# 	periods=[1., 17.8, 31.25,  41.6, 62.5, 125.], 
			# 	color=color,
			# 	# compute_isi=False,
			# 	# compute_vectorstrength=True,
			# 	compute_isi=True,
			# 	compute_vectorstrength=False,
			# 	box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	opposite=True, # OPPOSITE
			# 	addon = "surround_opposite",
			# 	fit=fit,
			# )

			# activity_ratio( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	addon = addon,
			# 	color = color, 
			# 	arborization_diameter = arborization *60 # 
			# )


			# # #ORIENTATION
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False, #True,
			# 	ylim=[0,50],
			# 	# opposite=False, # to select cortical cells with SAME orientation preference
			# 	opposite=True, # to select cortical cells with OPPOSITE orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center - TOTAL
			# 	# box = [[-.25,-.25],[.25,.25]], # center
			# 	# box = [[-.4,.15],[.2,.4]], # large center SAME
			# 	# box = [[-.4,.0],[.2,.4]], # large center SAME
			# 	# box = [[-.5,-.5],[.0,.0]], # center OPPOSITE
			# 	# box = [[.2,-.15],[.5,.15]], # larger center OPPOSITE
			# 	# addon = "center",
			# 	# box = [[-.5,.5],[.5,1.5]], # surround - TOTAL
			# 	# box = [[-.4,.5],[.1,1.]], # far up surround SAME
			# 	box = [[-.5,1.],[.0,1.5]], # far up surround OPPOSITE
			# 	addon = "surround",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
			# 	# data_curve=False,
			# )
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	ylim=[0,50],
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# )
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	ylim=[0,50],
			# 	opposite=True, # to select cortical cells with OPPOSITE orientation preference
			# 	radius = [.0,.7], # center
			# 	addon = "center",
			# )
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	ylim=[0,100],
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	radius = [1.,4.], # surround
			# 	addon = "surround",
			# )
			# trial_averaged_tuning_curve_errorbar( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Orientation", 
			# 	ylabel="firing rate (sp/s)", 
			# 	color=color, 
			# 	useXlog=False, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	ylim=[0,50],
			# 	opposite=True, #
			# 	radius = [1.,4.], # surround
			# 	addon = "surround",
			# )
			# orientation_bias_barplot( 
			# 	sheet=s, 
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
			# 	# percentage=True,
			# )
			# orientation_selectivity_index_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='orientation',
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="exp",
			# 	ylabel="Orientation selectivity index",
			# 	color=color,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/NaitoOkamotoSadakaneShimegiOsakiHaraKimuraIshikawaSuematsuSato2013_2b.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/NaitoSadakaneOkamotoSato2007_1A.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/SuematsuNaitoSato2012_4B.csv",
			# 	color_data=color_data,
			# )
			# orientation_bias_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='orientation',
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="exp",
			# 	ylabel="Orientation bias",
			# 	closed=False,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_open.csv",
			# 	# closed=True,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_closed.csv",
			# 	# percentage=True,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter='orientation',
			# 	percentile=False,
			# 	ylim=[0,15],
			# 	box = [[-.25,-.25],[.25,.25]], # center
			# 	dashed=True,
			# 	addon = "open",
			# 	# dashed=False,
			# 	# addon = "closed",
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	addon = "center_same_0.1",
			# 	opposite=False, # SAME
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	addon = "center_opposite_0.1",
			# 	opposite=True, # OPPOSITE
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	addon = "surround_same_0.1",
			# 	opposite=False, # SAME
			# )
			# variability( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	box = [[-.5,.5],[.5,1.5]], # SURROUND
			# 	addon = "surround_opposite_0.1",
			# 	opposite=True, # OPPOSITE
			# )

			# correlation( # CORTICO-CORTICAL
			# 	sheet1=s, 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	box1=[[-.6,-.6],[.6,.6]], # center
			# 	# addon="center_feedforward",
			# 	# addon="center_closed",
			# 	box2 = [[-0.1,.6],[.3,1.]], # surround
			# 	# addon="surround_feedforward",
			# 	addon="center_vs_surround",
			# )
			# correlation( 
			# 	sheet1='X_ON', 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	preferred2=True,
			# 	addon="preferred_closed",
			# 	# addon="preferred_feedforward",
			# )
			# correlation( 
			# 	sheet1='X_OFF', 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	preferred2=True,
			# 	addon="preferred_closed",
			# 	# addon="preferred_feedforward",
			# )
			# correlation( 
			# 	sheet1='X_ON', 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	preferred2=False,
			# 	addon="opposite_closed",
			# 	# addon="opposite_feedforward",
			# )
			# correlation( 
			# 	sheet1='X_OFF', 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	preferred2=False,
			# 	addon="opposite_closed",
			# 	# addon="opposite_feedforward",
			# )

			# phase_synchrony( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	stimulus_parameter='orientation',
			# 	       # 1K  56    32      24    16    8   Hz
			# 	periods=[1., 17.8, 31.25,  41.6, 62.5, 125.], 
			# 	addon=addon, 
			# 	color=color,
			# 	box = [[-3.5,-3.5],[3.5,3.5]], # ALL
			# 	opposite=False, # SAME
			# 	compute_isi=True,
			# 	compute_vectorstrength=False,
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
				# 	ylabel="PGN-off",
				# 	withRegression=False,
				# 	withCorrCoef=False,
				# 	withCentroid=True,
				# 	data_marker = "o",
				# 	# LOW-PASS
				# 	stimulus_band=1,
				# 	withPassIndex=True,
				# 	reference_band=3,
				# 	data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_closed.csv",
				# 	data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_open.csv",
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

				# TEMPORAL
				# pairwise_response_reduction( 
				# 	sheet=s, 
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
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/c50_closed_cells.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/c50_open_cells.csv",
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

	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_closed.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_0.1_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_closed.csv",

    ]

closed_ON_files = [ 
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_ON_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.12_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.19_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.28_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.42_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.63_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.94_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.41_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_2.10_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_3.15_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_4.71_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.12_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.19_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.28_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.42_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.63_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_0.94_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_1.41_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_2.10_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_3.15_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_ON_radius_4.71_surround_closed.csv",

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
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_X_OFF_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.12_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.19_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.28_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.42_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.63_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.94_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.41_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_2.10_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_3.15_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_4.71_center_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.12_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.19_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.28_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.42_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.63_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_0.94_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_1.41_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_2.10_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_3.15_surround_closed.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_closed_____/TrialAveragedMeanVariance_X_OFF_radius_4.71_surround_closed.csv",

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
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_0.1_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_feedforward.csv",

	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_feedforward.csv",

	]

open_ON_files = [ 
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.12_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.19_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.28_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.42_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.63_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.94_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.41_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_2.10_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_3.15_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_4.71_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.12_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.19_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.28_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.42_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.63_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_0.94_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_1.41_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_2.10_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_3.15_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_ON_radius_4.71_center_feedforward.csv",

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
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.125.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.163713961769.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.214418090225.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.280825880206.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.367800939327.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.481713191357.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.630905399949.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.826304180217.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.08222024776.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.41739651414.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.85638078982.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_2.43132362923.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_3.18433298947.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_4.17055815439.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_5.46222878595.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.12_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.19_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.28_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.42_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.63_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.94_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.41_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_2.10_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_3.15_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_4.71_surround_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.12_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.19_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.28_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.42_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.63_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_0.94_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_1.41_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_2.10_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_3.15_center_feedforward.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____/TrialAveragedMeanVariance_X_OFF_radius_4.71_center_feedforward.csv",

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

# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____", closed_files, open_files, [-3., 15.] ) # cortex
# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____", closed_ON_files, open_ON_files, [-2., 5.], closed_OFF_files, open_OFF_files ) # LGN

# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____", closed_files, open_files, [-5, 5], averaged=True )
# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____", closed_ON_files, open_ON_files, [-0.5, 2.], closed_OFF_files, open_OFF_files ) # LGN

# cx2 places
# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____", closed_files, open_files, [-.2, 1.5], averaged=True ) # cortex
# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_cx2_feedforward_____", closed_ON_files, open_ON_files, [0., 2.], closed_OFF_files, open_OFF_files, averaged=True ) # LGN









# files_center_closed = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_same_0.1_closed.csv",
# ]
# files_center_opposite_closed = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_opposite_0.1_closed.csv",
# ]
# files_surround_closed = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_same_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_same_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_same_0.1_closed.csv",
# ]
# files_surround_opposite_closed = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_0.1_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_closed.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_0.1_closed.csv",
# 	"Thalamocortical_size_closed/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_0.1_closed.csv",
# ]
# files_surround_feedforward = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_same_0.1_feedforward.csv",
# ]
# files_surround_opposite_feedforward = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_surround_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_surround_opposite_0.1_feedforward.csv",
# ]
# files_center_feedforward = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_same_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_same_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_same_0.1_feedforward.csv",
# ]
# files_center_opposite_feedforward = [
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_opposite_0.1_feedforward.csv",
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.19_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.42_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.94_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.41_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.10_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.15_center_opposite_0.1_feedforward.csv",
# 	"Thalamocortical_size_feedforward/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.71_center_opposite_0.1_feedforward.csv",
# ]





files_surround_closed = [
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_surround_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_surround_same_0.1_closed.csv",

	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_surround_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_surround_same__closed.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_surround_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_surround_same__closed_nulltime.csv",
]

# files_surround_opposite_closed = [
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_surround_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_surround_opposite_0.1_closed.csv",
# ]

files_center_closed = [
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_center_same_0.1_closed.csv",
	# "ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_center_same_0.1_closed.csv",

	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_center_same__closed.csv",
	"Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_center_same__closed.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_center_same__closed_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_center_same__closed_nulltime.csv",
]

# files_center_opposite_closed = [
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_center_opposite_0.1_closed.csv",
# 	"ThalamoCorticalModel_data_orientation_closed_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_center_opposite_0.1_closed.csv",
# ]



files_surround_feedforward = [
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_surround_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_surround_same_0.1_feedforward.csv",

	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_surround_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_surround_same__feedforward.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_surround_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_surround_same__feedforward_nulltime.csv",
]

# files_surround_opposite_feedforward = [
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_surround_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_surround_opposite_0.1_feedforward.csv",
# ]


files_center_feedforward = [
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_center_same_0.1_feedforward.csv",
	# "ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_center_same_0.1_feedforward.csv",

	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_center_same__feedforward.csv",
	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_center_same__feedforward.csv",

	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.12_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.16_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.21_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.28_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.37_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.48_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.63_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_0.83_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.08_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.42_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_1.86_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_2.43_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_3.18_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_4.17_center_same__feedforward_nulltime.csv",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_radius_5.46_center_same__feedforward_nulltime.csv",
]


# files_center_opposite_feedforward = [
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.00_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.31_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.63_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_0.94_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.26_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.57_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_1.88_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.20_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.51_center_opposite_0.1_feedforward.csv",
# 	"ThalamoCorticalModel_data_orientation_feedforward_____2/TrialAveragedMeanVariance_V1_Exc_L4_orientation_2.83_center_opposite_0.1_feedforward.csv",
# ]

# files_orientation_sponatenous_feedforward = [
# 	"ThalamoCorticalModel_data_orientation_feedforward_____/TrialAveragedMeanVariance_V1_Exc_L4_duration_1029.00_feedforward.csv",
# ]

# files_orientation_sponatenous_closed = [
# 	"ThalamoCorticalModel_data_orientation_closed_____/TrialAveragedMeanVariance_V1_Exc_L4_duration_1029.00_closed.csv",
# ]


# fano_comparison_timecourse( 
# 	sheets, 

# 	# "Thalamocortical_size_feedforward", 
# 	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large", 
# 	# "ThalamoCorticalModel_data_orientation_feedforward_____2", 
# 	"Deliverable/ThalamoCorticalModel_data_size_feedforward_____",

# 	# files_orientation_sponatenous_closed,
# 	# files_orientation_sponatenous_feedforward,

# 	# files_center_closed,
# 	# files_center_feedforward,
# 	# # addon="center_nulltime",
# 	# addon="center",

# 	files_surround_closed,
# 	files_surround_feedforward,
# 	# addon="surround_nulltime",
# 	addon="surround",

# 	# files_center_opposite_closed,
# 	# files_center_opposite_feedforward,
# 	# addon="center_opposite",

# 	# files_surround_opposite_closed,
# 	# files_surround_opposite_feedforward,
# 	# addon="surround_opposite",

# 	# averaged=False, 
# 	averaged=True, 

# 	# ORIENTATION
# 	# sliced=[0,2], # center, optimal

# 	# SIZE
# 	# sliced=[3,5], # center, 
# 	# sliced=[5,8], # only the peak
# 	# sliced=[2,None], # all except the noisy beginning
# 	sliced=[0,None], # all
# 	# sliced=[5,None], # only large sizes

# 	ylim=[.0, 2.], 
#  ) # cortex

# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large", closed_files, open_files, [.0, 1.5], averaged=True ) # cortex
# fano_comparison_timecourse( sheets, "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large", closed_ON_files, open_ON_files, [0., 2.], closed_OFF_files, open_OFF_files, averaged=True ) # LGN


















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


# Raster + Histogram
# RasterPlot(
# 	param_filter_query(data_store,st_name=stimulus,sheet_name=sheet),
# 	ParameterSet({'sheet_name' : sheet, 'neurons' : spike_ids, 'trial_averaged_histogram': True, 'spontaneous' : False}),
# 	fig_param={'dpi' : 100,'figsize': (200,100)},
# 	plot_file_name=folder+"/HistRaster_"+sheet+"_"+stimulus+".png"
# ).plot({'SpikeRasterPlot.group_trials':True})
