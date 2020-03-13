#!/usr/bin/python
# -*- coding: latin-1 -*-

# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import os
import ast
import re
import glob
import json
import inspect
import math

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




def select_ids_by_position(positions, sheet_ids, radius=[0,0], box=[], reverse=False, origin=[[0.],[0.],[0.]]):
	selected_ids = []
	distances = []
	min_radius = radius[0]
	max_radius = radius[1]

	for i in sheet_ids:
		a = numpy.array((positions[0][i],positions[1][i],positions[2][i]))
		# print a

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
			l = numpy.linalg.norm(origin-a)

			# print "distance",l
			# print abs(l),">",min_radius,"and", abs(l), "<", max_radius
			if abs(l)>min_radius and abs(l)<max_radius:
				# print abs(l),">",min_radius,"and", abs(l), "<", max_radius, "TAKEN"
				# print i
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
	if not len(neurons)>0:
		spike_ids = param_filter_query(datastore, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = datastore.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		neurons = datastore.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)
	print "SpikeCount neurons", len(neurons)

	SpikeCount( 
		param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
		ParameterSet({'bin_length':bin, 'neurons':list(neurons), 'null':False}) 
	).analyse()
	# datastore.save()
	TrialMean(
		param_filter_query( datastore, name='AnalogSignalList', analysis_algorithm='SpikeCount' ),
		ParameterSet({'vm':False, 'cond_exc':False, 'cond_inh':False})
	).analyse()

	dsvTM = param_filter_query( datastore, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialMean' )
	# dsvTM.print_content(full_recordings=False)
	pnvsTM = [ dsvTM.get_analysis_result() ]
	# print pnvsTM
	# get stimuli from PerNeuronValues
	st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvsTM[-1]]

	asl_id = numpy.array([z.get_asl_by_id(neurons) for z in pnvsTM[-1]])
	print asl_id.shape
	# Example:
	# (8, 133, 1, 1029)
	# 8 stimuli
	# 133 cells
	# 1 wrapper
	# 1029 bins

	dic = colapse_to_dictionary([z.get_asl_by_id(neurons) for z in pnvsTM[-1]], st, stimulus_parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))

	stimuli = dic.values()[0][0]
	means = asl_id.mean(axis=2) # mean of
	print means.shape

	return means, stimuli

	# if not spikecount:
	# 	TrialAveragedFiringRateCutout( 
	# 		param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
	# 		ParameterSet({}) 
	# 	).analyse(start=start, end=end)
	# else:
	#	SpikeCount( 
	#		param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
	#		ParameterSet({'bin_length':bin, 'neurons':list(neurons), 'null':False}) 
	#	).analyse()

	# mean_rates = []
	# stimuli = []
	# if not isinstance(sheet, list):
	# 	sheet = [sheet]
	# for sh in sheet:
	# 	print sh
	# 	dsv = param_filter_query( datastore, identifier='PerNeuronValue', sheet_name=sh, st_name=stimulus )
	# 	# dsv.print_content(full_recordings=False)
	# 	pnvs = [ dsv.get_analysis_result() ]
	# 	# print pnvs
	# 	# get stimuli from PerNeuronValues
	# 	st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs[-1]]

	# 	dic = colapse_to_dictionary([z.get_value_by_id(neurons) for z in pnvs[-1]], st, stimulus_parameter)
	# 	for k in dic:
	# 		(b, a) = dic[k]
	# 		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
	# 		dic[k] = (par,numpy.array(val))
	# 	print dic

	# 	if len(mean_rates)>1:
	# 		# print mean_rates.shape, dic.values()[0][1].shape
	# 		mean_rates = numpy.append( mean_rates, dic.values()[0][1], axis=1 )
	# 	else:
	# 		# print dic
	# 		mean_rates = dic.values()[0][1]

	# 	stimuli = dic.values()[0][0]
	# 	neurons = [] # reset, if we are in a loop we don't want the old neurons id to be still present
	# 	print mean_rates.shape

	# return mean_rates, stimuli




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




def size_tuning_index( sheet, folder_full, stimulus, parameter, box=None, radius=None, csvfile=None ):
	print inspect.stack()[0][3]
	print folder_full
	folder_nums = re.findall(r'\d+', folder_full)
	print folder_nums
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)

	# GET RECORDINGS

	# get the list of all recorded neurons in X_ON
	# Full
	spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons (closed):", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store_full.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons_full = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=ids1)


	neurons = neurons_full
	num_cells = len(neurons_full)

	print "neurons:", num_cells, neurons

	assert num_cells > 0 , "ERROR: the number of recorded neurons is 0"

	# compute firing rates
	# required shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)
	dstims = {}

	# Closed
	TrialAveragedFiringRate(
		param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv1 = param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls1 = dsv1.get_analysis_result( sheet_name=sheet )
	# for key, asl in sorted( asls1, key=lambda x: x.get(parameter) ):
	closed_dict = {}
	for asl in asls1:
		# print asl.stimulus_id
		stim = eval(asl.stimulus_id).get(parameter)
		dstims[stim] = stim
		closed_dict[stim] = asl.get_value_by_id(neurons)
	# print closed_dict
	all_closed_values = numpy.array([closed_dict[k] for k in sorted(closed_dict)])
	# print all_closed_values

	stims = sorted(dstims)

	# END-INHIBITION as in MurphySillito1987 and AlittoUsrey2008:
	# "The responses of the cell with corticofugal feedback are totally suppressed at bar lenghts of 2deg and above, 
	#  and those of cell lacking feedback are reduced up to 40% at bar lenghts of 8deg and above."
	# 1. find the peak response at large sizes
	peaks_closed = numpy.amax(all_closed_values, axis=0) # as in AlittoUsrey2008
	# 2. compute average response at large sizes
	plateaus_closed = numpy.mean( all_closed_values[5:], axis=0) 
	# 3. compute the difference from peak 
	ends_closed = (peaks_closed-plateaus_closed)/peaks_closed # as in MurphySillito1987
	print "closed",ends_closed

	if csvfile:
		csvrow = ",".join(folder_nums)+",("+ str(numpy.mean(ends_closed))+ "), "
		print csvrow
		csvfile.write( csvrow )




def size_tuning_comparison( sheet, folder_full, folder_inactive, stimulus, parameter, box=None, radius=None, csvfile=None, plotAll=False ):
	print inspect.stack()[0][3]
	print folder_full
	folder_nums = re.findall(r'\d+', folder_full)
	print folder_nums
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	# rings analysis
	rowplots = 0

	# GET RECORDINGS

	# get the list of all recorded neurons in X_ON
	# Full
	spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons (closed):", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store_full.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons_full = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	# Inactivated
	spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	if radius or box:
		sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids2)
		positions2 = data_store_inac.get_neuron_postions()[sheet]
		if box:
			ids2 = select_ids_by_position(positions2, sheet_ids2, box=box)
		if radius:
			ids2 = select_ids_by_position(positions2, sheet_ids2, radius=radius)
		neurons_inac = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=ids2)

	# Intersection of full and inac
	if set(neurons_full)==set(neurons_inac):
		neurons = neurons_full
		num_cells = len(neurons_full)
	else:
		neurons = numpy.intersect1d(neurons_full, neurons_inac)
		num_cells = len(neurons)

	if len(neurons) > rowplots:
		rowplots = num_cells

	print "neurons_full:", len(neurons_full), neurons_full
	print "neurons_inac:", len(neurons_inac), neurons_inac
	print "neurons:", num_cells, neurons

	assert num_cells > 0 , "ERROR: the number of recorded neurons is 0"

	# compute firing rates
	# required shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)
	dstims = {}

	# Closed
	TrialAveragedFiringRate(
		param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv1 = param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls1 = dsv1.get_analysis_result( sheet_name=sheet )
	# for key, asl in sorted( asls1, key=lambda x: x.get(parameter) ):
	closed_dict = {}
	for asl in asls1:
		# print asl.stimulus_id
		stim = eval(asl.stimulus_id).get(parameter)
		dstims[stim] = stim
		closed_dict[stim] = asl.get_value_by_id(neurons)
	# print closed_dict
	all_closed_values = numpy.array([closed_dict[k] for k in sorted(closed_dict)])
	# print all_closed_values

	stims = sorted(dstims)

	# Open
	TrialAveragedFiringRate(
		param_filter_query(data_store_inac, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv2 = param_filter_query(data_store_inac, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls2 = dsv2.get_analysis_result( sheet_name=sheet )
	open_dict = {}
	for asl in asls2:
		open_dict[eval(asl.stimulus_id).get(parameter)] = asl.get_value_by_id(neurons)
	# print open_dict
	all_open_values = numpy.array([open_dict[k] for k in sorted(open_dict)])
	# print all_open_values

	# Population histogram
	diff_full_inac = []
	sem_full_inac = []

	# -------------------------------------
	# DIFFERENCE BETWEEN INACTIVATED AND CONTROL
	# We want to have a summary measure of the population of cells with and without inactivation.
	# The null-hypothesis is that the inactivation does not change the activity of cells.
	# A different result in the inactivated-cortex condition will tell us that the inactivation DOES something.
	# The null-hypothesis is the result obtained in the intact system.
	# If in the inactivated cortex something changes, then we label that cell for being added to the count of changed results

	# 1. MASK IN ONLY CHANGING UNITS

	# 1.1 Search for the units that are NOT changing (within a certain absolute tolerance)
	unchanged_units = numpy.isclose(all_closed_values, all_open_values, rtol=0., atol=30.) # 20 spikes/s
	print "unchanged units:"
	print unchanged_units

	# 1.2 Reverse them into those that are changing
	changed_units_mask = numpy.invert( unchanged_units )

	# 1.3 Get indexes for printing
	changed_units = numpy.nonzero( changed_units_mask )
	changing_idxs = zip(changed_units[0], changed_units[1])
	print "changing units (unchanged inverted)"
	print changing_idxs

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

	# 3. Calculate difference (inac - control)
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
	perc_smaller = sign_smaller * (smaller/norm) *100
	perc_equal = sign_equal * (equal/norm) *100
	perc_larger = sign_larger * (larger/norm) *100
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
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_only_bars.png", dpi=300, transparent=True )
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_only_bars.svg", dpi=300, transparent=True )
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
		# each cell couple 
		axes[0,1].set_ylabel("Response (spikes/sec)", fontsize=10)
		for j,nid in enumerate(neurons):
			# print col,j,nid
			y_full = all_closed_values[:,j]
			y_inac = all_open_values[:,j]
			axes[0,j+1].plot(stims, y_full, linewidth=2, color='b')
			axes[0,j+1].plot(stims, y_inac, linewidth=2, color='r')
			axes[0,j+1].set_title(str(nid), fontsize=10)
			axes[0,j+1].set_xscale("log")

		fig.subplots_adjust(hspace=0.4)
		# fig.suptitle("All recorded cells grouped by circular distance", size='xx-large')
		fig.text(0.5, 0.04, 'cells', ha='center', va='center')
		fig.text(0.06, 0.5, 'ranges', ha='center', va='center', rotation='vertical')
		for ax in axes.flatten():
			ax.set_ylim([0,60])
			ax.set_xticks(stims)
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
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+".png", dpi=150, transparent=True )
		plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+".svg", dpi=150, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()




def mean_confidence_interval(data, confidence=0.95):
	print inspect.stack()[0][3]
	import scipy.stats
	a = 1.0*numpy.array(data)
	n = len(a)
	m, se = numpy.mean(a, axis=0), scipy.stats.sem(a, axis=0)
	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
	return m, m-h, m+h




def activity_ratio( sheet, folder, stimulus, stimulus_parameter, arborization_diameter=10, addon="", color="black" ):
	print inspect.stack()[0][3]
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
	plt.savefig( folder+"/Activity_ratio_"+str(sheet)+"_"+stimulus_parameter+"_"+addon+".svg", dpi=200, transparent=True )
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
		plt.savefig( folder+"/Activity_Map_"+str(sheet)+"_"+stimulus_parameter+"_"+key+"_"+addon+".svg", dpi=200, transparent=True )
		plt.close()
		gc.collect()




def spectrum( sheet, folder, stimulus, stimulus_parameter, addon="", color="black", box=False, radius=None, preferred=True, ylim=[0.,200.] ):
	print inspect.stack()[0][3]
	import ast
	import scipy.signal as signal
	from matplotlib.ticker import NullFormatter
	from scipy import signal

	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(spike_ids)

	if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
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
	dt = 0.002 # sampling interval, PyNN default time is in ms, 'bin_length': 2.0
	fs = 1.0/dt # 
	sr = fs**2
	plt.figure()
	for stim, ps in psths.items():
		print stim, ps

		# spectrum
		for psth in ps:
			freq, power = signal.welch(ps, sr, window='hamming')
			plt.semilogx( freq, power, linewidth=1, color=color, linestyle='-' )

	plt.ylabel("Power (μV2)")
	plt.xlabel("Frequency (Hz)")
	plt.savefig( folder+"/spectrums_"+str(sheet)+"_"+stim+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()


	# 	power = numpy.mean(spectra, axis=0)
		
	# 	powers.append( power )

	# 	# if float(stim) > 4.:
	# 	# 	plt.figure()
	# 	# 	idx = numpy.argsort(freqs) # last one is as all the others
	# 	# 	plt.plot( freqs[idx], power[idx], linewidth=3, color=color, linestyle='-' )
	# 	# 	plt.ylim([0., 30000000.])
	# 	# 	plt.xlim([0., 150.])
	# 	# 	plt.tight_layout()
	# 	# 	plt.savefig( folder+"/spectrum_"+str(sheet)+"_"+stim+"_"+addon+".svg", dpi=300, transparent=True )
	# 	# 	plt.close()
	# 	# 	gc.collect()

	# powers = numpy.array(powers)
	# # print powers.shape

	# powers = numpy.mean(powers, axis=0)
	# idx = numpy.argsort(freqs) # last one is as all the others
	# plt.figure()
	# idx = numpy.argsort(freqs) # last one is as all the others
	# plt.plot( freqs[idx], powers[idx], linewidth=3, color=color, linestyle='-' )
	# plt.ylim([0., 30000000.])
	# plt.xlim([0., 150.])
	# plt.tight_layout()
	# # plt.savefig( folder+"/avg_spectrum_"+str(sheet)+"_"+stim+"_"+addon+".png", dpi=300, transparent=True )
	# plt.savefig( folder+"/avg_spectrum_"+str(sheet)+"_"+stim+"_"+addon+".svg", dpi=300, transparent=True )
	# plt.close()
	# gc.collect()




def isi( sheet, folder, stimulus, stimulus_parameter, addon="", color="black", box=False, radius=False, opposite=False, ylim=[0.,5.] ):
	print inspect.stack()[0][3]
	import ast
	from scipy.stats import norm
	matplotlib.rcParams.update({'font.size':22})

	print "folder: ",folder
	print "sheet: ",sheet
	print addon
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )
	dsv = queries.param_filter_query(data_store, sheet_name=sheet, st_name=stimulus)
	segments = dsv.get_segments()
	spike_ids = dsv.get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons :", len(spike_ids)

	if radius or box:
		if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
			NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
			l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
			if opposite:
				l4_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in spike_ids]) < .1)[0]]
			else:
				l4_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0.,numpy.pi) for i in spike_ids]) < .1)[0]]
			print "# of V1 cells range having orientation 0:", len(l4_many)
			spike_ids = list(l4_many)

		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons :", len(spike_ids)
	if len(spike_ids) < 1:
		return

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
			# center iso inh: 1.082, 4.171
			# center ortho inh: 1.082, 4.171
			if stim > 0.6: # 4.:
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
	# plt.tight_layout()
	plt.ylim(ylim)
	plt.plot( x, hisi, linewidth=3, color=color) 
	# plt.savefig( folder+"/ISI_"+str(sheet)+"_"+stimulus_parameter+"_x"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/ISI_"+str(sheet)+"_"+stimulus_parameter+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()




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
	plt.savefig( folder+"/SpontaneousActivity_"+sheet+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def correlation( sheet1, sheet2, folder, stimulus, stimulus_parameter, box1=None, box2=None, radius1=None, radius2=None, preferred1=True, preferred2=True, sizes=[], addon="", color="black" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet1: ",sheet1
	print "sheet2: ",sheet2
	import ast
	from matplotlib.ticker import NullFormatter
	from scipy import signal

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}), replace=True )

	spike_ids1 = param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 1:", len(spike_ids1)
	spike_ids2 = param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons 2:", len(spike_ids2)

	if sheet1=='V1_Exc_L4' or sheet1=='V1_Inh_L4':
		spike_ids1 = select_by_orientation(data_store, sheet1, spike_ids1, preferred=preferred1)

	if sheet2=='V1_Exc_L4' or sheet2=='V1_Inh_L4':
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
		psth = numpy.mean(psth, axis=0) # mean over stimuli
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
		psth = numpy.mean(psth, axis=0) # mean over stimuli
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
			# plt.savefig( folder+"/xcorr_rawshift_"+str(sheet1)+"_"+str(sheet2)+"_"+corr[0]+"_"+str(i)+"_"+addon+".svg", dpi=300, transparent=True )
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
	plt.ylim([0, err_max.max()])
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
	# ax1.acorr( signal1, usevlines=True, normed=True, maxlags=300, linewidth=2, color=color )
	# ax1.spines['left'].set_visible(False)
	# ax1.spines['right'].set_visible(False)
	# ax1.spines['top'].set_visible(False)
	# ax1.spines['bottom'].set_visible(False)
	# ax2 = fig.add_subplot(212, sharex=ax1)
	# ax2.acorr( signal2, usevlines=True, normed=True, maxlags=300, linewidth=2, color=color )
	# ax2.spines['left'].set_visible(False)
	# ax2.spines['right'].set_visible(False)
	# ax2.spines['top'].set_visible(False)
	# ax2.spines['bottom'].set_visible(False)
	# plt.tight_layout()
	# # plt.savefig( folder+"/autocorrelation_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".png", dpi=300, transparent=True )
	# plt.savefig( folder+"/autocorrelation_"+str(sheet1)+"_"+str(sheet2)+"_"+addon+".svg", dpi=300, transparent=True )
	# fig.clf()
	# plt.close()
	# gc.collect()




def variability( sheet, folder, stimulus, stimulus_parameter, box=None, radius=None, addon="", opposite=False, nulltime=False ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
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




def end_inhibition_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None, opposite=False, box=None, radius=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
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
	if len(neurons) < 1:
		return

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
	# plt.savefig( folder+"/suppression_index_box_"+str(sheet)+"_"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/suppression_index_box_"+str(sheet)+"_"+addon+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def end_inhibition_barplot( sheet, folder, stimulus, parameter, start, end, box=None, radius=None, xlabel="", ylabel="", closed=True, data=None, csvfile=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	folder_nums = re.findall(r'\d+', folder)
	print folder_nums
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	# get the list of all recorded neurons in sheet
	# Full
	spike_ids1 = param_filter_query(data_store, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	neurons = spike_ids1 # sheet ids
	print "Recorded neurons:", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=False  )
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
	plt.savefig( folder+"/suppression_index_"+str(sheet)+"_"+addon+"_"+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/suppression_index_"+str(sheet)+"_"+addon+"_"+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def orientation_selectivity_index_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", color_data="grey", data=None ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
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
	plt.savefig( folder+"/orientation_selectivity_index_box_"+str(sheet)+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def orientation_bias_boxplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", closed=True, data=None ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
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
	plt.savefig( folder+"/orientation_bias_box_"+str(sheet)+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def orientation_bias_barplot( sheet, folder, stimulus, parameter, start, end, box=None, radius=None, xlabel="", ylabel="", closed=True, data=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	folder_nums = re.findall(r'\d+', folder)
	print folder_nums
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	# get the list of all recorded neurons in sheet
	# Full
	spike_ids1 = param_filter_query(data_store, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	neurons = spike_ids1 # sheet ids
	print "Recorded neurons:", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=False )
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
	plt.savefig( folder+"/orientation_bias_"+str(sheet)+"_"+addon+"_"+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/orientation_bias_"+str(sheet)+"_"+addon+"_"+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], xlim=False, opposite=False, box=None, radius=None, addon="", data=None, data_curve=True ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(neurons)

	if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
		l4_exc_or_many = l4_exc_or # init
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons]) < .8)[0]]
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

	print "Selected neurons:", len(neurons)#, neurons
	if len(neurons) < 1:
		return

	TrialAveragedFiringRate(
		param_filter_query(data_store,sheet_name=sheet,st_name=stimulus),
	ParameterSet({'neurons':list(neurons)})).analyse()

	PlotTuningCurve(
	   param_filter_query( data_store, st_name=stimulus, analysis_algorithm=['TrialAveragedFiringRate'] ),
	   ParameterSet({
	        'polar': False,
	        'pool': False,
	        'centered': False,
	        'percent': False,
	        'mean': True,
	        'parameter_name' : parameter, 
	        'neurons': list(neurons), 
	        'sheet_name' : sheet
	   }), 
	   fig_param={'dpi' : 200}, 
	   plot_file_name= folder+"/TrialAveragedSensitivity_"+stimulus+"_"+parameter+"_"+str(sheet)+"_"+addon+"_mean.svg"
	).plot({
		# '*.y_lim':(0,30), 
		# '*.x_lim':(-10,100), 
		# '*.x_scale':'log', '*.x_scale_base':10,
		'*.fontsize':17
	})
	return

	# rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=True ) 
	# # print rates

	# # compute per-trial mean rate over cells
	# mean_rates = numpy.mean(rates, axis=1) 
	# # print "Ex. collapsed_mean_rates: ", mean_rates.shape
	# # print "Ex. collapsed_mean_rates: ", mean_rates
	# std_rates = numpy.std(rates, axis=1, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	# # print "Ex. collapsed_std_rates: ", std_rates

	# # print "stimuli: ", stimuli
	# print "final means and stds: ", mean_rates, std_rates
	# # print sorted( zip(stimuli, mean_rates, std_rates) )
	# final_sorted = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates, std_rates) ) ) ]

	# if parameter == "orientation":
	# 	# append first mean and std to the end to close the circle
	# 	print len(final_sorted), final_sorted
	# 	final_sorted[0] = numpy.append(final_sorted[0], 3.14)
	# 	final_sorted[1] = numpy.append(final_sorted[1], final_sorted[1][0])
	# 	final_sorted[2] = numpy.append(final_sorted[2], final_sorted[2][0])

	# if percentile:
	# 	firing_max = numpy.amax( final_sorted[1] )
	# 	final_sorted[1] = (final_sorted[1] / firing_max) * 100

	# # Plotting tuning curve
	# matplotlib.rcParams.update({'font.size':22})
	# fig,ax = plt.subplots()

	# if data:
	# 	data_list = numpy.genfromtxt(data, delimiter=',', filling_values=None)
	# 	print data_list.shape
	# 	if data_curve:
	# 		# bootstrap taking the data as limits for uniform random sampling (10 samples)
	# 		data_rates = []
	# 		for cp in data_list:
	# 			data_rates.append( numpy.random.uniform(low=cp[0], high=cp[1], size=(10)) )
	# 		data_rates = numpy.array(data_rates)
	# 		# means as usual
	# 		data_mean_rates = numpy.mean(data_rates, axis=1) 
	# 		# print data_mean_rates
	# 		if percentile:
	# 			firing_max = numpy.amax( data_mean_rates )
	# 			data_mean_rates = data_mean_rates / firing_max * 100
	# 		data_std_rates = numpy.std(data_rates, axis=1, ddof=1) 
	# 		# print data_std_rates
	# 		# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( stimuli, data_mean_rates )
	# 		# print "Data Slope:", slope, intercept
	# 		ax.plot( stimuli, data_mean_rates, color='black', label='data' )
	# 		data_err_max = data_mean_rates + data_std_rates
	# 		data_err_min = data_mean_rates - data_std_rates
	# 		ax.fill_between(stimuli, data_err_max, data_err_min, color='black', alpha=0.6)
	# 	else:
	# 		print stimuli, data_list[:,0], data_list[:,1]
	# 		ax.scatter(stimuli, data_list[:,0], marker="o", s=80, facecolor="black", alpha=0.6, edgecolor="white")
	# 		ax.scatter(stimuli, data_list[:,1], marker="D", s=80, facecolor="black", alpha=0.6, edgecolor="white")

	# # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress( final_sorted[0], final_sorted[1] )
	# # print "Model Slope:", slope, intercept
	# ax.plot( final_sorted[0], final_sorted[1], color=color, label=sheet, linewidth=3 )
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	# ax.xaxis.set_ticks_position('bottom')
	# ax.yaxis.set_ticks_position('left')

	# if useXlog:
	# 	ax.set_xscale("log", nonposx='clip')
	# if useYlog:
	# 	ax.set_yscale("log", nonposy='clip')

	# err_max = final_sorted[1] + final_sorted[2]
	# err_min = final_sorted[1] - final_sorted[2]
	# ax.fill_between(final_sorted[0], err_max, err_min, color=color, alpha=0.3)

	# # if len(ylim)>1:
	# # 	ax.set_ylim(ylim)

	# if xlim:
	# 	ax.set_xlim(xlim)

	# if percentile:
	# 	ax.set_ylim([0,100+10])
	# 	# ax.set_ylim([0,100+numpy.amax(final_sorted[2])+10])

	# # text
	# ax.set_xlabel( xlabel )
	# if percentile:
	# 	ylabel = "Percentile " + ylabel
	# 	sheet = str(sheet) + "_percentile"
	# ax.set_ylabel( ylabel )
	# # ax.legend( loc="lower right", shadow=False )
	# plt.tight_layout()
	# dist = box if not radius else radius
	# plt.savefig( folder+"/TrialAveragedTuningCurve_"+parameter+"_"+str(sheet)+"_"+addon+"_"+str(dist)+".png", dpi=200, transparent=True )
	# plt.savefig( folder+"/TrialAveragedTuningCurve_"+parameter+"_"+str(sheet)+"_"+addon+"_"+str(dist)+".svg", dpi=300, transparent=True )
	# fig.clf()
	# plt.close()
	# # garbage
	# gc.collect()




def pairwise_scatterplot( sheet, folder_full, folder_inactive, stimulus, parameter, start, end, box=None, radius=None, xlabel="", ylabel="", withRegression=True, withCorrCoef=True, withCentroid=False, xlim=[], ylim=[], data_full="", data_inac="", data_marker="D" ):
	print inspect.stack()[0][3]
	print folder_full
	folder_nums = re.findall(r'\d+', folder_full)
	print folder_nums
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	# GET RECORDINGS

	# get the list of all recorded neurons in X_ON
	# Full
	spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons (closed):", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store_full.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons_full = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	# Inactivated
	spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	if radius or box:
		sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids2)
		positions2 = data_store_inac.get_neuron_postions()[sheet]
		if box:
			ids2 = select_ids_by_position(positions2, sheet_ids2, box=box)
		if radius:
			ids2 = select_ids_by_position(positions2, sheet_ids2, radius=radius)
		neurons_inac = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=ids2)

	# Intersection of full and inac
	if set(neurons_full)==set(neurons_inac):
		neurons = neurons_full
		num_cells = len(neurons_full)
	else:
		neurons = numpy.intersect1d(neurons_full, neurons_inac)
		num_cells = len(neurons)

	# print "neurons_full:", len(neurons_full), neurons_full
	# print "neurons_inac:", len(neurons_inac), neurons_inac
	# print "neurons:", num_cells, neurons

	assert num_cells > 0 , "ERROR: the number of recorded neurons is 0"

	# compute firing rates
	# required shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)
	dstims = {}

	# Closed
	TrialAveragedFiringRate(
		param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv1 = param_filter_query(data_store_full, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls1 = dsv1.get_analysis_result( sheet_name=sheet )
	# for key, asl in sorted( asls1, key=lambda x: x.get(parameter) ):
	closed_dict = {}
	for asl in asls1:
		# print asl.stimulus_id
		stim = eval(asl.stimulus_id).get(parameter)
		dstims[stim] = stim
		closed_dict[stim] = asl.get_value_by_id(neurons)
	# print closed_dict
	all_closed_values = numpy.array([closed_dict[k] for k in sorted(closed_dict)])
	# print all_closed_values.shape
	# (10, 133)

	stims = sorted(dstims)

	# Open
	TrialAveragedFiringRate(
		param_filter_query(data_store_inac, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv2 = param_filter_query(data_store_inac, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls2 = dsv2.get_analysis_result( sheet_name=sheet )
	open_dict = {}
	for asl in asls2:
		open_dict[eval(asl.stimulus_id).get(parameter)] = asl.get_value_by_id(neurons)
	# print open_dict
	all_open_values = numpy.array([open_dict[k] for k in sorted(open_dict)])
	# print all_open_values

	# (10, 133)
	# 10 stimuli
	# 133 recorded cells

	SEM_full = scipy.stats.sem(all_closed_values)
	SEM_inac = scipy.stats.sem(all_open_values)
	print "SEM", SEM_full
	print "SEM", SEM_inac
	SEM_full = SEM_full / numpy.amax(SEM_full) # normalize SEM
	SEM_inac = SEM_inac / numpy.amax(SEM_inac) # to reinject it into cell positions
	print "SEM", SEM_full
	print "SEM", SEM_inac

	index_full = numpy.argmax(all_closed_values, axis=0).astype(int)
	index_inac = numpy.argmax(all_open_values, axis=0).astype(int)
	print "indices:", index_full.shape, index_full
	# SEM_full = scipy.stats.sem(index_full)
	# SEM_inac = scipy.stats.sem(index_inac)
	# print "SEM", SEM_full
	# print "SEM", SEM_inac
	# collapsing all cells into the index of stimulus array makes the data unreadable
	# therefore we inject the SEM in the response to account for inter-cell variability
	# x_full = numpy.take( stims, index_full ) + ((2*SEM_full) * numpy.random.random_sample((len(index_full),)) - SEM_full)
	# x_inac = numpy.take( stims, index_inac ) + ((2*SEM_inac) * numpy.random.random_sample((len(index_inac),)) - SEM_inac)
	x_full = numpy.take( stims, index_full ) 
	x_inac = numpy.take( stims, index_inac ) 
	print "control: ", x_full
	print "altered: ", x_inac

	# read external data to plot as well
	if data_full and data_inac:
		data_full_list = numpy.genfromtxt(data_full, delimiter='\n')
		print "Data control: ", data_full_list
		data_inac_list = numpy.genfromtxt(data_inac, delimiter='\n')
		print "Data altered: ", data_inac_list

	# PLOTTING
	fig,ax = plt.subplots()

	ax.set_xlim( (0,1) )
	ax.set_ylim( (0,1) )
	if len(xlim)>1:
		ax.set_xlim( xlim )
		x0,x1 = ax.get_xlim()
	if len(ylim)>1:
		ax.set_ylim( ylim )
		y0,y1 = ax.get_ylim()

	ax.set_aspect( abs(x1-x0)/abs(y1-y0) )
	# add diagonal
	ax.plot( [x0,x1], [y0,y1], linestyle='--', color="k" )
	ax.scatter( x_full, x_inac, marker="o", s=80, facecolor="blue", edgecolor="white", label=sheet )
	if data_full and data_inac:
		ax.scatter( data_full_list, data_inac_list, marker=data_marker, s=60, facecolor="black", edgecolor="white", label=sheet )


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
	plt.savefig( folder_inactive+"/TrialAveragedPairwiseScatter_"+parameter+"_"+str(sheet)+".svg", dpi=200, transparent=True )
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
	plt.savefig( folder_inactive+"/response_reduction_"+str(sheet)+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()




def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = numpy.floor(numpy.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    # print X_resample.shape, X_resample
    return X_resample




def response_barplot( sheet, folder, stimulus, parameter, num_stim=10, max_stim=1., box=None, radius=None, xlabel="", data="", data_marker="D" ):
	print inspect.stack()[0][3]
	print folder
	folder_nums = re.findall(r'\d+', folder)
	print folder_nums
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)


	# get the list of all recorded neurons in sheet
	# Full
	spike_ids1 = param_filter_query(data_store, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	neurons = spike_ids1 # sheet ids
	print "Recorded neurons:", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)


	num_cells = len(neurons)
	assert num_cells > 0 , "ERROR: the number of recorded neurons is 0"
	print "neurons:", num_cells, neurons

	# compute firing rates
	# required shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)
	dstims = {}

	TrialAveragedFiringRate(
		param_filter_query(data_store, sheet_name=sheet, st_name=stimulus),
		ParameterSet({'neurons':list(neurons)})
	).analyse()
	dsv1 = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialAveragedFiringRate')
	asls1 = dsv1.get_analysis_result( sheet_name=sheet )
	# for key, asl in sorted( asls1, key=lambda x: x.get(parameter) ):
	recdict = {}
	for asl in asls1:
		# print asl.stimulus_id
		stim = eval(asl.stimulus_id).get(parameter)
		dstims[stim] = stim
		recdict[stim] = asl.get_value_by_id(neurons)
	# print recdict
	all_values = numpy.array([recdict[k] for k in sorted(recdict)])
	# print all_closed_values.shape
	# (10, 133)

	stims = sorted(dstims)
	print "stimulus parameter values: ",stims
	# (10, 133)
	# 10 stimuli
	# 133 recorded cells

	if parameter=="contrast": # take the c50 index
		def NakaRushton(c, n, Rmax, c50, m):
			return Rmax * (c**n / (c**n + c50**n)) + m
		from scipy.optimize import curve_fit
		# Naka-Rushton fit to find the c50 of each cell
		c50 = []
		print sheet
		for i,r in enumerate(numpy.transpose(all_values)):
			# bounds and margins
			Rmax = numpy.amax(r) # 
			Rmax_up = Rmax + ((numpy.amax(r)/100)*10) # 
			m = numpy.amin(r) # 
			m_down = m - ((m/100)*10) # 
			# popt, pcov = curve_fit( NakaRushton, numpy.asarray(stims), r, maxfev=10000000 ) # workaround for scipy < 0.17
			popt, pcov = curve_fit( 
				NakaRushton, # callable
				numpy.asarray(stims), # X data
				r, # Y data
				method='trf', 
				bounds=((3., Rmax, 20., m_down), (numpy.inf, Rmax_up, 50., m)), 
				p0=(3., Rmax_up, 30., m), 
			) 
			c50.append( popt[2] ) # c50 fit
			# print popt
			# plt.plot(stims, r, 'b-', label='data')
			# plt.plot(stims, NakaRushton(stims, *popt), 'r-', label='fit')
			# plt.savefig( folder+"/NakaRushton_fit_"+str(sheet)+"_"+str(i)+".png", dpi=100 )
			# plt.close()
		cfifty = numpy.array(c50) 
		print "cfifty:",cfifty
		indices = []
		for c in cfifty:
			array = numpy.asarray(stims)
			idx = (numpy.abs(stims - c)).argmin() # after subtracting the value c to all elements of stims, find the minimum
			indices.append( idx )
		indices = numpy.array(indices)

	else: # take the max response index
		indices = numpy.argmax(all_values, axis=0).astype(int)
		print "indices:", indices.shape, indices

	x = numpy.take( stims, indices ) 
	print "hist values", x
	mean = numpy.mean(x)
	print "mean:",mean

	# read external data to plot as well
	data_list = []
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')
		print "Data: ", data_list
		data_resample = bootstrap_resample(data_list, n=len(x))
		# data_hist, data_edges = numpy.histogram( data_list, bins=num_stim, range=(0.0,max_stim), density=True )
		# print data_hist
		data_mean = numpy.mean(data_list)
		print "data_mean:",data_mean
		data_mean = numpy.mean(data_resample)
		print "data_mean:",data_mean

	# plt.hist([x,data_resample], bins=num_stim, normed=True, histtype='bar', range=(0.0,max_stim), color=['blue', 'black'], label=['model', 'data'])
	#ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
	d = numpy.array([x,data_resample])
	# print d.shape, d
	dt = numpy.transpose(d)
	# print dt.shape, dt
	plt.hist(dt, num_stim, normed=1, histtype='bar', stacked=True, range=(0.0,max_stim))

	plt.plot((mean, mean), (0,.5), 'r--', linewidth=2)
	if data:
		plt.plot((data_mean, data_mean), (0,.5), 'b--', linewidth=2)

	plt.legend()
	plt.savefig( folder+"/response_barplot_"+str(sheet)+"_"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/response_barplot_"+str(sheet)+"_"+addon+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()

	# # group cells by maximal firing rate stimulus
	# hist, edges = numpy.histogram( x, bins=num_stim, range=(0.0,max_stim), density=True )

	# # PLOTTING
	# width = 1.
	# ind = numpy.arange(num_stim)
	# matplotlib.rcParams.update({'font.size':22})
	# fig,ax = plt.subplots()
	# if closed:
	# 	barlist = ax.bar(ind, hist, align='center', width=width, facecolor='blue', edgecolor='blue')
	# 	ax.plot((mean, mean), (0,160), 'b--', linewidth=2)
	# else:
	# 	barlist = ax.bar(ind, hist, align='center', width=width, facecolor='cyan', edgecolor='cyan')
	# 	ax.plot((mean, mean), (0,160), 'c--', linewidth=2)

	# ax.set_xlabel(xlabel)
	# ax.set_ylabel("cells")
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	# ax.axis([ind[0]-width/2, ind[-1], 0, 160])
	# ax.set_xticks(ind) 
	# # ax.set_xticklabels(('0','','','','','0.5','','','','','1.0'))
	# if data: # in front of the synthetic
	# 	if closed:
	# 		datalist = ax.bar(ind, data_hist, align='center', width=width, facecolor='black', edgecolor='black')
	# 		ax.plot((data_mean, data_mean), (0,160), 'k--', linewidth=2)
	# 	else:
	# 		datalist = ax.bar(ind, data_hist, align='center', width=width, facecolor='grey', edgecolor='grey')
	# 		ax.plot((data_mean, data_mean), (0,160), '--', linewidth=2, color='grey')
	# plt.tight_layout()
	# plt.savefig( folder+"/response_barplot_"+str(sheet)+".png", dpi=200, transparent=True )
	# plt.savefig( folder+"/response_barplot_"+str(sheet)+".svg", dpi=200, transparent=True )
	# plt.close()
	# # garbage
	# gc.collect()




def response_boxplot( sheet, folder, stimulus, parameter, start, end, box=None, radius=None, xlabel="", ylabel="", closed=True, data=None ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	print data
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	# get the list of all recorded neurons in sheet
	# Full
	spike_ids1 = param_filter_query(data_store, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	neurons = spike_ids1 # sheet ids
	print "Recorded neurons:", len(spike_ids1)
	if radius or box:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids1)
		positions1 = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons, spikecount=False )
	print "rates.shape", rates.shape # (stimuli, cells)
	print "stimumli",stimuli

	if parameter=="contrast": # take the c50 index
		def NakaRushton(c, n, Rmax, c50, m):
			return Rmax * (c**n / (c**n + c50**n)) + m
		from scipy.optimize import curve_fit
		# Naka-Rushton fit to find the c50 of each cell
		c50 = []
		print sheet
		for i,r in enumerate(numpy.transpose(rates)):
			# bounds and margins
			Rmax = numpy.amax(r) # 
			Rmax_up = Rmax + ((numpy.amax(r)/100)*10) # 
			m = numpy.amin(r) # 
			m_down = m - ((m/100)*10) # 
			# popt, pcov = curve_fit( NakaRushton, numpy.asarray(stims), r, maxfev=10000000 ) # workaround for scipy < 0.17
			popt, pcov = curve_fit( 
				NakaRushton, # callable
				numpy.asarray(stimuli), # X data
				r, # Y data
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
		d = numpy.array(c50) 

	else: # take the max response
		d = numpy.amax(rates, axis=0)

	# read external data to plot as well
	if data:
		data_list = numpy.genfromtxt(data, delimiter='\n')

	# PLOTTING
	matplotlib.rcParams.update({'font.size':22})
	fig, axes = plt.subplots(nrows=1, ncols=2)

	box1 = axes[0].boxplot( d, notch=False, patch_artist=True, showfliers=True )
	for item in ['boxes', 'whiskers', 'caps']:
		plt.setp(box1[item], color='b', linewidth=2, linestyle='solid')

	if data:
		box2 = axes[1].boxplot( data_list, notch=False, patch_artist=True, showfliers=True )
		for item in ['boxes', 'whiskers', 'caps']:
			plt.setp(box2[item], color='g', linewidth=2, linestyle='solid')

	plt.tight_layout()
	plt.savefig( folder+"/response_boxplot_"+str(sheet)+"_"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/response_boxplot_"+str(sheet)+"_"+addon+".svg", dpi=200, transparent=True )
	plt.close()
	# garbage
	gc.collect()



def LHI( sheet, folder, stimulus, parameter, num_stim=2, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)

	trials = len(segs) / num_stim
	print "trials:",trials

	analog_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()
	if analog_ids == None or len(analog_ids)<1:
		print "No Vm recorded.\n"
		return
	ids = analog_ids

	# spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	# if spike_ids == None or len(spike_ids)<1:
	# 	print "No spike recorded.\n"
	# 	return
	# ids = spike_ids

	print "Recorded neurons:", len(ids)
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)

	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	print pnv #[0]
	pnv_ids = sorted( set(pnv.ids).intersection(set(ids)) )
	# print pnv_ids==ids
	orientations = []
	for i in pnv_ids:
		orientations.append( pnv.get_value_by_id(i) )
	# print orientations
	orinorm = ml.colors.Normalize(vmin=0., vmax=numpy.pi, clip=True)
	orimapper = ml.cm.ScalarMappable(norm=orinorm, cmap=plt.cm.hsv)
	orimapper._A = [] 

	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	# take the positions of the ids
	ids_positions = numpy.transpose(positions)[sheet_indexes,:]
	print ids_positions.shape
	# print ids_positions

	# ##############################
	# # Local Homogeneity Index
	# # the current V1 orientation map has a pixel for each 100 um, so a reasonable way to look at a neighborhood is in the order of 300 um radius
	# sigma = 0.280 # mm, since 1mm = 1deg in this cortical space
	# # sigma = 0.180 # um #   max: 0.0               min:0.0  
	# # for each location P(x,y):
	# # take an interval sigma of cells Q(x,y)
	# # For each Q:
	# # compute the complex domain exp(2i * thetaQ)
	# # multiply by the current Vm for Q <-- to give a measure of how the activity is related to the orientation
	# # Gaussianly weight it by its distance from P
	# # Sum it
	# # Divide the whole by the norm factor: 2 * numpy.pi * sigma**2

	# norm = ml.colors.Normalize(vmin=0., vmax=1., clip=True)
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.gray)
	# mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

	# plt.figure()
	# # print (2 * numpy.pi * sigma**2), numpy.sqrt(2 * numpy.pi) * sigma

	# LHI = {}
	# for i,x in zip(ids,ids_positions):
	# 	# select all cells within 3*sigma radius
	# 	sheet_Ys = select_ids_by_position(positions, sheet_indexes, radius=[0,3*sigma], origin=x.reshape(3,1))
	# 	Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_Ys)
	# 	# integrate 
	# 	vector_sum = 0.
	# 	for y,sy in zip(Ys,sheet_Ys):
	# 		# print "dist",numpy.linalg.norm( x - numpy.transpose(positions)[sy] )
	# 		complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y))
	# 		distance_weight = numpy.exp( -numpy.linalg.norm( x - numpy.transpose(positions)[sy] )**2 / (2 * sigma**2) )
	# 		vector_sum += distance_weight * complex_domain # * activity
	# 	LHI[i] = abs(vector_sum) # normalization outside of the loop
	# 	# LHI[i] = abs(vector_sum) / (2 * numpy.pi * sigma*sigma) 

	# # print max(LHI.values()), min(LHI.values())
	# for l in LHI:
	# 	LHI[l] = LHI[l] / max(LHI.values()) # nomrliazation based on maximal value

	# for i,x,o in zip(ids, ids_positions, orientations):
	# 	plt.scatter( x[0][0], x[0][1], marker='o', c=orimapper.to_rgba(o), alpha=LHI[i], edgecolors='none' ) # color orientation, alpha LHI
	# 	# plt.scatter( x[0][0], x[0][1], marker='o', c=mapper.to_rgba(LHI[i]), edgecolors='none' ) # b/w LHI

	# plt.savefig( folder+"/LHI_"+sheet+"_"+addon+".svg", dpi=300, transparent=True )
	# plt.close()
	# gc.collect()

	##############################
	# Synergy Index
	# computed from the static LHI for each cell * by its activity over time / normalized by all orientations
	sigma = 0.280 # mm, since 1mm = 1deg in this cortical space
	# sigma = 0.180 # um #   max: 0.0               min:0.0  

	# norm = ml.colors.Normalize(vmin=-1., vmax=1., clip=True) 
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.PiYG) # form black pink to green
	norm = ml.colors.Normalize(vmin=0., vmax=1., clip=True) 
	mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.binary) # form black 0 to white 1
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.gray) # form black 0 to white 1
	mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

	def dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma):
		dLHI = {}
		ivms = dict(zip(ids, vms))
		for vm, i, x in zip(vms, ids, ids_positions):
			# select all cells within 3*sigma radius
			sheet_Ys = select_ids_by_position(positions, sheet_indexes, radius=[0,3*sigma], origin=x.reshape(3,1))
			Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_Ys)
			# integrate 
			local_sum = 0.
			vector_sum = 0.
			for y,sy in zip(Ys,sheet_Ys):
				complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y))
				distance_weight = numpy.exp( -numpy.linalg.norm( x - numpy.transpose(positions)[sy] )**2 / (2 * sigma**2) )
				vector_sum += distance_weight * complex_domain * ivms[y].magnitude
			dLHI[i] = (abs(vector_sum) / len(Ys)) / (2 * numpy.pi * sigma**2) # norm terms
			# print "dLHI",dLHI[i]
		print "dLHI extremes",numpy.mean(dLHI.values()),numpy.std(dLHI.values()),min(dLHI.values()), max(dLHI.values())
		return dLHI # for each neuron

	# SI over all trials and orientations
	trial_avg_mean_SI = []
	trial_avg_stdev_SI = []
	SItrials = 0 # index of trials

	for s in segs:
		dist = eval(s.annotations['stimulus'])
		# print dist
		if dist['radius'] < 0.1:
			continue
		# if dist['trial'] > 0: # only one trial, for the moment
		# 	continue
		if dist['orientation'] > 0.0: # only one orientation, for the moment
			continue

		s.load_full()

		SItrials += 1

		for a in s.analogsignalarrays:
			# print "a.name: ",a.name
			if a.name == 'v':
				# print "a",a.shape # (10291, 900)  (vm instants t, cells)
				# print "max", numpy.amax(a.magnitude, axis=0)#.shape
				# normSI = numpy.maximum(normSI, numpy.amax(a.magnitude, axis=0)) 
				# print "normSI",normSI
				# print (2 * numpy.pi * sigma**2), numpy.sqrt(2 * numpy.pi) * sigma

				avg_resting_dLHI = {}
				for i in ids:
					avg_resting_dLHI[i] = 0.0

				for t,vms in enumerate(a):
					# print vms.shape

					if t < 50: #from 2ms on (to avoid beginnings)
						continue

					if t%30 != 0: # Dt=3ms
						continue
					print t

					if t<210: # from 5ms to 20ms
						resting_dLHI = dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma)
						# print "resting_dLHI",resting_dLHI # will be averaged over the 40ms
						for k in ids:
							avg_resting_dLHI[k] = avg_resting_dLHI[k] + resting_dLHI[k]
							# print avg_resting_dLHI[k]

					if t==210: # avg
						for k in ids:
							avg_resting_dLHI[k] = avg_resting_dLHI[k] / 5 # (15ms)/3ms
						# print "avg resting dLHI extremes",numpy.mean(avg_resting_dLHI.values()),numpy.std(avg_resting_dLHI.values()),min(avg_resting_dLHI.values()), max(avg_resting_dLHI.values())

					if t>210 and t<5000: # from 200ms up to 500ms
						stim_dLHI = dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma)

						SI = {}
						# average resting_dLHI
						for k in ids:
							SI[k] = (stim_dLHI[k] - avg_resting_dLHI[k]) / avg_resting_dLHI[k]
						# print "SI extremes",numpy.mean(SI.values()),numpy.std(SI.values()),min(SI.values()), max(SI.values())

						trialSI = [] # avg over all cells for this timestep
						norm = max(SI.values())
						for l in SI: # normalization based on maximal value, done at the end to avoid loosing info
							SI[l] = SI[l] / norm # 0 to 1
							trialSI.append(SI[l])
						trial_avg_mean_SI.append( numpy.mean(trialSI) )
						trial_avg_stdev_SI.append( numpy.std(trialSI) )

						## plot each instant
						# time = '{:04d}'.format(t/10) # sim at 0.1ms

						# # open image
						# plt.figure()

						# for i,x in zip(ids, ids_positions):
						# 	# print i, x
						# 	plt.scatter( x[0][0], x[0][1], marker='o', c=mapper.to_rgba(SI[i]), edgecolors='none' )
						# 	plt.xlabel(time, color='silver', fontsize=22)
						# # close image
						# print 'printing', time
						# plt.savefig( folder+"/SI_"+sheet+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
						# plt.close()
						# gc.collect()

	# SI mean over trials
	# print "trial_avg_mean_SI", trial_avg_mean_SI, len(trial_avg_mean_SI)
	trial_avg_mean_SI = numpy.array(trial_avg_mean_SI)
	trial_avg_mean_SI /= SItrials 
	trial_avg_stdev_SI = numpy.array(trial_avg_stdev_SI)
	trial_avg_stdev_SI /= SItrials 
	print "max mean:",max(trial_avg_mean_SI), "max std:",max(trial_avg_stdev_SI)

	plt.figure()
	# err_max = trial_avg_mean_SI + trial_avg_stdev_SI
	# err_min = trial_avg_mean_SI - trial_avg_stdev_SI
	# plt.fill_between(range(0,len(trial_avg_mean_SI)), err_max, err_min, color='grey', alpha=0.3)
	plt.plot(trial_avg_mean_SI, color="black", linewidth=3.)
	# plt.yscale('log')
	plt.savefig( folder+"/trial_avg_SI_"+sheet+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()



def SynergyIndex_Vm( sheet, folder, stimulus, parameter, num_stim=2, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)

	trials = len(segs) / num_stim
	print "trials:",trials

	analog_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()
	if analog_ids == None or len(analog_ids)<1:
		print "No Vm recorded.\n"
		return
	ids = analog_ids

	print "Recorded neurons:", len(ids)
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)

	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]

	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	# take the positions of the ids
	ids_positions = numpy.transpose(positions)[sheet_indexes,:]
	print ids_positions.shape
	# print ids_positions

	##############################
	# Synergy Index
	# computed from the static LHI for each cell * by its activity over time / normalized by all orientations
	sigma = 0.280 # mm, since 1mm = 1deg in this cortical space
	# sigma = 0.180 # um #   max: 0.0               min:0.0  

	# norm = ml.colors.Normalize(vmin=-1., vmax=1., clip=True) 
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.PiYG) # form black pink to green
	norm = ml.colors.Normalize(vmin=0., vmax=1., clip=True) 
	mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.binary) # form black 0 to white 1
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.gray) # form black 0 to white 1
	mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

	def dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma):
		dLHI = {}
		ivms = dict(zip(ids, vms))
		for vm, i, x in zip(vms, ids, ids_positions):
			# select all cells within 3*sigma radius
			sheet_Ys = select_ids_by_position(positions, sheet_indexes, radius=[0,3*sigma], origin=x.reshape(3,1))
			Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_Ys)
			# integrate 
			local_sum = 0.
			vector_sum = 0.
			for y,sy in zip(Ys,sheet_Ys):
				complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y))
				distance_weight = numpy.exp( -numpy.linalg.norm( x - numpy.transpose(positions)[sy] )**2 / (2 * sigma**2) )
				vector_sum += distance_weight * complex_domain * ivms[y].magnitude
			dLHI[i] = (abs(vector_sum) / len(Ys)) / (2 * numpy.pi * sigma**2) # norm terms
			# print "dLHI",dLHI[i]
		print "dLHI extremes",numpy.mean(dLHI.values()),numpy.std(dLHI.values()),min(dLHI.values()), max(dLHI.values())
		return dLHI # for each neuron

	# SI over all trials and orientations
	trial_avg_mean_SI = []
	trial_avg_stdev_SI = []
	SItrials = 0 # index of trials

	for s in segs:
		dist = eval(s.annotations['stimulus'])
		# print dist
		if dist['radius'] < 0.1:
			continue
		# if dist['trial'] > 0: 
		# 	continue
		if dist['orientation'] > 0.0: # only one orientation, for the moment
			continue

		s.load_full()

		SItrials += 1

		for a in s.analogsignalarrays:
			# print "a.name: ",a.name
			if a.name == 'v':
				# print "a",a.shape # (10291, 900)  (vm instants t, cells)
				# print "max", numpy.amax(a.magnitude, axis=0)#.shape
				# normSI = numpy.maximum(normSI, numpy.amax(a.magnitude, axis=0)) 
				# print "normSI",normSI
				# print (2 * numpy.pi * sigma**2), numpy.sqrt(2 * numpy.pi) * sigma

				avg_resting_dLHI = {}
				for i in ids:
					avg_resting_dLHI[i] = 0.0

				for t,vms in enumerate(a):
					# print vms.shape

					if t < 50: #from 2ms on (to avoid beginnings)
						continue

					if t%30 != 0: # Dt=3ms
						continue
					print t

					if t<210: # from 5ms to 20ms
						resting_dLHI = dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma)
						# print "resting_dLHI",resting_dLHI # will be averaged over the 40ms
						for k in ids:
							avg_resting_dLHI[k] = avg_resting_dLHI[k] + resting_dLHI[k]
							# print avg_resting_dLHI[k]

					if t==210: # avg
						for k in ids:
							avg_resting_dLHI[k] = avg_resting_dLHI[k] / 5 # (15ms)/3ms
						# print "avg resting dLHI extremes",numpy.mean(avg_resting_dLHI.values()),numpy.std(avg_resting_dLHI.values()),min(avg_resting_dLHI.values()), max(avg_resting_dLHI.values())

					if t>210 and t<5000: # from 200ms up to 500ms
						stim_dLHI = dLHI(sheet, positions, sheet_indexes, vms, ids, ids_positions, sigma)

						SI = {}
						# average resting_dLHI
						for k in ids:
							SI[k] = (stim_dLHI[k] - avg_resting_dLHI[k]) / avg_resting_dLHI[k]
						# print "SI extremes",numpy.mean(SI.values()),numpy.std(SI.values()),min(SI.values()), max(SI.values())

						trialSI = [] # avg over all cells for this timestep
						norm = max(SI.values())
						for l in SI: # normalization based on maximal value, done at the end to avoid loosing info
							SI[l] = SI[l] / norm # 0 to 1
							trialSI.append(SI[l])
						trial_avg_mean_SI.append( numpy.mean(trialSI) )
						trial_avg_stdev_SI.append( numpy.std(trialSI) )

						## plot each instant
						# time = '{:04d}'.format(t/10) # sim at 0.1ms

						# # open image
						# plt.figure()

						# for i,x in zip(ids, ids_positions):
						# 	# print i, x
						# 	plt.scatter( x[0][0], x[0][1], marker='o', c=mapper.to_rgba(SI[i]), edgecolors='none' )
						# 	plt.xlabel(time, color='silver', fontsize=22)
						# # close image
						# print 'printing', time
						# plt.savefig( folder+"/SI_"+sheet+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
						# plt.close()
						# gc.collect()

	# SI mean over trials
	# print "trial_avg_mean_SI", trial_avg_mean_SI, len(trial_avg_mean_SI)
	trial_avg_mean_SI = numpy.array(trial_avg_mean_SI)
	trial_avg_mean_SI /= SItrials 
	trial_avg_stdev_SI = numpy.array(trial_avg_stdev_SI)
	trial_avg_stdev_SI /= SItrials 
	print "max mean:",max(trial_avg_mean_SI), "max std:",max(trial_avg_stdev_SI)

	plt.figure()
	# err_max = trial_avg_mean_SI + trial_avg_stdev_SI
	# err_min = trial_avg_mean_SI - trial_avg_stdev_SI
	# plt.fill_between(range(0,len(trial_avg_mean_SI)), err_max, err_min, color='grey', alpha=0.3)
	plt.plot(trial_avg_mean_SI, color="black", linewidth=3.)
	plt.yscale('log')
	plt.savefig( folder+"/trial_avg_SI_"+sheet+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()



def Xcorr_SynergyIndex_spikes( sheet1, folder1, sheet2, folder2, stimulus, parameter, num_stim=1, sigma=.280, bins=102, addon="" ):
	print inspect.stack()[0][3]
	print "folder1: ",folder1
	print "sheet1: ",sheet1
	print "folder2: ",folder2
	print "sheet2: ",sheet2

	import scipy.stats as stats

	#######################################
	# SI is trial avgd then xcorr is computed
	# # center iso vs center iso
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2 = SynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# # center iso vs center cross
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2 = SynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2 = SynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# # center iso vs surround iso
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2 = SynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[1.5, 2.9], addon=addon )
	# center iso vs surround cross
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI1 = SynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 0.7], addon=addon )
	# SI2 = SynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[1.7, 2.2], addon=addon )

	# r, p = stats.pearsonr(SI1, SI2)
	# print "Scipy computed Pearson r: ",r," and p-value: ",p
	# # iso cross full - Pearson r:  0.919129693878  and p-value:  3.02948884563e-42
	# # iso cross ffw  - Pearson r:  0.884136698631  and p-value:  8.05506594409e-35
	# # iso-full iso-ffw - Pearson r:  0.249474498973  and p-value:  0.0114507861861
	# # iso-full cross-ffw - Pearson r:  0.314796808134  and p-value:  0.00127099771683
	# # cross-full cross-ffw - Pearson r:  0.439838045185  and p-value:  3.7382938663e-06
	# fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
	# ax1.xcorr(SI1, SI2, usevlines=False, maxlags=None, normed=True, lw=2, ls='-', ms=0.)
	# ax1.set_ylim([0.,1.])
	# ax1.grid(True)
	# ax2.acorr(SI1, usevlines=False, normed=True, maxlags=None, lw=2, ls='-', ms=0.)
	# ax2.set_ylim([0.,1.])
	# ax2.grid(True)
	# ax3.acorr(SI2, usevlines=False, normed=True, maxlags=None, lw=2, ls='-', ms=0.)
	# ax3.set_ylim([0.,1.])
	# ax3.grid(True)
	# plt.savefig( folder1+"/trial_avg_SI_xcorr_"+sheet1+"_"+sheet2+"_"+addon+".svg", dpi=300, transparent=True )
	# plt.close()
	# gc.collect()

	#######################################
	# SI is computed, then xcorr per trial, then avgd

	# SI is whole population, each trial  
	SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=None, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=None, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# # center iso vs center iso
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# # center iso vs center cross
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# # center iso vs surround iso
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[1.5, 2.9], addon=addon )
	# center iso vs surround cross
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI1,_ = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 0.7], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[1.7, 2.2], addon=addon )

	# center iso vs center cross
	# SI1,_ = ( sheet1, folder1, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2,_ = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI1 = SingleTrialSynergyIndex_spikes( sheet1, folder1, stimulus, parameter, preferred=False, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# SI2 = SingleTrialSynergyIndex_spikes( sheet2, folder2, stimulus, parameter, preferred=True, num_stim=num_stim, sigma=sigma, bins=bins, radius=[.0, 1.3], addon=addon )
	# print len(SI1), SI1

	mean_R = []
	mean_p = []
	for i,j in zip(SI1, SI2):
		r, p = stats.pearsonr(i, j)
		print "trial Pearson's r: ",r," and p-value: ",p
		mean_R.append(r)
		mean_p.append(p)
	mean_R = numpy.mean(mean_R)
	mean_p = numpy.mean(mean_p)
	print "Trial avg Pearson's r: ",mean_R," and p-value: ",mean_p
	# # iso cross full - Pearson r:  0.919129693878  and p-value:  3.02948884563e-42
	# # iso cross ffw  - Pearson r:  0.884136698631  and p-value:  8.05506594409e-35
	# # iso-full iso-ffw - Pearson r:  0.249474498973  and p-value:  0.0114507861861
	# # iso-full cross-ffw - Pearson r:  0.314796808134  and p-value:  0.00127099771683
	# # cross-full cross-ffw - Pearson r:  0.439838045185  and p-value:  3.7382938663e-06
	# fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)
	# ax1.xcorr(SI1, SI2, usevlines=False, maxlags=None, normed=True, lw=2, ls='-', ms=0.)
	# ax1.set_ylim([0.,1.])
	# ax1.grid(True)
	# ax2.acorr(SI1, usevlines=False, normed=True, maxlags=None, lw=2, ls='-', ms=0.)
	# ax2.set_ylim([0.,1.])
	# ax2.grid(True)
	# ax3.acorr(SI2, usevlines=False, normed=True, maxlags=None, lw=2, ls='-', ms=0.)
	# ax3.set_ylim([0.,1.])
	# ax3.grid(True)
	# plt.savefig( folder1+"/trial_avg_SI_xcorr_"+sheet1+"_"+sheet2+"_"+addon+".svg", dpi=300, transparent=True )
	# plt.close()
	# gc.collect()




def SynergyIndex_spikes( sheet, folder, stimulus, parameter, preferred=True, num_stim=2, sigma=.280, bins=102, radius=None, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)

	trials = len(segs) / num_stim
	print "trials:",trials

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	if spike_ids == None or len(spike_ids)<1:
		print "No spikes recorded.\n"
		return
	ids = spike_ids

	print "Recorded neurons:", len(ids)
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)

	if radius:
		positions = data_store.get_neuron_postions()[sheet]
		ids1 = select_ids_by_position(positions, sheet_indexes, radius=radius)
		ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	# # get orientation
	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]

	# get only preferred orientation
	if preferred:
		or_ids = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(pnv.get_value_by_id(i),0,numpy.pi)  for i in ids]) < .1)[0]]
		ids = list(or_ids)
	else:
		or_ids = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(pnv.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in ids]) < .2)[0]]
		ids = list(or_ids)
		# addon = addon + "_mid"
		addon = addon + "opposite"

	print "Selected neurons:", len(ids)

	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	# take the positions of the ids
	ids_positions = numpy.transpose(positions)[sheet_indexes,:]
	print ids_positions.shape
	# print ids_positions

	##############################
	# Synergy Index
	# computed from the static LHI for each cell * by its activity over time / normalized by all orientations
	# sigma = 0.180 # um #   max: 0.0               min:0.0  
	# bins = 102 # 10ms for 1029ms

	# norm = ml.colors.Normalize(vmin=-1., vmax=1., clip=True) 
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.PiYG) # form black pink to green
	norm = ml.colors.Normalize(vmin=0., vmax=1., clip=True) 
	mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.binary) # form black 0 to white 1
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.gray) # form black 0 to white 1
	mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

	def dLHI(sheet, positions, sheet_indexes, h, ids, ids_positions, sigma, bins):
		dLHI = {}
		for i, x in zip(ids, ids_positions):
			# select all cells within 3*sigma radius
			sheet_Ys = select_ids_by_position(positions, sheet_indexes, radius=[0,3*sigma], origin=x.reshape(3,1))
			Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_Ys)
			# integrate 
			local_sum = 0.
			vector_sum = numpy.zeros(bins)
			for y,sy in zip(Ys,sheet_Ys):
				complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y))
				distance_weight = numpy.exp( -numpy.linalg.norm( x - numpy.transpose(positions)[sy] )**2 / (2 * sigma**2) )
				if h is None:
					vector_sum += [distance_weight * complex_domain] * bins
				else:
					vector_sum += distance_weight * complex_domain * h[y]
			dLHI[i] = (abs(vector_sum) / len(Ys)) / (2 * numpy.pi * sigma**2) # norm terms
			# print "dLHI[i]",dLHI[i]
		return dLHI # for each neuron

	# SI over all trials and orientations
	trials_mean_SI = []
	trials_stdev_SI = []
	trial_avg_mean_SI = []
	trial_avg_stdev_SI = []
	SItrials = 0 # index of trials

	# the resting SI is the static LHI
	print "resting LHI ..."
	LHI = dLHI(sheet, positions, sheet_indexes, None, ids, ids_positions, sigma, bins) # for each unit

	# each segment has all cells spiketrains for one trial
	for s in segs:
		dist = eval(s.annotations['stimulus'])
		# print dist
		if dist['radius'] < 0.1:
			continue
		# if dist['trial'] > 0: 
		# 	continue
		if dist['orientation'] > 0.0: # only one orientation, for the moment
			continue

		# s.load_full()

		SItrials += 1

		print "cell's spiketrains:", len(s.spiketrains) # number of cells
		h = {}
		for a in s.spiketrains: # spiketrains of one trial
			# print "notes",
			# print "a ",a # spiketrains are already in ms
			# print numpy.histogram(a, bins=bins, range=(0.0,1029.0))[0]
			h[ a.annotations['source_id'] ] = numpy.histogram(a, bins=bins, range=(0.0,1029.0))[0] # 10ms bin
			# h[ a.annotations['source_id'] ] = len(a) # number of spikes fired in this trial

		print "stimulation dLHI ...", SItrials
		stim_dLHI = dLHI(sheet, positions, sheet_indexes, h, ids, ids_positions, sigma, bins)
		# print stim_dLHI

		SI = {}
		# average resting_dLHI
		for k in ids:
			SI[k] = stim_dLHI[k] / LHI[k] # resting LHI has been initialised with ones
		# print "SI", len(SI), SI

		trials_mean_SI.append( numpy.mean(SI.values(), axis=0) )
		trials_stdev_SI.append( numpy.std(SI.values(), axis=0) )

	# SI mean over trials
	# print "trial_avg_mean_SI before", trial_avg_mean_SI, len(trial_avg_mean_SI[0])
	trial_avg_mean_SI = numpy.array(trials_mean_SI)[0]
	trial_avg_mean_SI /= SItrials 
	trial_avg_stdev_SI = numpy.array(trials_stdev_SI)[0]
	trial_avg_stdev_SI /= SItrials 

	print "mean S:", trial_avg_mean_SI.mean()
	print "mean std S:", trial_avg_stdev_SI.mean()
	# plt.figure()
	# # err_max = trial_avg_mean_SI + trial_avg_stdev_SI
	# # err_min = trial_avg_mean_SI - trial_avg_stdev_SI
	# # below_threshold_indices = err_min < 0.0
	# # err_min[below_threshold_indices] = 0.0
	# # plt.fill_between(range(0,len(trial_avg_mean_SI)), err_max, err_min, color='grey', alpha=0.3)
	# plt.plot(trial_avg_mean_SI, color="black", linewidth=3.)
	# # plt.yscale('log')
	# plt.ylim([0.,1.])
	# plt.savefig( folder+"/spikes_trial_avg_SI_"+sheet+"_"+addon+".svg", dpi=300, transparent=True )
	# plt.close()
	# gc.collect()

	return trial_avg_mean_SI



def SingleTrialSynergyIndex_spikes( sheet, folder, stimulus, parameter, preferred=True, num_stim=2, sigma=.280, bins=102, radius=None, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)

	print len(segs), len(segs) / num_stim
	trials = len(segs) / num_stim
	print "trials:",trials

	spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	if spike_ids == None or len(spike_ids)<1:
		print "No spikes recorded.\n"
		return
	ids = spike_ids

	print "Recorded neurons:", len(ids)
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)

	if radius:
		positions = data_store.get_neuron_postions()[sheet]
		ids1 = select_ids_by_position(positions, sheet_indexes, radius=radius)
		ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	# # get orientation
	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]

	# get only preferred orientation
	if preferred!=None:
		if preferred:
			or_ids = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(pnv.get_value_by_id(i),0,numpy.pi)  for i in ids]) < .1)[0]]
			ids = list(or_ids)
		else:
			or_ids = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(pnv.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in ids]) < .2)[0]]
			ids = list(or_ids)
			# addon = addon + "_mid"
			addon = addon + "opposite"

	print "Selected neurons:", len(ids)

	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	# take the positions of the ids
	ids_positions = numpy.transpose(positions)[sheet_indexes,:]
	print ids_positions.shape
	# print ids_positions

	##############################
	# Synergy Index
	# computed from the static LHI for each cell * by its activity over time / normalized by all orientations
	# sigma = 0.180 # um #   max: 0.0               min:0.0  
	# bins = 102 # 10ms for 1029ms

	# norm = ml.colors.Normalize(vmin=-1., vmax=1., clip=True) 
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.PiYG) # form black pink to green
	norm = ml.colors.Normalize(vmin=0., vmax=1., clip=True) 
	mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.binary) # form black 0 to white 1
	# mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.gray) # form black 0 to white 1
	mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

	# No locality: instead of a certain number of positions around the currently considered neuron, all recorded cells
	def dHI(sheet, sheet_indexes, h, ids, ids_positions, bins):
		dHI = {}
		sum_h = 0.
		for i, x in zip(ids, ids_positions):
			# select all cells within 3*sigma radius
			Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_indexes)
			# integrate 
			local_sum = 0.
			vector_sum = numpy.zeros(bins)
			for y,sy in zip(Ys,ids_positions):
				complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y)[0] )
				# print complex_domain.real, h, y[0]
				vector_sum += complex_domain.real * h[y[0]]
				sum_h = sum_h + h[y[0]]
			dHI[i] = (abs(vector_sum) / len(Ys)) / (2 * numpy.pi * sigma**2) # norm terms
			if h: # normalised by each psth
				dHI[i] /= sum_h/len(Ys)
			# print "dHI[i]",dHI[i]
		return dHI # for each neuron
	# def dLHI(sheet, positions, sheet_indexes, h, ids, ids_positions, sigma, bins):
	# 	dLHI = {}
	# 	sum_h = 0.
	# 	for i, x in zip(ids, ids_positions):
	# 		# select all cells within 3*sigma radius
	# 		sheet_Ys = select_ids_by_position(positions, sheet_indexes, radius=[0,3*sigma], origin=x.reshape(3,1))
	# 		Ys = data_store.get_sheet_ids(sheet_name=sheet, indexes=sheet_Ys)
	# 		# integrate 
	# 		local_sum = 0.
	# 		vector_sum = numpy.zeros(bins)
	# 		for y,sy in zip(Ys,sheet_Ys):
	# 			complex_domain = numpy.exp( 1j * 2 * pnv.get_value_by_id(y))
	# 			distance_weight = numpy.exp( -numpy.linalg.norm( x - numpy.transpose(positions)[sy] )**2 / (2 * sigma**2) )
	# 			if h is None:
	# 				vector_sum += [distance_weight * complex_domain] * bins
	# 			else:
	# 				vector_sum += distance_weight * complex_domain * h[y]
	# 				sum_h = sum_h + h[y]
	# 		dLHI[i] = (abs(vector_sum) / len(Ys)) / (2 * numpy.pi * sigma**2) # norm terms
	# 		# if h:
	# 		# 	dLHI[i] /= sum_h/len(Ys)
	# 		# print "dLHI[i]",dLHI[i]
	# 	return dLHI # for each neuron

	# SI over all trials and orientations
	trials_pop_mean_SI = []
	trials_pop_stdev_SI = []
	SItrials = 0 # index of trials

	# # the resting SI is the static HI
	# print "resting HI ..."
	# HI = dHI(sheet, sheet_indexes, None, ids, ids_positions, bins)

	# each segment has all cells spiketrains for one trial
	for s in segs:
		dist = eval(s.annotations['stimulus'])
		# # print dist
		# if dist['radius'] < 0.1:
		# 	continue
		if dist['trial'] > 10: 
			continue
		# if dist['orientation'] > 0.0: # only one orientation, for the moment
		# 	continue
		# print dist

		s.load_full()

		SItrials += 1

		print "cell's spiketrains:", len(s.spiketrains) # number of cells
		psth = {}
		for a in s.spiketrains: # spiketrains of one trial
			# print "notes",
			# print "a ",a # spiketrains are already in ms
			# print numpy.histogram(a, bins=bins, range=(0.0,1029.0))[0]
			psth[ a.annotations['source_id'] ] = numpy.histogram(a, bins=bins, range=(0.0, 1029.0))[0] # 10ms bin
			# psth[ a.annotations['source_id'] ] = len(a) # number of spikes fired in this trial

		print "stimulation dHI ...", SItrials
		stim_dHI = dHI(sheet, sheet_indexes, psth, ids, ids_positions, bins)
		# print stim_dHI

		SI = {}
		# average resting_dHI
		for k in ids:
			# SI[k] = stim_dHI[k] / HI[k] 
			SI[k] = stim_dHI[k] 
		print "SI", len(SI) #, SI # SI 529 {57346: array([ 0.99585163, ...]), ...}
		# print "SI.values()", SI.values()
		# print "times:", len(numpy.mean(SI.values(), axis=0))
		trials_pop_mean_SI.append( numpy.mean(SI.values(), axis=0) ) # mean SI of all units over time for this trial
		trials_pop_stdev_SI.append( numpy.std(SI.values(), axis=0) ) # its std

	plt.figure()
	# err_max = trials_pop_mean_SI + trials_pop_stdev_SI
	# err_min = trials_pop_mean_SI - trials_pop_stdev_SI
	# below_threshold_indices = err_min < 0.0
	# err_min[below_threshold_indices] = 0.0
	# plt.fill_between(range(0,len(trials_pop_mean_SI)), err_max, err_min, color='grey', alpha=0.3)
	# plt.plot(SI.values(), color="black", linewidth=3.)
	for pop_si in trials_pop_mean_SI:
		plt.plot(pop_si, color="black", linewidth=1., alpha=0.5)
	plt.plot(numpy.mean(trials_pop_mean_SI, axis=0), color="black", linewidth=3.)
	# plt.yscale('log')
	# plt.ylim([0.,1.])
	plt.savefig( folder+"/spikes_trial_avg_pop_avg_SI_"+sheet+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()

	return trials_pop_mean_SI, trials_pop_stdev_SI



def OrientedRaster( sheet, folder, stimulus, parameter, num_stim=2, radius=None, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	if True:
		segs = sorted( 
			param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
			key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
		)
	else: # spontaneous
		segs = sorted( 
			param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(null=True), # Init 150ms with no stimulus
			key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
		)

	trials = len(segs) / num_stim
	print "trials:",trials

	ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	if ids == None or len(ids)<1:
		print "No spikes recorded.\n"
		return
	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)
	print "Recorded neurons:", len(ids)

	if radius:
		positions = data_store.get_neuron_postions()[sheet]
		ids1 = select_ids_by_position(positions, sheet_indexes, radius=radius)
		ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=sorted(ids1))

	# # get orientation
	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	# print pnv #[0]
	pnv_ids = sorted( set(pnv.ids).intersection(set(ids)) )
	# print pnv_ids==ids
	print "Selected neurons:", len(pnv_ids)
 
	# orientations
	orientations = []
	for i in pnv_ids:
		orientations.append( pnv.get_value_by_id(i) )
	# print orientations
	sorted_ids = [x for _,x in sorted(zip(orientations,ids), key=lambda x: x[0])]

	# colorbar min=0deg, max=180deg
	orinorm = ml.colors.Normalize(vmin=0., vmax=numpy.pi, clip=True)
	orimapper = ml.cm.ScalarMappable(norm=orinorm, cmap=plt.cm.hsv)
	orimapper._A = [] 

	# cell trial averg spike count
	trial_avg_count = {}
	for o in sorted_ids:
		trial_avg_count[o] = 0.0

	# each segment has all cells spiketrains for one trial
	for s in segs:
		dist = eval(s.annotations['stimulus'])
		# print dist
		if dist['radius'] < 0.1:
			continue
		if dist['orientation'] > 0.0: # only one orientation, for the moment
			continue

		# trial avg spike count
		for o in sorted_ids:
			for a in s.spiketrains: # spiketrains of one trial
				if o == a.annotations["source_id"]:
					trial_avg_count[o] = trial_avg_count[o] + len(a)
					break # no need to iterate over the whole array once we found the id

		if dist['trial'] > 0: 
			continue

		plt.figure()
		orifr = []
		for i,o in enumerate(sorted_ids):

			for a in s.spiketrains: # spiketrains of one trial
				if o == a.annotations["source_id"]:
					orifr.append( len(a) )
					for t in a:
						plt.scatter( t, i, marker='o', c=orimapper.to_rgba(pnv.get_value_by_id(o)), edgecolors='none', s=1 )
					break # no need to iterate over the whole array once we found the id

		plt.xlabel(time, color='silver', fontsize=22)
		plt.savefig( folder+"/orisorted_evoked_raster_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
		# plt.savefig( folder+"/orisorted_spont_raster_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
		# plt.savefig( folder+"/orisorted_test_raster_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
		plt.close()

		# Spike count by orientation
		plt.figure()
		plt.plot(orifr)
		plt.savefig( folder+"/orisorted_evoked_nspikes_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
		plt.close()

	# Trial avg spike count by orientation
	plt.figure()
	trial_avg_sc = []
	for o in sorted_ids:
		trial_avg_sc.append( trial_avg_count[o] / trials )
	plt.plot(trial_avg_sc)
	plt.savefig( folder+"/trials_orisorted_evoked_nspikes_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()

	gc.collect()



def VSDI( sheet, folder, stimulus, parameter, num_stim=2, addon="" ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	polarity = True # exc
	c = 'red'
	if "Inh" in sheet:
		polarity = False
		c = 'blue'

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	spont_segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(null=True), # Init 150ms with no stimulus
		# param_filter_query(data_store, sheet_name=sheet, st_direct_stimulation_name="None", st_name='InternalStimulus').get_segments(),
		# param_filter_query(data_store, direct_stimulation_name='None', sheet_name=sheet).get_segments(), # 1029ms NoStimulation
		key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	)
	# print segs
	print "spont_trials:",len(spont_segs)
	spont_trials = len(spont_segs) / num_stim
	print "spont_trials:",spont_trials
	trials = len(segs) / num_stim
	print "trials:",trials

	analog_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()
	if analog_ids == None or len(analog_ids)<1:
		print "No Vm recorded.\n"
		return
	print "Recorded neurons:", len(analog_ids)
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	# avg vm
	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	###############################
	# # Vm PLOTS
	###############################
	# segs = spont_segs

	# the cortical surface is going to be divided into annuli (beyond the current stimulus size)
	# this mean vm composes a plot of each annulus (row) over time
	
	# annulus_radius = 0.3
	# start = 1.4
	# stop = 3. - annulus_radius
	# num = 5 # annuli

	annulus_radius = 0.3
	start = 0.0
	stop = 1.6 - annulus_radius
	num = 5 # annuli

	# open image
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(num, 1, hspace=0.3)
	arrival = []
	for n,r in enumerate(numpy.linspace(start, stop, num=num)):
		radius = [r,r+annulus_radius]
		annulus_ids = select_ids_by_position(positions, sheet_indexes, radius=radius)
		print "annulus:  ",radius, "(radii)  ", len(annulus_ids), "(#ids)"
		# print len(annulus_ids), annulus_ids

		trial_avg_prime_response = []
		trial_avg_annulus_mean_vm = []
		for s in segs:

			dist = eval(s.annotations['stimulus'])
			if dist['radius'] < 0.1:
				continue
			print "radius",dist['radius'], "trial",dist['trial']

			s.load_full()
			# print "s.analogsignalarrays", s.analogsignalarrays # if not pre-loaded, it results empty in loop

			# print gs, n
			ax = plt.subplot(gs[n])

			for a in s.analogsignalarrays:
				# print "a.name: ",a.name
				if a.name == 'v':
					# print "a",a.shape # (10291, 900)  (vm instants t, cells)

					# annulus population average
					# print "annulus_ids",len(annulus_ids)
					# print annulus_ids
					# for aid in annulus_ids:
					# 	print aid, numpy.nonzero(sheet_indexes == aid)[0][0]

					# annulus_vms = numpy.array([a[:,numpy.nonzero(sheet_indexes == aid)[0]] for aid in annulus_ids])
					annulus_mean_vm = numpy.array([a[:,numpy.nonzero(sheet_indexes == aid)[0]] for aid in annulus_ids]).mean(axis=0)[0:2000,:]
					# print "annulus_vms",annulus_vms.shape
					# only annulus ids in the mean
					# annulus_mean_vm = numpy.mean( annulus_vms, axis=0)[0:2000,:]
					# print "annulus_mean_vm", annulus_mean_vm.shape
					trial_avg_annulus_mean_vm.append(annulus_mean_vm)
					# print "annulus_mean_vm", annulus_mean_vm
					# threshold = annulus_mean_vm.max() - (annulus_mean_vm.max()-annulus_mean_vm.min())/10 # threshold at: 90% of the max-min interval
					# prime_response = numpy.argmax(annulus_mean_vm > threshold)
					# trial_avg_prime_response.append(prime_response)

					plt.axvline(x=numpy.argmax(annulus_mean_vm), color=c, alpha=0.5)
					ax.plot(annulus_mean_vm, color=c, alpha=0.5)
					ax.set_ylim([-75.,-50.])

		# means
		# trial_avg_prime_response = numpy.mean(trial_avg_prime_response)
		trial_avg_annulus_mean_vm = numpy.mean(trial_avg_annulus_mean_vm, axis=0)

		from scipy.signal import argrelextrema
		peaks = argrelextrema(trial_avg_annulus_mean_vm, numpy.greater, order=200)[0]
		print peaks

		for peak in peaks:
			plt.axvline(x=peak, color=c, linewidth=3.)#, linestyle=linestyle)

		ax.plot(trial_avg_annulus_mean_vm, color=c, linewidth=3.)
		ax.set_ylim([-75.,-50.])
		fig.add_subplot(ax)
		# s.release()

	# close image
	# title = "propagation velocity {:f} SD {:f} m/s".format((annulus_radius*.001)/(numpy.mean(arrival)*.0001), numpy.std(arrival)) # 
	plt.xlabel("time (0.1 ms) ")#+title)
	plt.savefig( folder+"/VSDI_mean_vm_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+".svg", dpi=300, transparent=True )
	plt.close()
	gc.collect()
	################################


	# #################################################
	# # FULL MAP FRAMES - ***** SINGLE TRIAL ONLY *****
	# #################################################
	# # segs = spont_segs # to visualize only spontaneous activity

	# positions = numpy.transpose(positions)
	# print positions.shape # all 10800

	# # take the sheet_indexes positions of the analog_ids
	# analog_positions = positions[sheet_indexes,:]
	# print analog_positions.shape
	# # print analog_positions

	# NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	# pnv = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	# print pnv #[0]
	# pnv_ids = sorted( set(pnv.ids).intersection(set(analog_ids)) )
	# # print pnv_ids==analog_ids
	# orientations = []
	# for i in pnv_ids:
	# 	orientations.append( pnv.get_value_by_id(i) )
	# # print orientations
	# # colorbar min=0deg, max=180deg
	# orinorm = ml.colors.Normalize(vmin=0., vmax=numpy.pi, clip=True)
	# orimapper = ml.cm.ScalarMappable(norm=orinorm, cmap=plt.cm.hsv)
	# orimapper._A = [] 

	# # # colorbar min=resting, max=threshold
	# # norm = ml.colors.Normalize(vmin=-80., vmax=-50., clip=True)
	# # mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
	# # mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
	# # # ml.rcParams.update({'font.size':22})
	# # # ml.rcParams.update({'font.color':'silver'})

	# for s in segs:

	# 	dist = eval(s.annotations['stimulus'])
	# 	print dist['radius']
	# 	if dist['radius'] < 0.1:
	# 		continue

	# 	s.load_full()
	# 	# print "s.analogsignalarrays", s.analogsignalarrays # if not pre-loaded, it results empty in loop

	# 	for a in s.analogsignalarrays:
	# 		# print "a.name: ",a.name
	# 		if a.name == 'v':
	# 			print a.shape # (10291, 900)  (instant t, cells' vm)

	# 			for t,vms in enumerate(a):
	# 				if t/10 > 500:
	# 					break

	# 				if t%20 == 0: # each 2 ms
	# 					time = '{:04d}'.format(t/10)

	# 					# # open image
	# 					# plt.figure()
	# 					# for vm,i,p in zip(vms, analog_ids, analog_positions):
	# 					# 	# print vm, i, p
	# 					# 	plt.scatter( p[0][0], p[0][1], marker='o', c=mapper.to_rgba(vm), edgecolors='none' )
	# 					# 	plt.xlabel(time, color='silver', fontsize=22)
	# 					# # cbar = plt.colorbar(mapper)
	# 					# # cbar.ax.set_ylabel('mV', rotation=270)
	# 					# # close image
	# 					# plt.savefig( folder+"/VSDI_spont_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
	# 					# plt.close()
	# 					# gc.collect()

	# 					# # open image polarity
	# 					# plt.figure()
	# 					# for vm,i,p in zip(vms, analog_ids, analog_positions):
	# 					# 	# print vm, i, p
	# 					# 	# the alpha value (between 0 and 1) is the normalized Vm
	# 					# 	a = 1 - (numpy.abs(vm.magnitude)-40.)/(180.-40.) # 1 - (abs(vm)-vm_peak) / (vm_min-vm_peak)
	# 					# 	# print a
	# 					# 	color = 'red' if polarity else 'blue'
	# 					# 	plt.scatter( p[0][0], p[0][1], marker='o', c=color, alpha=a, edgecolors='none' )
	# 					# 	plt.xlabel(time, color='silver', fontsize=22)
	# 					# # close image
	# 					# plt.savefig( folder+"/VSDI_polarity_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
	# 					# plt.close()
	# 					# gc.collect()

	# 					# open image
	# 					plt.figure()
	# 					for vm,i,p,o in zip(vms, analog_ids, analog_positions, orientations):
	# 						# print vm, i, p, o
	# 						# the alpha value (between 0 and 1) is the normalized Vm in the range -60 < < -50
	# 						clipped_vm = vm.magnitude 
	# 						if vm.magnitude > -50.:
	# 							clipped_vm = -50.
	# 						if vm.magnitude < -60.:
	# 							clipped_vm = -60.
	# 						a = 1 - (numpy.abs(clipped_vm)-50)/10. # (abs(vm)-vm_min) / (vm_min-vm_peak)
	# 						# print a
	# 						plt.scatter( p[0][0], p[0][1], marker='o', c=orimapper.to_rgba(o), alpha=a, edgecolors='none' )
	# 						plt.xlabel(time, color='silver', fontsize=22)
	# 					# close image
	# 					plt.savefig( folder+"/VSDI_oriented_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
	# 					plt.close()
	# 					gc.collect()

	# 	s.release()



def trial_averaged_Vm( sheet, folder, stimulus, parameter, opposite=False, box=None, radius=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids()
	if analog_ids == None:
		print "No Vm recorded.\n"
		return
	print "Recorded neurons:", len(analog_ids)

	if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in analog_ids]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < .1)[0]]
		analog_ids = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		analog_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(analog_ids)
	if len(analog_ids) < 1:
		return

	dsv = param_filter_query( data_store, sheet_name=sheet, st_name=stimulus )

	dist = box if not radius else radius
	for n in analog_ids:
		VmPlot(
			dsv,
			ParameterSet({
				'neuron': n, 
				'sheet_name' : sheet,
				'spontaneous' : True,
			}), 
			fig_param={'dpi' : 300, 'figsize': (40,5)}, 
			# plot_file_name=folder+"/Vm_"+parameter+"_"+str(sheet)+"_"+str(dist)+"_"+str(n)+"_"+addon+".png"
			plot_file_name=folder+"/Vm_"+parameter+"_"+str(sheet)+"_radius"+str(dist)+"_"+str(n)+"_"+addon+".svg"
		).plot({
			# '*.y_lim':(0,60), 
			# '*.x_scale':'log', '*.x_scale_base':2,
			# '*.y_ticks':[5, 10, 25, 50, 60], 
			# # '*.y_scale':'linear', 
			# '*.y_scale':'log', '*.y_scale_base':2,
			# '*.fontsize':24
		})



def trial_averaged_raster( sheet, folder, stimulus, parameter, opposite=False, box=None, radius=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	spike_ids = param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	if spike_ids == None:
		print "No spikes recorded.\n"
		return
	print "Recorded neurons:", len(spike_ids)

	if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in spike_ids]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in spike_ids]) < .1)[0]]
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
	if len(spike_ids) < 1:
		return

	dsv = param_filter_query( data_store, sheet_name=sheet, st_name=stimulus )
	dist = box if not radius else radius

	# Raster + Histogram
	RasterPlot(
		dsv,
		ParameterSet({
			'sheet_name':sheet, 
			'neurons':list(spike_ids), 
			'trial_averaged_histogram':True,
			'spontaneous' : True
		}),
		fig_param={'dpi' : 100,'figsize': (100,50)},
		plot_file_name=folder+"/HistRaster_"+parameter+"_"+str(sheet)+"_radius"+str(dist)+"_"+addon+".svg"
	).plot({'SpikeRasterPlot.group_trials':True})



def trial_averaged_conductance_tuning_curve( sheet, folder, stimulus, parameter, percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], opposite=False, box=None, radius=None, addon="", inputoutputratio=False, dashed=False ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_esyn_ids()
	if analog_ids == None:
		print "No Vm recorded.\n"
		return
	print "Recorded neurons:", len(analog_ids)

	# if sheet=='V1_Exc_L4' or sheet=='V1_Inh_L4':
	# 	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	# 	l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	# 	if opposite:
	# 		addon = addon +"_opposite"
	# 		l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in analog_ids]) < .1)[0]]
	# 	else:
	# 		addon = addon +"_same"
	# 		l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < .1)[0]]
	# 	analog_ids = list(l4_exc_or_many)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		analog_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(analog_ids)
	if len(analog_ids)<1:
		return

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
	print pop_e.shape
	mean_pop_e = numpy.mean(pop_e, axis=(2,1) )
	mean_pop_i = numpy.mean(pop_i, axis=(2,1) ) 
	std_pop_e = numpy.std(pop_e, axis=(2,1), ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	std_pop_i = numpy.std(pop_i, axis=(2,1), ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
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

	matplotlib.rcParams.update({'font.size':22})

	# Conductances contrast
	fig,ax = plt.subplots()
	color = 'black' if 'closed' in folder else 'gray'
	print final_sorted_e[1]/(final_sorted_e[1]+final_sorted_i[1])
	ax.plot( final_sorted_e[0], final_sorted_e[1]/(final_sorted_e[1]+final_sorted_i[1]), color=color, linewidth=3 )
	ax.set_xlabel( parameter )
	# ax.set_ylim([0,1])
	plt.tight_layout()
	dist = box if not radius else radius
	plt.savefig( folder+"/TrialAveragedConductancesContrast_"+sheet+"_"+parameter+str(dist)+"_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()

	# Conductances ratio
	fig,ax = plt.subplots()
	color = 'black' if 'closed' in folder else 'gray'
	ax.plot( final_sorted_i[0], final_sorted_e[1]/final_sorted_i[1], color=color, linewidth=3 )
	ax.set_xlabel( parameter )
	# ax.set_ylim([0,2])
	plt.tight_layout()
	dist = box if not radius else radius
	plt.savefig( folder+"/TrialAveragedConductancesRatio_"+sheet+"_"+parameter+str(dist)+"_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()

	# Plotting tuning curve
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
	# ax2.set_xlabel( parameter )
	plt.tight_layout()
	# plt.savefig( folder+"/TrialAveragedConductances_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/TrialAveragedConductances_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".svg", dpi=300, transparent=True )
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
		ax.plot( final_sorted_e[1], final_sorted[1], color=color, linewidth=2 )
		ax.set_xlim([.0,6.])
		ax.set_xlabel( "Input (nS)" )
		ax.set_ylim([.0,1.])
		ax.set_ylabel( "Spike probability" )
		plt.tight_layout()
		# plt.savefig( folder+"/TrialAveragedInputOutputRatio_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".png", dpi=200, transparent=True )
		plt.savefig( folder+"/TrialAveragedInputOutputRatio_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".svg", dpi=300, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()



def trial_averaged_ratio_tuning_curve( sheet1, sheet2, folder, stimulus, parameter, percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], opposite=False, box=None, radius=None, addon="", dashed=False ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet1: ",sheet1
	print "sheet2: ",sheet2
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons1 = param_filter_query(data_store, sheet_name=sheet1, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons1:", len(neurons1)
	neurons2 = param_filter_query(data_store, sheet_name=sheet2, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons2:", len(neurons2)
	if len(neurons1)<1 or len(neurons2)<1:
		print "ERROR: Not enough recorded neurons."
		return

	if sheet1=='V1_Exc_L4' or sheet1=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet1)[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons1)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons1]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons1)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons1]) < .1)[0]]
		neurons1 = list(l4_exc_or_many)
	if sheet2=='V1_Exc_L4' or sheet2=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet2)[0]
		if opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons2)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in neurons2]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons2)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons2]) < .1)[0]]
		neurons2 = list(l4_exc_or_many)

	if radius or box:
		sheet_ids1 = data_store.get_sheet_indexes(sheet_name=sheet1, neuron_ids=neurons1)
		positions1 = data_store.get_neuron_postions()[sheet1]
		if box:
			ids1 = select_ids_by_position(positions1, sheet_ids1, box=box)
		if radius:
			ids1 = select_ids_by_position(positions1, sheet_ids1, radius=radius)
		neurons1 = data_store.get_sheet_ids(sheet_name=sheet1, indexes=ids1)

		sheet_ids2 = data_store.get_sheet_indexes(sheet_name=sheet2, neuron_ids=neurons2)
		positions2 = data_store.get_neuron_postions()[sheet2]
		if box:
			ids2 = select_ids_by_position(positions2, sheet_ids2, box=box)
		if radius:
			ids2 = select_ids_by_position(positions2, sheet_ids2, radius=radius)
		neurons2 = data_store.get_sheet_ids(sheet_name=sheet2, indexes=ids2)

	print "Selected neurons1:", len(neurons1)
	if len(neurons1)<1:
		return
	print "Selected neurons2:", len(neurons2)
	if len(neurons2)<1:
		return

	rates1, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet1, 100., 1000., parameter, neurons=neurons1, spikecount=False ) 
	rates2, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet2, 100., 1000., parameter, neurons=neurons2, spikecount=False ) 
	print stimuli
	# compute per-trial mean rate over cells
	mean_rates1 = numpy.mean(rates1, axis=1) / numpy.max(rates1)
	mean_rates2 = numpy.mean(rates2, axis=1) / numpy.max(rates2)
	final_sorted1 = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates1) ) ) ]
	final_sorted2 = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates2) ) ) ]
	ratio = final_sorted1[1] / final_sorted2[1]
	fig,ax = plt.subplots()
	ax.plot( final_sorted1[0], final_sorted1[1], color='red', linewidth=2 )
	ax.plot( final_sorted1[0], final_sorted2[1], color='blue', linewidth=2 )
	ax.plot( final_sorted1[0], ratio, color=color, linewidth=2 )
	# ax.set_xlim([.0,1.])
	ax.set_xlabel( "Radius" )
	# ax.set_ylim([.0,5.])
	ax.set_ylabel( "Exc/Inh ratio" )
	if useXlog:
		ax.set_xscale("log", nonposx='clip')
	plt.tight_layout()
	# plt.savefig( folder+"/TrialAveragedRatio_"+sheet+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".png", dpi=200, transparent=True )
	plt.savefig( folder+"/TrialAveragedRatio_"+sheet1+"_"+sheet2+"_"+parameter+"_box"+str(box)+"_pop_"+addon+".svg", dpi=300, transparent=True )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()



def trial_averaged_conductance_timecourse( sheet, folder, stimulus, parameter, ticks, ylim=[0.,100.], box=[], addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
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

		# # text
		# plt.tight_layout()
		# plt.savefig( folder+"/TimecourseConductances_"+sheet+"_"+parameter+"_"+str(ticks[s])+".png", dpi=200, transparent=True )
		# plt.savefig( folder+"/TimecourseConductances_"+sheet+"_"+parameter+"_"+str(ticks[s])+".svg", dpi=200, transparent=True )
		# fig.clf()
		# plt.close()
		# # garbage
		# gc.collect()


		# Conductances contrast
		fig,ax = plt.subplots()
		color = 'black' if 'closed' in folder else 'gray'
		ax.plot( range(0,len(mean_pop_e[s])), mean_pop_e[s]/(mean_pop_e[s]+mean_pop_i[s]), color=color, linewidth=3 )
		ax.set_xlabel( parameter )
		ax.set_ylim([0,1])
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseTrialAveragedConductancesContrast_"+sheet+"_"+parameter+str(box)+"_"+addon+"_"+str(ticks[s])+".svg", dpi=300, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()

		# Conductances ratio
		fig,ax = plt.subplots()
		color = 'black' if 'closed' in folder else 'gray'
		ax.plot( range(0,len(mean_pop_e[s])), mean_pop_e[s]/mean_pop_i[s], color=color, linewidth=3 )
		ax.set_xlabel( parameter )
		# ax.set_ylim([0,2])
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseTrialAveragedConductancesRatio_"+sheet+"_"+parameter+str(box)+"_"+addon+"_"+str(ticks[s])+".svg", dpi=300, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()



def save_positions( sheet, folder, stimulus, parameter ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )
	print "Recorded neurons:", len(analog_ids)

	sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
	positions = data_store.get_neuron_postions()[sheet]

	idpos = {}
	for i in sheet_ids:
		idpos[i[0]] = [ positions[0][i][0],positions[1][i][0],positions[2][i][0] ]
	print idpos

	import json
	json = json.dumps(idpos)
	print folder+"/id_positions_"+str(sheet)+".json"
	f = open(folder+"/id_positions_"+str(sheet)+".json", 'w')
	f.write(json)
	f.close()


	# segs = sorted( 
	# 	param_filter_query(data_store, st_name=stimulus, sheet_name=sheet).get_segments(), 
	# 	# key = lambda x : MozaikParametrized.idd(x.annotations['stimulus']).radius 
	# 	key = lambda x : getattr(MozaikParametrized.idd(x.annotations['stimulus']), parameter) 
	# )



def SpikeTriggeredAverage(lfp_sheet, spike_sheet, folder, stimulus, parameter, ylim=[0.,100.], tip_box=[[.0,.0],[.0,.0]], radius=False, lfp_opposite=False, spike_opposite=False, addon="", color="black"):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "LFP sheet: ",lfp_sheet
	print "Spike sheet: ",spike_sheet
	import ast
	from scipy import signal
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=False)

	# LFP
	# 95% of the LFP signal is a result of all exc and inh cells conductances from 250um radius from the tip of the electrode (Katzner et al. 2009).
	# Therefore we include all recorded cells but account for the distance-dependent contribution weighting currents /r^2
	# We assume that the electrode has been placed in the cortical coordinates <tip>
	# Only excitatory neurons are relevant for the LFP (because of their geometry) Bartos
	lfp_neurons = param_filter_query(data_store, sheet_name=lfp_sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()

	if lfp_neurons == None:
		print "No Exc Vm recorded.\n"
		return
	print "Recorded neurons for LFP:", len(lfp_neurons)
	lfp_positions = data_store.get_neuron_postions()[lfp_sheet] # position is in visual space degrees

	# choose LFP tip position
	# select all neurons id having a certain orientation preference
	or_neurons = lfp_neurons
	if lfp_sheet=='V1_Exc_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=lfp_sheet)[0]
		if lfp_opposite:
			exc_or_g = numpy.array(lfp_neurons)[numpy.nonzero(numpy.array([circular_dist(exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi)  for i in lfp_neurons]) < .1)[0]]
		else:
			exc_or_g = numpy.array(lfp_neurons)[numpy.nonzero(numpy.array([circular_dist(exc_or.get_value_by_id(i),0,numpy.pi)  for i in lfp_neurons]) < .1)[0]]
		or_neurons = list(exc_or_g)

	or_sheet_ids = data_store.get_sheet_indexes(sheet_name=lfp_sheet, neuron_ids=or_neurons)
	or_neurons = select_ids_by_position(lfp_positions, or_sheet_ids, box=tip_box)
	print "Oriented neurons idds to choose the LFP tip electrode location:", len(or_neurons), or_neurons

	xs = lfp_positions[0][or_neurons]
	ys = lfp_positions[1][or_neurons]
	# create list for each point
	or_dists = [[] for i in range(len(or_neurons))]
	selected_or_ids = [[] for i in range(len(or_neurons))]
	for i,_ in enumerate(or_neurons):
		# calculate distance from all others
		for j,o in enumerate(or_neurons):
			dist = math.sqrt( (xs[j]-xs[i])**2 + (ys[j]-ys[i])**2 )
			if dist <= 0.8: # minimal distance between oriented blots in cortex 
				selected_or_ids[i].append(o)
				# print lfp_positions[0][o], lfp_positions[1][o]
	# pick the largest list	
	selected_or_ids.sort(key = lambda x: len(x), reverse=True)
	print selected_or_ids[0]
	# average the selected xs and ys to generate the centroid, which is the tip
	x = 0
	y = 0
	for i in selected_or_ids[0]:
		# print i
		x = x + lfp_positions[0][i]
		y = y + lfp_positions[1][i]
	x = x / len(selected_or_ids[0])
	y = y / len(selected_or_ids[0])
	tip = [[x],[y],[.0]]
	print "LFP electrod tip location (x,y) in degrees:", tip

	# not all neurons are necessary, >100 are enough
	chosen_ids = numpy.random.randint(0, len(lfp_neurons), size=100 )
	# print chosen_ids
	lfp_neurons = lfp_neurons[chosen_ids]

	distances = [] # distances form origin of all excitatory neurons
	sheet_e_ids = data_store.get_sheet_indexes(sheet_name=lfp_sheet, neuron_ids=lfp_neurons) # all neurons
	magnification = 1000 # magnification factor to convert the degrees in to um
	if "X" in lfp_sheet:
		magnification = 200
	for i in sheet_e_ids:
		distances.append( numpy.linalg.norm( numpy.array((lfp_positions[0][i],lfp_positions[1][i],lfp_positions[2][i])) - numpy.array(tip) ) *magnification ) 
	distances = numpy.array(distances)
	print "Recorded distances:", len(distances)#, distances #, distances**2

	# gather vm and conductances
	segs = sorted( 
		param_filter_query(data_store, st_name=stimulus, sheet_name=lfp_sheet).get_segments(), 
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
	gc.collect()

	pop_vm = []
	pop_gsyn_e = []
	pop_gsyn_i = []
	for n,idd in enumerate(lfp_neurons):
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
			e = e.rescale(mozaik.tools.units.nS) # NEST is in nS, PyNN is in uS
			i = i.rescale(mozaik.tools.units.nS) # NEST is in nS, PyNN is in uS
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

	# We produce the current for each cell for this time interval, with the Ohm law:
	# I = ge(V-Ee) + gi(V+Ei)
	# where 
	# Ee is the equilibrium for exc, which is 0.0
	# Ei is the equilibrium for inh, which is -80.0
	i = pop_e*pop_v + pop_i*(pop_v-80.0)
	# the LFP is the result of cells' currents
	avg_i = numpy.average( i, weights=distances**2, axis=0 )
	std_i = numpy.std( i, axis=0 )
	sigma = 0.1 # [0.1, 0.01] # Dobiszewski_et_al2012.pdf
	lfp = ( (1/(4*numpy.pi*sigma)) * avg_i ) / std_i # Z-score
	print "LFP:", lfp.shape, lfp.min(), lfp.max()
	print lfp

	#TEST: plot the LFP for each stimulus
	for s in range(num_ticks):
		# for each stimulus plot the average conductance per cell over time
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()

		ax.plot( range(0,len(lfp[s])), lfp[s], color=color, linewidth=3 )

		ax.set_ylim([lfp.min(), lfp.max()])
		ax.set_ylabel( "LFP (z-score)" )
		ax.set_xlabel( "Time (ms)" )

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.xaxis.set_ticks(ticks, ticks)
		ax.yaxis.set_ticks_position('left')

		# text
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseLFP_"+lfp_sheet+"_"+spike_sheet+"_"+parameter+"_"+str(ticks[s])+"_"+addon+".svg", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()

	# MUA
	# data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':'Thalamocortical_size_closed', 'store_stimuli' : False}),replace=True)
	neurons = []
	neurons = param_filter_query(data_store, sheet_name=spike_sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()

	if spike_sheet=='V1_Exc_L4' or spike_sheet=='V1_Inh_L4':
		NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
		l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name=spike_sheet)[0]
		if spike_opposite:
			addon = addon +"_opposite"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),numpy.pi/2,numpy.pi) for i in neurons]) < .1)[0]]
		else:
			addon = addon +"_same"
			l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi) for i in neurons]) < .1)[0]]
		neurons = list(l4_exc_or_many)

	if radius:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=spike_sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[spike_sheet]
		ids = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=spike_sheet, indexes=ids)

	print "MUA neurons:", len(neurons)
	if len(neurons) < 1:
		return

	print "Collecting spiketrains of selected neurons into dictionary ..."
	dsv1 = queries.param_filter_query(data_store, sheet_name=spike_sheet, st_name=stimulus)
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
		sta_mean = STA[i].mean(axis=0) # mean across spikes
		fig = plt.figure()
		x = range(-duration, duration)
		plt.plot( x, sta_mean, color=color, linewidth=3 )
		# plt.fill_between(x, sta_mean-sta_std, sta_mean+sta_std, color=color, alpha=0.3)
		plt.tight_layout()
		plt.ylim([sta_mean.min(),sta_mean.max()])
		plt.savefig( folder+"/STA_"+lfp_sheet+"_"+spike_sheet+"_"+st+"_"+addon+".svg", dpi=300, transparent=True )
		fig.clf()
		plt.close()

	print STA_tuning
	fig = plt.figure()
	plt.plot( range(len(STA_tuning)), numpy.array(STA_tuning)*-1., color=color, linewidth=3 ) # reversed
	plt.ylim([0, min(STA_tuning)*-1.])
	plt.tight_layout()
	# plt.savefig( folder+"/STAtuning_"+str(spike_sheet)+"_"+addon+".png", dpi=300, transparent=True )
	plt.savefig( folder+"/STAtuning_"+lfp_sheet+"_"+spike_sheet+"_"+addon+".svg", dpi=300, transparent=True )
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
		mapname = os.path.splitext(name)[0]+'.svg'
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
		ca = plt.imshow(colors, interpolation='nearest', cmap='Greys')
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



def comparison_size_tuning_map(filename, xvalues, yvalues, ticks):

	colors = numpy.zeros( (len(xvalues),len(yvalues)) )
	alpha = numpy.zeros( (len(xvalues),len(yvalues)) )

	print filename
	mapname = os.path.splitext(filename)[0]+'.svg'
	print mapname

	# cycle over lines
	with open(filename,'r') as csv:
		for i,line in enumerate(csv): 
			print line
			print eval(line)
			xvalue = eval(line)[0]
			yvalue = eval(line)[1]
			s = eval(line)[2]
			print xvalue, yvalue, s
			
			colors[xvalues.index(xvalue)][yvalues.index(yvalue)] = s # if fit[0]>0. else 0. # slope

	print colors

	plt.figure()
	ca = plt.imshow(colors, interpolation='nearest', cmap='Greys')
	cbara = plt.colorbar(ca, ticks=[numpy.amin(colors), 0, numpy.amax(colors)])
	cbara.set_label('Suppression Index')
	plt.xticks(ticks, xvalues)
	plt.yticks(ticks, yvalues)
	plt.xlabel('V1-PGN arborization radius')
	plt.ylabel('PGN-LGN arborization radius')
	plt.savefig( mapname, dpi=300, transparent=True )
	plt.close()



def trial_averaged_connectivity_count( sheet, folder, stimulus, parameter, incoming=False, box=None, radius=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	spike_ids = param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	if spike_ids == None:
		print "No spikes recorded.\n"
		return
	print "Recorded neurons:", len(spike_ids)

	if radius or box:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		if box:
			ids1 = select_ids_by_position(positions, sheet_ids, box=box)
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		spike_ids = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	print "Selected neurons:", len(spike_ids)
	# print spike_ids
	if len(spike_ids) < 1:
		return

	for i in spike_ids:
		ConnectivityPlot(
			data_store,
			ParameterSet({
				'neuron': i,  # the target neuron whose connections are to be displayed
				'reversed': incoming,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
				'sheet_name': sheet,  # for neuron in which sheet to display connectivity
			}),
			# pnv_dsv=data_store,
			fig_param={'dpi':100, 'figsize': (10,12)},
			plot_file_name='connections_'+sheet+'_cell'+str(i)+'_'+str(incoming)+'.svg'
		).plot({
			# '*.line':True,
		})

	dsv = param_filter_query( data_store, sheet_name=sheet, st_name=stimulus )
	dist = box if not radius else radius

	# Raster + Histogram
	RasterPlot(
		dsv,
		ParameterSet({
			'sheet_name':sheet, 
			'neurons':list(spike_ids), 
			'trial_averaged_histogram':True,
			'spontaneous' : True
		}),
		fig_param={'dpi' : 100,'figsize': (100,50)},
		plot_file_name=folder+"/HistRaster_"+parameter+"_"+str(sheet)+"_radius"+str(dist)+"_"+addon+".svg"
	).plot({'SpikeRasterPlot.group_trials':True})




###################################################
# Execution

full_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_luminance_open_____",

	# "Deliverable/ThalamoCorticalModel_data_contrast_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_contrast_open_____",

	# "ThalamoCorticalModel_data_spatial_open_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_LGNonly_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "/media/do/HANGAR/Thalamocortical_size_closed", # BIG
	# "/media/do/HANGAR/ThalamoCorticalModel_data_size_closed_cond_____",
	# "/media/do/HANGAR/Deliverable/ThalamoCorticalModel_data_size_closed_____", # <<<<<<< ISO Coherence, V1 conductance
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____large",
	# "ThalamoCorticalModel_data_size_closed_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_nonoverlapping_____",

	# "ThalamoCorticalModel_data_size_Yves_____", # control: small static bar
	# "Deliverable/ThalamoCorticalModel_data_size_closed_vsdi_00_radius14_____",
	# "/media/do/HANGAR/ThalamoCorticalModel_data_size_closed_vsdi_08_radius14_____",
	# "ThalamoCorticalModel_data_size_closed_vsdi_____20trials",
	# "/media/do/DATA/Deliverable/ThalamoCorticalModel_data_size_closed_vsdi_____10trials",
	# "ThalamoCorticalModel_data_size_closed_vsdi_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_vsdi_00_radius14_____",
	# "/media/do/HANGAR/ThalamoCorticalModel_data_size_feedforward_vsdi_08_radius14_____",
	# "/media/do/Sauvegarde Système/ThalamoCorticalModel_data_size_closed_vsdi_____20trials",
	# "/media/do/Sauvegarde Système/ThalamoCorticalModel_data_size_feedforward_vsdi_____20trials",
	# "ThalamoCorticalModel_data_size_feedforward_vsdi_____",

	# # Synergy Index
	"ThalamoCorticalModel_data_orientation_ffw_____",
	"ThalamoCorticalModel_data_orientation_closed_____",
	# "ThalamoCorticalModel_data_size_closed_vsdi_100micron_____",
	# "ThalamoCorticalModel_data_size_feedforward_vsdi_100micron_____", # 

	# # # sizes of feedback radius
	# "/media/do/Sauvegarde Système/ThalamoCorticalModel_data_size_closed_vsdi_____5radius",
	# "/media/do/Sauvegarde Système/ThalamoCorticalModel_data_size_closed_vsdi_____30radius",
	# "/media/do/OLD_SYST/ThalamoCorticalModel_data_size_closed_vsdi_smaller_10-15_____6trials",
	# "/media/do/OLD_SYST/ThalamoCorticalModel_data_size_closed_vsdi_larger_120-270_____6trials",
	# "ThalamoCorticalModel_data_size_closed_vsdi_larger_____",

	# "Deliverable/ThalamoCorticalModel_data_size_open_____",
	# "Deliverable/Thalamocortical_size_feedforward", # BIG
	# "ThalamoCorticalModel_data_size_feedforward_cond_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____", # <<<<<<< ISO Coherence, V1 conductance
	# "ThalamoCorticalModel_data_size_feedforward_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_LGNonly_____",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____large",
	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_vsdi_00_radius14_____",
	# "Deliverable/ThalamoCorticalModel_data_size_open_____",

	# # Andrew's machine
	# "/data1/do/ThalamoCorticalModel_data_size_open_____",
	# "/data1/do/ThalamoCorticalModel_data_size_closed_many_____",
	# "/data1/do/ThalamoCorticalModel_data_size_nonoverlapping_many_____",
	# "/data1/do/ThalamoCorticalModel_data_size_overlapping_many_____",

	# "/media/do/DATA/Deliverable/ThalamoCorticalModel_data_orientation_closed_____",
	# "/media/do/DATA/Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____",
	# "/media/do/DATA/Deliverable/ThalamoCorticalModel_data_orientation_open_____",
	# "ThalamoCorticalModel_data_orientation_open_____",

	# "ThalamoCorticalModel_data_xcorr_open_____1", # just one trial
	# "ThalamoCorticalModel_data_xcorr_open_____2deg", # 2 trials
	# "ThalamoCorticalModel_data_xcorr_closed_____2deg", # 2 trials

	# "Deliverable/CombinationParamSearch_LGN_PGN_core",
	# "Deliverable/CombinationParamSearch_LGN_PGN_2",

	# "CombinationParamSearch_large_closed",
	# "CombinationParamSearch_more_focused_closed_nonoverlapping",
	
	# "/media/do/Sauvegarde Système/CombinationParamSearch_closed_overlapping_new",
	# "/media/do/HANGAR/CombinationParamSearch_intact_nonoverlapping",
	]

inac_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "Deliverable/ThalamoCorticalModel_data_size_open_____",

	# "Deliverable/ParamSearch",

	# "CombinationParamSearch_nonoverlapping",

	# "Deliverable/ThalamoCorticalModel_data_size_overlapping_____",
	# "Deliverable/ThalamoCorticalModel_data_size_nonoverlapping_____",
	# "ThalamoCorticalModel_data_size_nonoverlapping_____",
	# "ThalamoCorticalModel_data_size_overlapping_____",

	# "/media/do/Sauvegarde Système/CombinationParamSearch_nonoverlapping",
	# "/media/do/HANGAR/CombinationParamSearch_altered_nonoverlapping",
	]


addon = ""
# sheets = ['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4', 'V1_Inh_L4']
# sheets = ['X_ON', 'X_OFF', 'V1_Exc_L4', 'V1_Inh_L4']
# sheets = ['X_ON', 'X_OFF', 'PGN']
# sheets = ['V1_Exc_L4'] 
# sheets = ['X_ON', 'PGN']
# sheets = [ ['X_ON', 'X_OFF'], 'PGN']
# sheets = [ ['X_ON', 'X_OFF'] ]
# sheets = [ 'X_ON', 'X_OFF', ['X_ON', 'X_OFF'] ]
# sheets = ['X_ON', 'X_OFF', 'V1_Inh_L4']
# sheets = ['X_ON', 'X_OFF']
# sheets = ['X_ON']
# sheets = ['X_OFF'] 
# sheets = ['PGN']
sheets = ['V1_Exc_L4'] 
# sheets = ['V1_Inh_L4'] 
# sheets = ['V1_Exc_L4', 'V1_Inh_L4'] 
# sheets = [ ['V1_Exc_L4', 'V1_Inh_L4'] ]
# sheets = ['V1_Exc_L4', 'V1_Inh_L4', 'X_OFF', 'PGN'] 

print sheets

# ONLY for comparison parameter search
if False: 

	# box = [[-.5,-.5],[.5,.5]] # close to the overlapping
	# box = [[-.5,.0],[.5,.5]] # close to the overlapping
	# box = [[-.5,.0],[.5,.1]] # far from the overlapping
	radius = [.0,.4]
	dist = box if not radius else radius

	# if len(inac_list):
	# 	csvfile = open(inac_list[0]+"/barsizevalues_"+str(sheets[0])+"_dist"+str(dist)+".csv", 'w')
	# else:
	# 	csvfile = open(full_list[0]+"/endinhibitionindex_"+str(sheets[0])+"_dist"+str(dist)+".csv", 'w')

	# for i,l in enumerate(full_list):
	# 	# for parameter search
	# 	full = [ l+"/"+f for f in sorted(os.listdir(l)) if os.path.isdir(os.path.join(l, f)) ]
	# 	if len(inac_list):
	# 		large = [ inac_list[i]+"/"+f for f in sorted(os.listdir(inac_list[i])) if os.path.isdir(os.path.join(inac_list[i], f)) ]

	# 	for i,f in enumerate(full):
	# 		print i,f

	# 		color = "black"
	# 		if "open" in f:
	# 			color = "grey"
	# 		if "closed" in f:
	# 			color = "black"
	# 		if "Kimura" in f:
	# 			color = "#CCCC55"
	# 		if "LGNonly" in f:
	# 			color = "#FFEE33"

	# 		for s in sheets:

	# 			if "open" in f and 'PGN' in s:
	# 				color = "#11AA99"
	# 			if "closed" in f and 'PGN' in s:
	# 				color = "#66AA55"

	# 			print color

	# 			if len(inac_list):
	# 				size_tuning_comparison( 
	# 					sheet=s, 
	# 					folder_full=f, 
	# 					folder_inactive=large[i],
	# 					stimulus="DriftingSinusoidalGratingDisk",
	# 					parameter='radius',
	# 					# box = box,
	# 					radius = radius,
	# 					csvfile = csvfile
	# 					# , plotAll = True # plot all barplots per folder?
	# 				)
	# 			else:
	# 				size_tuning_index( 
	# 					sheet=s, 
	# 					folder_full=f, 
	# 					stimulus="DriftingSinusoidalGratingDisk",
	# 					parameter='radius',
	# 					# box = box,
	# 					radius = radius,
	# 					csvfile = csvfile
	# 				)

	# 			csvfile.write("\n")

	# # plot map
	# csvfile.close()


	###############################
	# directory = "CombinationParamSearch_more_focused_nonoverlapping"
	# xvalues = [70, 80, 90, 100, 110]
	# yvalues = [130, 140, 150, 160, 170]
	# ticks = [0,1,2,3,4]

	# directory = "CombinationParamSearch_large_nonoverlapping"
	# xvalues = [30, 50, 70, 90]
	# yvalues = [150, 200, 250, 300]
	# ticks = [0,1,2,3]

	# xvalues = [5, 75, 200]
	# yvalues = [10, 150, 300]
	# ticks = [0,1,2]
	xvalues = [25, 50, 75, 100, 125]
	yvalues = [100, 130, 150, 170, 200]
	ticks = [0,1,2,3,4]

	# comparison_tuning_map(directory, xvalues, yvalues, ticks)
	# comparison_tuning_map(inac_list[0], xvalues, yvalues, ticks)
	# comparison_size_tuning_map(inac_list[0]+"/barsizevalues_X_ON_dist[0.2, 0.6].csv", xvalues, yvalues, ticks)

	comparison_size_tuning_map(inac_list[0]+"/endinhibitionindex_"+str(sheets[0])+"_dist"+str(dist)+"1.csv", xvalues, yvalues, ticks)
	# comparison_tuning_map(inac_list[0], xvalues, yvalues, ticks)


else:

	for i,f in enumerate(full_list):
		print i,f

		# color = "saddlebrown"
		color = "black"
		color_data = "black"
		if "feedforward" in f:
			# addon = "feedforward"
			addon = "feedforward"
			closed = False
			color = "grey"
			color_data = "grey"
			fit = "gamma"
			# color = "red"
			trials = 12
		if "open" in f:
			addon = "open"
			closed = False
			color = "grey"
			color_data = "grey"
		if "closed" in f:
			addon = "closed"
			closed = True
			fit = "bimodal"
			trials = 6
		if "Kimura" in f:
			color = "#CCCC55" # grey-yellow
		if "LGNonly" in f:
			color = "#FFEE33" # yellow

		for s in sheets:

			if 'PGN' in s:
				arborization = 300
				color = "#66AA55" # pale green
				if "open" in f or "feedforward" in f:
					addon = "open"
					color = "#11AA99" # aqua
				if "closed" in f:
					addon = "closed"
					color = "#66AA55"
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
			# 	useXlog=True,
			# 	addon = addon,
			# )
			# response_barplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="Null",
			# 	parameter='background_luminance',
			# 	num_stim = 4,
			# 	max_stim = 35.,
			# 	radius = [0., 5.], 
			# 	# xlabel="Control",
			# 	xlabel="Cooled",
			# 	data_marker = "s",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_closed.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_open.csv",
			# )
			# response_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="Null",
			# 	parameter='background_luminance',
			# 	start=100., 
			# 	end=2000., 
			# 	radius = [0., 5.], 
			# 	closed=closed,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_"+addon+".csv",
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='Null',
			# 	parameter="background_luminance",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
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
			# 	percentile=False,
			# 	# ylim=[0,35],
			# 	addon = addon,
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
			# 	radius=[0., 5.],
			# 	percentile=False,
			# 	ylim=[0,10],
			# 	useXlog=False, 
			# 	addon = addon,
			# 	inputoutputratio = False,
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
			# response_barplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='contrast',
			# 	num_stim = 13,
			# 	max_stim = 60.,
			# 	radius = [0., 5.], 
			# 	# xlabel="Control",
			# 	xlabel="",
			# 	data_marker = "s",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_c50_control.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_c50_led.csv",
			# )
			# response_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='contrast',
			# 	start=100., 
			# 	end=2000., 
			# 	radius = [0., 5.], 
			# 	closed=closed,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_c50_control.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_c50_led.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/PrzybyszewskiGaskaFootePollen2000_1b_cooled.csv",
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="contrast",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
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
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="temporal_frequency",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
			# )
			# response_barplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='temporal_frequency',
			# 	num_stim = 8,
			# 	max_stim = 30.,
			# 	radius = [0., 5.], 
			# 	# xlabel="Control",
			# 	xlabel="LED",
			# 	data_marker = "s",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_control.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_led.csv",
			# )
			# response_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='temporal_frequency',
			# 	start=100., 
			# 	end=2000., 
			# 	radius = [0., 5.], 
			# 	closed=closed,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_control.csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_led.csv",
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
			# 	addon = addon,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=True, 
			# 	addon = addon,
			# )
			# trial_averaged_connectivity_count( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	incoming=False, 
			# 	box=None, 
			# 	radius = [.0,.5], # center
			# 	addon = addon
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="spatial_frequency",
			# 	percentile=False,
			# 	ylim=[0,20],
			# 	useXlog=True, 
			# 	radius = [.0, 1.7], # center
			# 	addon = addon,
			# 	inputoutputratio = False,
			# )
			# response_barplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='spatial_frequency',
			# 	num_stim = 10,
			# 	max_stim = 1.,
			# 	radius = [0., 5.], 
			# 	xlabel="Control",
			# 	data_marker = "s",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_control.csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_led.csv",
			# )
			# response_boxplot( 
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	parameter='spatial_frequency',
			# 	start=100., 
			# 	end=2000., 
			# 	radius = [0., 5.], 
			# 	closed=closed,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_control.csv",
			#  	data="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_led.csv",
			# )


			# SIZE
			# Ex: ThalamoCorticalModel_data_size_V1_full_____
			# Ex: ThalamoCorticalModel_data_size_open_____
			# end_inhibition_barplot(
			# 	sheet=s, 
			# 	folder=f, 
			# 	stimulus="DriftingSinusoidalGratingDisk",
			# 	parameter='radius',
			# 	start=100., 
			# 	end=1000., 
			# 	xlabel="exp",
			# 	ylabel="Index of end-inhibition",
			# 	closed=closed,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_"+addon+".csv",
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3E.csv", # closed drifting gratings
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3F.csv", # retinal drifting gratings
			# 	radius = [.0,.7], # center
			# 	addon = addon,
			# )
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
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/MurphySillito1987_"+addon+".csv",
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3E.csv", # closed drifting gratings
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey_3F.csv", # retinal drifting gratings
			# 	opposite=False, # to select cortical cells with SAME orientation preference
			# 	# box = [[-.5,-.5],[.5,.5]], # center
			# 	radius = [.0,.7], # center
			# 	addon = addon,
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
			# 	addon = "center_iso_"+addon,
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
			# 	addon = "center_ortho_"+addon,
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
			# 	radius = [1.,4.], # surround
			# 	addon = "surround_iso_"+addon,
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
			# 	radius = [1.,4.], # surround
			# 	addon = "surround_ortho_" + addon,
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
			# 	useXlog=True, 
			# 	useYlog=False, 
			# 	percentile=False,
			# 	# ylim=[0,50],
			# 	opposite=False, # to select cells with SAME orientation preference
			# 	radius = [.0,.7], # center
			# 	addon = addon,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
			# 	# data_curve=False,
			# )
			# LHI( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	# radius = [.0, 0.7], # center
			# 	addon = addon,
			# )
			# SynergyIndex_spikes( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	radius = [.0, 1.3], 
			# 	preferred = False,
			# 	addon = addon,
			# )
			# SynergyIndex_spikes( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	radius = [.0, 1.3],
			# 	preferred = True,
			# 	addon = addon,
			# )
			# SynergyIndex_spikes( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	radius = [1.4, 3.], # surround
			# 	preferred = False,
			# 	addon = addon + "_surround_",
			# )
			# SynergyIndex_spikes( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	radius = [1.4, 3.], # surround
			# 	preferred = True,
			# 	addon = addon + "_surround_",
			# )
			# Xcorr_SynergyIndex_spikes( 
			# 	sheet1=s, 
			# 	folder1=f,
			# 	# sheet2=s, 
			# 	folder2=f,
			# 	sheet2='V1_Inh_L4', 
			# 	# sheet2='V1_Exc_L4', 
			# 	# folder2="ThalamoCorticalModel_data_size_feedforward_vsdi_100micron_____",
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	addon = addon,
			# )
			# trial_averaged_raster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [.0, 1.4], # center
			# 	addon = addon,
			# )
			# OrientedRaster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	radius = [.0, 1.2], # center
			# 	parameter="radius",
			# 	addon = addon,
			# )
			# VSDI( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	# radius = [.0, 0.7], # center
			# 	addon = addon,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
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
			# 	radius = [.0,.7], # center
			# 	addon = "center_iso_" + addon,
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
			# 	percentile=True,
			# 	ylim=[0,50],
			# 	opposite=True, # to select cortical cells with OPPOSITE orientation preference
			# 	radius = [.0,.7], # center
			# 	addon = "center_ortho_" + addon,
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
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_iso_" + addon,
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
			# 	opposite=True, #
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_ortho_" + addon,
			# )

			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=True, 
			# 	# radius = [.0, 1.7], # center
			# 	radius = [.0, .7], # more center
			# 	addon = addon,
			# 	inputoutputratio = True,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_iso_" + addon,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_iso_" + addon,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	opposite=True, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_ortho_" + addon,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	opposite=True, #
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_ortho_" + addon,
			# )

			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1='X_ON', 
			# 	sheet2='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1='X_ON', 
			# 	sheet2='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1='X_OFF', 
			# 	sheet2='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1='X_OFF', 
			# 	sheet2='PGN', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_" + addon,
			# )

			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1=s, 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_iso_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1=s, 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_iso_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1=s, 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	opposite=True, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "center_ortho_" + addon,
			# )
			# trial_averaged_ratio_tuning_curve( 
			# 	sheet1=s, 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	opposite=True, #
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_ortho_" + addon,
			# )

			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "iso_center_" + addon,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [1.,1.8], # surround
			# 	# radius = [0.6,1.8], # surround
			# 	addon = "iso_surround_" + addon,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=True, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "ortho_center_" + addon,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=True, #
			# 	radius = [1.,1.8], # surround
			# 	# radius = [0.6,1.8], # surround
			# 	addon = "ortho_surround_" + addon,
			# )

			# trial_averaged_raster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "iso_center_" + addon,
			# )
			# trial_averaged_raster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=False, #
			# 	radius = [1.,1.8], # surround
			# 	# radius = [0.6,1.8], # surround
			# 	addon = "iso_surround_" + addon,
			# )
			# trial_averaged_raster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=True, #
			# 	radius = [.0, 0.7], # center
			# 	addon = "ortho_center_" + addon,
			# )
			# trial_averaged_raster( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	opposite=True, #
			# 	radius = [1.,1.8], # surround
			# 	# radius = [0.6,1.8], # surround
			# 	addon = "ortho_surround_" + addon,
			# )

			# save_positions( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
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

			# # stLFP: iso-surround postsynaptic response to LGN spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Exc_L4', 
			# 	spike_sheet='X_OFF', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-3.,1.],[3.,3.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [.0,.7], # spikes from the center of spike_sheet
			# 	addon = addon + "_surround2center",
			# 	color = color,
			# )
			
			# # stLFP: iso-center postsynaptic response to LGN spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Exc_L4', 
			# 	spike_sheet='X_OFF', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-.6,-.6],[.6,.6]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [.0,.7], # spikes from the center of spike_sheet
			# 	addon = addon + "_center2center",
			# 	color = color,
			# )
			
			# # stLFP: cross-center postsynaptic response to LGN spikes
			
			# # stLFP: LGN postsynaptic response to iso-center spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='X_OFF', 
			# 	spike_sheet='V1_Exc_L4', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-.5,-.5],[.5,.5]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [.0,.7], # spikes from the center of spike_sheet
			# 	addon = addon + "_center",
			# 	color = color,
			# )
			
			# # stLFP: LGN postsynaptic response to iso-surround spikes
						
			# # stLFP: Exc iso-surround postsynaptic response to iso-center spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Exc_L4', 
			# 	spike_sheet='V1_Exc_L4', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-3.,1.],[3.,3.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [.0,.7], # spikes from the center of spike_sheet
			# 	addon = addon + "_center2center",
			# 	color = color,
			# )

			# stLFP: Exc iso-center postsynaptic response to iso-surround spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Exc_L4', 
			# 	spike_sheet='V1_Exc_L4', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-.6,-.6],[.6,.6]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [1.,1.8], # surround
			# 	addon = addon + "_center2surround",
			# 	color = color,
			# )

			# stLFP: Inh iso-center postsynaptic response to iso-surround spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Inh_L4', 
			# 	spike_sheet='V1_Exc_L4', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=False, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-.6,-.6],[.6,.6]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [1.,1.8], # surround
			# 	addon = addon + "_center2surround",
			# 	color = color,
			# )

			# #stLFP: Inh cross-center postsynaptic response to iso-surround spikes
			# SpikeTriggeredAverage(
			# 	lfp_sheet='V1_Inh_L4', 
			# 	spike_sheet='V1_Exc_L4', 
			# 	folder=f, 
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	parameter="radius",
			# 	ylim=[0,50],
			# 	lfp_opposite=True, # ISO
			# 	spike_opposite=False, # ISO
			# 	tip_box = [[-.6,-.6],[.6,.6]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			# 	radius = [1.,1.8], # surround
			# 	addon = addon + "opposite_center2surround",
			# 	color = color,
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
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )

			# spectrum(
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius = [.0,1.4], 
			# 	addon = addon,
			# 	preferred=True, # 
			# 	color = color,
			# )

			# spectrum(
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius = [.0,.9], # center
			# 	addon = "center_iso_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_iso_"+addon,
			# 	preferred=True, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius = [.0,.9], # center
			# 	addon = "center_ortho_"+addon,
			# 	preferred=False, # 
			# 	color = color,
			# )
			# spectrum(
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius = [1.,1.8], # surround
			# 	addon = "surround_ortho_"+addon,
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
			# correlation( 
			# 	sheet1='X_ON', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # 
			# 	preferred2=True, # 
			# 	addon="LGNcenter_2_V1_iso_center_"+addon,
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color,
			# )
			# correlation( 
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='X_ON', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # 
			# 	preferred2=True, # 
			# 	addon="V1_iso_center_2_LGNcenter_"+addon,
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color,
			# )
			# correlation( # CORTICO-CORTICAL 
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [1.,1.8], # surround
			# 	preferred1=True, # ISO
			# 	preferred2=False, # ORTHO
			# 	addon="center_iso_2_surround_ortho_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # CORTICO-CORTICAL 
			# 	sheet1='PGN', 
			# 	sheet2='X_OFF', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [1.,1.8], # center
			# 	radius2 = [.0,.7], # center
			# 	# radius2 = [1.,1.8], # surround
			# 	preferred1=True, # 
			# 	preferred2=True, # 
			# 	addon="surround_2_center_"+addon,
			# 	sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	# sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Surround iso 2 Center iso
			# 	sheet1=s, 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [1.,1.8], # surround
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # SAME
			# 	preferred2=True, # SAME
			# 	addon="surround2center_iso_"+addon,
			# 	sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	color=color
			# )
			# correlation( # Exc center iso 2 Inh center iso
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # ISO
			# 	preferred2=True, # ORTHO
			# 	addon="center_iso_2_center_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Inh center iso 2 Exc center iso
			# 	sheet1='V1_Inh_L4', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # ISO
			# 	preferred2=True, # ORTHO
			# 	addon="center_iso_2_center_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Exc center iso 2 Inh center iso
			# 	sheet1='V1_Exc_L4', 
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # ISO
			# 	preferred2=False, # ORTHO
			# 	addon="center_iso_2_center_ortho_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Inh center iso 2 Exc center iso
			# 	sheet1='V1_Inh_L4', 
			# 	sheet2='V1_Exc_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [.0,.7], # center
			# 	preferred1=False, # ISO
			# 	preferred2=True, # ORTHO
			# 	addon="center_ortho_2_center_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big folder
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Surround iso 2 Center iso
			# 	sheet1='V1_Exc_L4',
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [1.,1.8], # surround
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # SAME
			# 	preferred2=True, # SAME
			# 	addon="surround_iso_2_center_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Center iso 2 Surround iso
			# 	sheet1='V1_Inh_L4', 
			# 	sheet2='V1_Exc_L4',
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [1.,1.8], # surround
			# 	preferred1=True, # SAME
			# 	preferred2=True, # SAME
			# 	addon="center_iso_2_surround_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Exc Surround iso 2 Inh Center ortho
			# 	sheet1='V1_Exc_L4',
			# 	sheet2='V1_Inh_L4', 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [1.,1.8], # surround
			# 	radius2 = [.0,.7], # center
			# 	preferred1=True, # SAME
			# 	preferred2=False, # SAME
			# 	addon="surround_iso_2_center_ortho_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # Inh Center ortho 2 Exc Surround iso
			# 	sheet1='V1_Inh_L4', 
			# 	sheet2='V1_Exc_L4',
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [.0,.7], # center
			# 	radius2 = [1.,1.8], # surround
			# 	preferred1=False, # ORTHO
			# 	preferred2=True, # SAME
			# 	addon="center_ortho_2_surround_iso_"+addon,
			# 	# sizes=[0.125, 0.187, 0.280, 0.419, 0.627, 0.939, 1.405, 2.103, 3.148, 4.711], # big
			# 	sizes=[0.125, 0.164, 0.214, 0.281, 0.368, 0.482, 0.631, 0.826, 1.082, 1.417, 1.856, 2.431, 3.184, 4.171, 5.462], # deliverable folder
			# 	color=color
			# )
			# correlation( # OPPOSITE-OPPOSITE
			# 	sheet1=s, 
			# 	sheet2=s, 
			# 	folder=f,
			# 	stimulus='DriftingSinusoidalGratingDisk',
			# 	stimulus_parameter='radius',
			# 	radius1 = [1.,1.8], # surround
			# 	radius2 = [.0,.7], # center
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
			# 	radius = [1.,1.8], # surround
			# 	opposite=True, # ORTHO
			# 	addon = "surround_ortho_"+addon,
			# 	ylim=[0.,1.]
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
			# 	# ylim=[0,50],
			# 	# opposite=False, # to select cortical cells with SAME orientation preference
			# 	# xlim=[-numpy.pi/2,numpy.pi/2],
			# 	radius = [.0,.7], # center
			# 	addon = addon,
			# 	# data="/home/do/Dropbox/PhD/LGN_data/deliverable/AlittoUsrey2008_6AC_fit.csv",
			# 	# data_curve=False,
			# )
			# trial_averaged_Vm( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	opposite=False, #
			# 	radius = [.0, 0.7], # center
			# 	addon = addon,
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
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	percentile=False,
			# 	ylim=[0,30],
			# 	useXlog=False, 
			# 	radius = [.0, 8.7], # center
			# 	addon = "center_" + addon,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	radius=[0., 1.],
			# 	percentile=False,
			# 	ylim=[-10,50],
			# 	useXlog=False, 
			# 	addon = addon,
			# 	inputoutputratio = False,
			# )
			# trial_averaged_conductance_timecourse( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	ticks=[0.0, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.],
			# 	ylim=[0,20],
			# 	# box = [[-.5,-.5],[.5,.5]], # CENTER
			# 	addon = "center",
			# 	# box = [[-.5,.0],[.5,.8]], # mixed surround (more likely to be influenced by the recorded thalamus)
			# 	# box = [[-.5,.5],[.5,1.]], # strict surround
			# 	# box = [[-0.1,.6],[.3,1.]], # surround
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
			# 	radius = [.5,.8], # center
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
			# 	closed=closed,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_"+addon+".csv",
			# 	radius = [.0,.7], # center
			# 	addon = addon,
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
			# 	closed=closed,
			# 	data="/home/do/Dropbox/PhD/LGN_data/deliverable/VidyasagarUrbas1982_"+addon+".csv",
			# 	# percentage=True,
			# )
			# trial_averaged_conductance_tuning_curve( 
			# 	sheet=s, 
			# 	folder=f,
			# 	stimulus='FullfieldDriftingSinusoidalGrating',
			# 	parameter="orientation",
			# 	percentile=False,
			# 	ylim=[0,20],
			# 	useXlog=True, 
			# 	radius = [.0, 1.7], # center
			# 	addon = addon,
			# 	inputoutputratio = False,
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
			Xcorr_SynergyIndex_spikes( 
				sheet1=s, 
				folder1=f,
				# sheet2=s, 
				folder2=f,
				sheet2='V1_Inh_L4', 
				# sheet2='V1_Exc_L4', 
				# folder2="ThalamoCorticalModel_data_size_feedforward_vsdi_100micron_____",
				stimulus='FullfieldDriftingSinusoidalGrating',
				parameter="orientation",
				addon = addon,
			)

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

				# ONGOING ACTIVITY
				#Ex: ThalamoCorticalModel_data_luminance_closed_____ vs ThalamoCorticalModel_data_luminance_open_____
				# pairwise_scatterplot( 
				# 	sheet=s,
				# 	# sheet=['X_ON', 'X_OFF'], # s,
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="Null",
				# 	# stimulus_band=6, # 1 cd/m2 as in WaleszczykBekiszWrobel2005
				# 	# stimulus_band=7, # 10 cd/m2 
				# 	stimulus_band=8, # 100 cd/m2 
				# 	parameter='background_luminance',
				# 	start=100., 
				# 	end=2000., 
				# 	xlabel="closed-loop ongoing activity (spikes/s)",
				# 	ylabel="open-loop ongoing activity (spikes/s)",
				# 	withRegression=True,
				# 	withCorrCoef=True,
				# 	withCentroid=True,
				# 	xlim=[0,5],
				# 	ylim=[0,5],
				# 	data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_closed.csv",
				# 	data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/WaleszczykBekiszWrobel2005_4A_open.csv",
				# 	data_marker = "D",
				# )

				# # SPATIAL FREQUENCY
				# # Ex: ThalamoCorticalModel_data_spatial_V1_full_____ vs ThalamoCorticalModel_data_spatial_Kimura_____
				# pairwise_scatterplot( 
				# 	sheet=s, 
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="FullfieldDriftingSinusoidalGrating",
				# 	parameter='spatial_frequency',
				# 	start=100., 
				# 	end=10000.,
				# 	radius = [0., 5.], 
				# 	xlabel="Control",
				# 	ylabel="cortex-off",
				# 	xlim=[0,1.5],
				# 	ylim=[0,1.5],
				# 	withRegression=False,
				# 	withCorrCoef=False,
				# 	withCentroid=True,
				# 	data_marker = "s",
				# 	data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_control.csv",
				# 	data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_SF_led.csv",
				# 	# data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_closed.csv",
				# 	# data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_open.csv",
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
				# pairwise_scatterplot( 
				# 	sheet=s, 
				# 	folder_full=f, 
				# 	folder_inactive=l,
				# 	stimulus="FullfieldDriftingSinusoidalGrating",
				# 	parameter='temporal_frequency',
				# 	start=100., 
				# 	end=10000., 
				# 	radius = [0., 5.], 
				# 	xlabel="Control",
				# 	ylabel="cortex-off",
				# 	xlim=[0,15],
				# 	ylim=[0,15],
				# 	withRegression=False,
				# 	withCorrCoef=False,
				# 	withCentroid=True,
				# 	data_marker = "s", #"^",
				# 	data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_control.csv",
				# 	data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/HasseBriggs2017_5_TF_led.csv",
				# 	# data_full="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_closed.csv",
				# 	# data_inac="/home/do/Dropbox/PhD/LGN_data/deliverable/KimuraShimegiHaraOkamotoSato2013_2A_open.csv",
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

# data_significance(
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/ongoing_closed.csv",
# 	"/home/do/Dropbox/PhD/LGN_data/deliverable/ongoing_open.csv",
# 	'anova'
# )

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

