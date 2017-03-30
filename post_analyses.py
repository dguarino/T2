# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import os
import ast

from functools import reduce # forward compatibility
import operator

import gc
import numpy
import scipy.stats
import scipy
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

from mozaik.tools.mozaik_parametrized import colapse

import neo



# neo spiktrains have:
# - fano factor
# - isi




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




def get_per_neuron_spike_count( datastore, stimulus, sheet, start, end, stimulus_parameter, bin=10.0, neurons=[] ):

	# if number of ADS==0:
	TrialAveragedFiringRateCutout( 
		param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
		ParameterSet({}) 
	).analyse(start=start, end=end)

	SpikeCount( 
		param_filter_query(datastore, sheet_name=sheet, st_name=stimulus), 
		ParameterSet({'bin_length' : bin }) 
	).analyse()

	dsv = param_filter_query( datastore, identifier='PerNeuronValue', sheet_name=sheet, st_name=stimulus )
	dsv.print_content(full_recordings=False)
	pnvs = [ dsv.get_analysis_result() ]
	# get stimuli
	st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs[-1]]

	if not len(neurons)>1:
		spike_ids = param_filter_query(datastore, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = datastore.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		neurons = datastore.get_sheet_ids(sheet_name=sheet, indexes=sheet_ids)
		print "neurons", len(neurons)

	dic = colapse_to_dictionary([z.get_value_by_id(neurons) for z in pnvs[-1]], st, stimulus_parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))
	print dic

	mean_rates = dic.values()[0][1]
	stimuli = dic.values()[0][0]

	return mean_rates, stimuli




def mean_confidence_interval(data, confidence=0.95):
	import scipy.stats
	a = 1.0*numpy.array(data)
	n = len(a)
	m, se = numpy.mean(a, axis=0), scipy.stats.sem(a, axis=0)
	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
	return m, m-h, m+h




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
	plt.savefig( folder+"/Covariogram_"+stimulus+"_"+sheet+".png", dpi=200 )
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
		plt.savefig( folder+"/Covariogram_"+stimulus+"_"+sheet+"_"+str(i)+".png", dpi=200 )
		fig.clf()
		plt.close()
		gc.collect()




def perform_end_inhibition_barplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="" ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter )
	print rates.shape # (stimuli, cells)
	#print rates[0][10]
	print stimuli

	# END-INHIBITION as in MurphySillito1987:
	# For each cell
	# 1. find the peak response stimulus value
	peaks = numpy.amax(rates, axis=0) # print peaks
	# 2. compute average of plateau response (before-last 3: [1.55, 2.35, 3.59])
	plateaus = numpy.mean( rates[6:9], axis=0) # print plateaus
	# 3. compute the percentage difference from peak: (peak-plateau)/peak *100
	ends = (peaks-plateaus)/peaks *100 # print ends
	# 4. group cells by end-inhibition
	hist, bin_edges = numpy.histogram( ends, bins=10 )
	mean = ends.mean()
	print mean
	hist = hist[::-1] # reversed
	bin_edges = bin_edges[::-1]
	print hist
	print bin_edges

	# PLOTTING
	width = bin_edges[1] - bin_edges[0]
	center = (bin_edges[:-1] + bin_edges[1:]) / 2
	fig,ax = plt.subplots()
	barlist = ax.bar(center, hist, align='center', width=width, facecolor='white')
	ax.plot((mean, mean), (0,70), 'k:', linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	barlist[0].set_color('k')
	barlist[1].set_color('k')
	barlist[2].set_color('k')
	barlist[3].set_color('k')
	barlist[4].set_color('k')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([bin_edges[0], bin_edges[-1], 0, 70])
	plt.xticks(bin_edges, (10,9,8,7,6,5,4,3,2,1))
	plt.savefig( folder+"/suppression_index_"+sheet+".png", dpi=200 )
	plt.close()
	# garbage
	gc.collect()




def perform_orientation_bias_barplot( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="" ):
	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter )
	print rates.shape # (stimuli, cells)
	#print rates[0][10]
	print stimuli

	# ORIENTATION BIAS as in VidyasagarUrbas1982:
	# For each cell
	# 1. find the peak response (optimal orientation)
	peaks = numpy.amax(rates, axis=0) # print peaks
	# 2. find the least response 
	trough = numpy.mean( rates[6:9], axis=0) # (mean of non-optimal orientations)
	# 3. compute the percentage difference from peak
	bias = (peaks-trough)/peaks *100 # print ends
	# 4. group cells by orientation bias
	hist, bin_edges = numpy.histogram( bias, bins=10 )
	mean = bias.mean()
	print mean
	print hist
	print bin_edges

	# PLOTTING
	width = bin_edges[1] - bin_edges[0]
	center = (bin_edges[:-1] + bin_edges[1:]) / 2
	fig,ax = plt.subplots()
	barlist = ax.bar(center, hist, align='center', width=width, facecolor='white', hatch='/')
	ax.plot((mean, mean), (0,130), 'k:', linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([bin_edges[0], bin_edges[-1], 0, 130])
	plt.xticks(bin_edges, (1,2,3,4,5,6,7,8,9,10))
	plt.savefig( folder+"/orientation_bias_"+sheet+".png", dpi=200 )
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.], box=False ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons =[]
	if box:
		spike_ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=spike_ids)
		positions = data_store.get_neuron_postions()[sheet]
		box1 = [[-.5,-.5],[.5,.5]]
		ids1 = select_ids_by_position(positions, sheet_ids, box=box1)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, neurons=neurons )
	# print rates

	# compute per-trial mean rate over cells
	mean_rates = numpy.mean(rates, axis=1) 
	# print "Ex. collapsed_mean_rates: ", mean_rates.shape
	# print "Ex. collapsed_mean_rates: ", mean_rates
	std_rates = numpy.std(rates, axis=1, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	# print "Ex. collapsed_std_rates: ", std_rates

	# final_mean_rates = numpy.mean(collapsed_mean_rates) 
	# final_std_rates = numpy.mean(collapsed_std_rates) # mean std
	# print "stimuli: ", stimuli
	# print "final means and stds: ", mean_rates, std_rates
	print sorted( zip(stimuli, mean_rates, std_rates) )
	final_sorted = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, mean_rates, std_rates) ) ) ]

	if percentile:
		firing_max = numpy.amax( final_sorted[1] )
		final_sorted[1] = final_sorted[1] / firing_max * 100

	# Plotting tuning curve
	fig,ax = plt.subplots()

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

	if not percentile:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim([0,100+numpy.amax(final_sorted[2])])

	# text
	ax.set_title( sheet )
	ax.set_xlabel( xlabel )
	if percentile:
		ylabel = "Percentile " + ylabel
		sheet = sheet + "_percentile"
	ax.set_ylabel( ylabel )
	# ax.legend( loc="lower right", shadow=False )
	# plt.show()
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+stimulus+"_"+parameter+"_"+sheet+".png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def perform_pairwise_scatterplot( sheet, folder_full, folder_inactive, stimulus, stimulus_band, parameter, start, end, xlabel="", ylabel="", withRegression=True, withCorrCoef=True, withCentroid=False, withLowPassIndex=False, highest=3 ):
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


	x_full_rates, stimuli_full = get_per_neuron_spike_count( data_store_full, stimulus, sheet, start, end, parameter )
	x_inac_rates, stimuli_inac = get_per_neuron_spike_count( data_store_inac, stimulus, sheet, start, end, parameter )

	x_full = x_full_rates[stimulus_band]
	# print "x_full: ", x_full.shape, x_full
	x_inac = x_inac_rates[stimulus_band]
	#!!!!!!!!!!!!! only for spontaneous test
	# x_inac = x_inac - numpy.random.uniform(low=0.0, high=1., size=(len(x_full)) ) 

	if withLowPassIndex:
		# normalized
		normalized_full = x_full / numpy.amax( x_full )
		normalized_inac = x_inac / numpy.amax( x_inac )
		# highest spatial frequency response is for [4]
		x_full_highest = x_full_rates[highest] 
		x_inac_highest = x_inac_rates[highest]
		# normalized highest
		normalized_full_highest = x_full_highest / numpy.amax( x_full_highest )
		normalized_inac_highest = x_inac_highest / numpy.amax( x_inac_highest )
		# Low-pass Index
		_x_full = normalized_full / normalized_full_highest
		_x_inac = normalized_inac / normalized_inac_highest
		# re-normalized
		x_full = _x_full / numpy.amax( _x_full )
		x_inac = _x_inac / numpy.amax( _x_inac )

	# PLOTTING
	fig,ax = plt.subplots()
	ax.scatter( x_full, x_inac, marker="D", facecolor="k", edgecolor="k", label=sheet )
	x0,x1 = ax.get_xlim()
	y0,y1 = ax.get_ylim()
	# to make it squared
	if x1 >= y1:
		y1 = x1
	else:
		x1 = y1
	if withLowPassIndex:
		x0 = y0 = 0.
		x1 = y1 = 1.

	ax.set_xlim( (x0,x1) )
	ax.set_ylim( (y0,y1) )
	ax.set_aspect( abs(x1-x0)/abs(y1-y0) )
	# add diagonal
	ax.plot( [x0,x1], [y0,y1], linestyle='--', color="k" )

	if withRegression:
		# add regression line
		m,b = numpy.polyfit(x_full,x_inac, 1)
		print "fitted line:", m, "x +", b
		x = numpy.arange(x0, x1)
		ax.plot(x, m*x+b, 'k-')

	if withCorrCoef:
		# add correlation coefficient
		corr = numpy.corrcoef(x_full,x_inac)
		sheet = sheet + " r=" + '{:.3f}'.format(corr[0][1])

	if withCentroid:
		cx = x_full.sum()/len(x_full)
		cy = x_inac.sum()/len(x_inac)
		print "cx", cx, "cy", cy
		ax.plot(cx, cy, 'k+', markersize=12, markeredgewidth=3)

	from scipy.stats import chi2_contingency
	obs = [ x_inac, x_full ]
	chi2, p, dof, ex = chi2_contingency( obs, correction=False )
	print "Scatter chi2:", chi2, "p:", p

	# text
	ax.set_title( sheet )
	ax.set_xlabel( xlabel )
	ax.set_ylabel( ylabel )
	ax.legend( loc="lower right", shadow=False, scatterpoints=1 )
	# plt.show()
	plt.savefig( folder_inactive+"/TrialAveragedPairwiseScatter_"+parameter+"_"+sheet+".png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def trial_averaged_conductance_tuning_curve( sheet, folder, stimulus, parameter, ticks, percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.] ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	analog_ids = sorted( param_filter_query(data_store,sheet_name=sheet).get_segments()[0].get_stored_vm_ids() )

	# NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	# l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]
	# l4_exc_or_many = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < 0.1)[0]]

	# position_V1 = data_store.get_neuron_postions()['V1_Exc_L4']
	# V1_sheet_ids = data_store.get_sheet_indexes(sheet_name='V1_Exc_L4',neuron_ids=l4_exc_or_many)
	# print "# of V1 cells within radius range having orientation close to 0:", len(V1_sheet_ids)
	# radius_V1_ids = select_ids_by_position(position_V1, V1_sheet_ids, radius=[0,.5])
	# radius_V1_ids = data_store.get_sheet_ids(sheet_name='V1_Exc_L4',indexes=radius_V1_ids)
	# analog_ids = radius_V1_ids
	print "analog_ids: ",analog_ids

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

	# if percentile:
	firing_max = numpy.amax( final_sorted_e[1] )
	final_sorted_e[1] = final_sorted_e[1] / firing_max * 100
	firing_max = numpy.amax( final_sorted_i[1] )
	final_sorted_i[1] = final_sorted_i[1] / firing_max * 100

	# Plotting tuning curve
	fig,ax = plt.subplots()
	ax.plot( final_sorted_e[0], final_sorted_e[1], color='red', linewidth=2 )
	ax.plot( final_sorted_i[0], final_sorted_i[1], color='blue', linewidth=2 )

	err_max = final_sorted_e[1] + final_sorted_e[2]
	err_min = final_sorted_e[1] - final_sorted_e[2]
	ax.fill_between(final_sorted_e[0], err_max, err_min, color='red', alpha=0.3)
	err_max = final_sorted_i[1] + final_sorted_i[2]
	err_min = final_sorted_i[1] - final_sorted_i[2]
	ax.fill_between(final_sorted_i[0], err_max, err_min, color='blue', alpha=0.3)

	# if not percentile:
	# 	ax.set_ylim(ylim)
	# else:
	# 	ax.set_ylim([0,numpy.amax(final_sorted_e[2])])

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
	ax.set_title( "PSP Area response plot V1excL4" )
	ax.set_xlabel( 'radius' )
	ax.set_ylabel( "PSP (%)" )
	plt.savefig( folder+"/TrialAveragedPSP_"+sheet+"_"+parameter+"_pop.png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def chisquared_test(observed, expected):

	# chi2 = numpy.sum( (observed-expected)**2 / expected )
	# p = 0.0

	from scipy.stats import chi2_contingency
	obs = [ observed, expected ]
	chi2, p, dof, ex = chi2_contingency( obs, correction=True )
	# print ex

	# from scipy.stats import chisquare
	# chi2, p = chisquare( f_obs=observed, f_exp=expected )

	print "chi2:", chi2, "p-value:", p




###################################################
# Execution

full_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_closed_____",

	# "ThalamoCorticalModel_data_contrast_____.0012",
	# "ThalamoCorticalModel_data_contrast_V1_full_____",

	# "Deliverable/ThalamoCorticalModel_data_temporal_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_temporal_open_____",

	# "Deliverable/ThalamoCorticalModel_data_contrast_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_contrast_open_____",

	# "Deliverable/ThalamoCorticalModel_data_size_feedforward_____",
	# "Deliverable/ThalamoCorticalModel_data_size_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_size_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_spatial_open_____",

	"Deliverable/ThalamoCorticalModel_data_orientation_feedforward_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_closed_____",
	# "Deliverable/ThalamoCorticalModel_data_orientation_open_____",

	# "ThalamoCorticalModel_data_xcorr_open_____1", # just one trial
	# "ThalamoCorticalModel_data_xcorr_open_____2deg", # 2 trials
	# "ThalamoCorticalModel_data_xcorr_closed_____2deg", # 2 trials
	]

inac_list = [ 
	# "Deliverable/ThalamoCorticalModel_data_luminance_open_____",

	# "Deliverable/ThalamoCorticalModel_data_spatial_Kimura_____",

	# "Deliverable/ThalamoCorticalModel_data_size_open_____",
	]



# sheets = ['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']
# sheets = ['X_ON', 'X_OFF']
# sheets = ['X_ON']
# sheets = ['X_OFF'] 
# sheets = ['PGN']
sheets = ['V1_Exc_L4'] 


for i,f in enumerate(full_list):
	print i,f

	for s in sheets:

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

		# # CONTRAST
		# trial_averaged_tuning_curve_errorbar( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='FullfieldDriftingSinusoidalGrating',
		# 	parameter="contrast",
		# 	start=100., 
		# 	end=10000., 
		# 	xlabel="contrast", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color="black", 
		# 	percentile=True 
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
		# 	percentile=False 
		# )

		# # SPATIAL
		# trial_averaged_tuning_curve_errorbar( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='FullfieldDriftingSinusoidalGrating',
		# 	# parameter="spatial_frequency",
		# 	parameter="temporal_frequency",
		# 	start=100., 
		# 	end=10000., 
		# 	# xlabel="Spatial frequency", 
		# 	xlabel="Temporal frequency", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color="black", 
		# 	useXlog=True, 
		# 	useYlog=True, 
		# 	percentile=False 
		# )

		# SIZE
		# Ex: ThalamoCorticalModel_data_size_V1_full_____
		# Ex: ThalamoCorticalModel_data_size_open_____
		# perform_end_inhibition_barplot( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus="DriftingSinusoidalGratingDisk",
		# 	parameter='radius',
		# 	start=100., 
		# 	end=1000., 
		# 	xlabel="Index of end-inhibition",
		# 	ylabel="Number of cells",
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
		# 	color="black", 
		# 	# color="red", 
		# 	useXlog=False, 
		# 	useYlog=False, 
		# 	percentile=True,
		# 	ylim=[0,20],
		# 	box=True
		# )
		# trial_averaged_conductance_tuning_curve( 
		# 	sheet=s, 
		# 	folder=f
		# 	stimulus='DriftingSinusoidalGratingDisk',
		# 	parameter="radius",
		# 	#       0      1     2     3     4     5     6     7     8     9
		# 	ticks=[0.125, 0.19, 0.29, 0.44, 0.67, 1.02, 1.55, 2.36, 3.59, 5.46],
		# 	# ylim=[0,6]
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
		# 	end=2000., 
		# 	xlabel="Orientation", 
		# 	ylabel="firing rate (sp/s)", 
		# 	# color="black", 
		# 	color="red", 
		# 	useXlog=False, 
		# 	useYlog=False, 
		# 	percentile=False,
		# 	ylim=[0,50]
		# )
		# perform_orientation_bias_barplot( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus="FullfieldDriftingSinusoidalGrating",
		# 	parameter='orientation',
		# 	start=100., 
		# 	end=2000., 
		# 	xlabel="Orientation bias",
		# 	ylabel="Number of cells"
		# )
		trial_averaged_conductance_tuning_curve( 
			sheet=s, 
			folder=f,
			stimulus='FullfieldDriftingSinusoidalGrating',
			parameter='orientation',
			ticks=[0.0, 0.314, 0.628, 0.942, 1.256, 1.570, 1.884, 2.199, 2.513, 2.827],
			# ylim=[0,6]
		)

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

			# # SPONTANEOUS ACTIVITY
			# #Ex: ThalamoCorticalModel_data_luminance_V1_fake_____ vs ThalamoCorticalModel_data_luminance_____
			# perform_pairwise_scatterplot( 
			# 	sheet=s, 
			# 	folder_full=f, 
			# 	folder_inactive=l,
			# 	stimulus="Null",
			# 	stimulus_band=0, #0, # OFF   #8, # ON
			# 	parameter='background_luminance',
			# 	start=100., 
			# 	end=2000., 
			# 	xlabel="spontaneous activity before cooling (spikes/s)",
			# 	ylabel="spontaneous activity during cooling (spikes/s)",
			# 	withRegression=True,
			# 	withCorrCoef=True,
			# 	withCentroid=False
			# )

			# # SPATIAL FREQUENCY
			# # Ex: ThalamoCorticalModel_data_spatial_V1_full_____ vs ThalamoCorticalModel_data_spatial_Kimura_____
			# perform_pairwise_scatterplot( 
			# 	sheet=s, 
			# 	folder_full=f, 
			# 	folder_inactive=l,
			# 	stimulus="FullfieldDriftingSinusoidalGrating",
			# 	stimulus_band=1, # 0, 1
			# 	parameter='spatial_frequency',
			# 	start=100., 
			# 	end=10000., 
			# 	xlabel="Control",
			# 	ylabel="PGN Inactivated",
			# 	withRegression=False,
			# 	withCorrCoef=False,
			# 	withCentroid=True,
			# 	withLowPassIndex=True,
			# 	highest=3
			# )




# # STATISTICAL SIGNIFICANCE
# # For binned histograms, Chi-square distance is the most used method to compare observed and expected data.

# # SIZE
# print
# print "Size statistical significance"
# print "chi-square test of null-hypothesis: 'No difference between open and closed loop condition' ..."
# data_size_open = numpy.array( [0, 0, 5, 1, 10, 19, 8, 7, 3, 3] ) # expected open
# data_size_closed = numpy.array( [9, 9, 9, 12, 17, 7, 3, 1, 1, 0] ) # expected closed
# model_size_open = numpy.array( [2, 0.00000000001, 0.0000000001, 1, 6, 5, 11, 21, 66, 21] ) # observed open
# model_size_closed = numpy.array( [6, 3, 11, 11, 11, 10, 13, 28, 25, 15] ) # observed closed
# # size contingency of closed vs open in data
# print "... in data"
# chisquared_test( data_size_closed, data_size_open )
# # size contingency of closed vs open in model
# print "... in model"
# chisquared_test( model_size_closed, model_size_open )

# # ORIENTATION
# print
# print "Orientation statistical significance"
# print "chi-square test of null-hypothesis: 'No difference between open and closed loop condition' ..."
# data_orientation_open = numpy.array( [16, 16, 7, 3, 2, 0.00000000001, 2, 0.00000000001, 0.00000000001, 0.00000000001] ) # expected open
# data_orientation_closed = numpy.array( [22, 39, 20, 10, 4, 2, 0, 0, 0, 0] ) # expected closed
# model_orientation_open = numpy.array( [30, 25, 24, 11, 15, 14, 8, 2, 3, 1] ) # observed open
# model_orientation_closed = numpy.array( [16, 22, 20, 14, 18, 11, 10, 11, 8, 3] ) # observed closed
# # size contingency of closed vs open in data
# print "... in data"
# chisquared_test( data_orientation_closed, data_orientation_open )
# # size contingency of closed vs open in model
# print "... in model"
# chisquared_test( model_orientation_closed, model_orientation_open )


