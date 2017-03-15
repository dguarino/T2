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
			# print box[0][0], a[0], box[1][0]
			# print box[0][1], a[1], box[1][1]
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





def get_per_neuron_spike_count( datastore, stimulus, sheet, start, end, stimulus_parameter, bin=10.0 ):

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

	# dsv = queries.param_filter_query( datastore, st_name=stimulus, sheet_name=sheet )
	# # get neo segment, sorted by stimulu parameter chosen
	# segs = dsv.get_segments()
	# print "NEO segments: ",len(segs)

	# st = [ MozaikParametrized.idd(s) for s in dsv.get_stimuli() ]
	# stimuli = []
	# for s in st:
	# 	param = getattr(s, stimulus_parameter)
	# 	if param not in stimuli:
	# 		stimuli.append(param)
	# stimuli = sorted(stimuli)
	# # print "Stimuli: ", len(stimuli), stimuli

	# spike_ids = sorted( segs[0].get_stored_spike_train_ids() )
	# # print "spike_ids:", len(spike_ids), spike_ids
 
	# qstart = (start * qt.ms).rescale(qt.s).magnitude
	# qend = (end * qt.ms).rescale(qt.s).magnitude
	# trials = len(st) / len(stimuli)
	# # print "trials:", trials

	# # Compute Mean firing rate per cell (source_id) and chosen stimulus_parameter
	# mean_rates = numpy.zeros( (len(stimuli), len(spike_ids)) )
	# # print "mean_rates.shape:", mean_rates.shape
	# # for each segment (whose content is unsorted)
	# for g in segs:

	# 	# get segment annotation
	# 	stim = ast.literal_eval( g.annotations['stimulus'] )
	# 	# get current stimulus parameter index among (sorted) stimuli
	# 	i = stimuli.index( stim[stimulus_parameter] )
	# 	# print i, "==", stim[stimulus_parameter] # check correct match all times

	# 	# for each cell spiketrain
	# 	for t in g.spiketrains:
	# 		# get current cell index among (sorted) spike_ids
	# 		j = spike_ids.index(t.annotations['source_id'])
	# 		# print j, "==", t.annotations['source_id'] # check correct match all times
	# 		# increment its spike count (cutout)
	# 		mean_rates[i][ j ] = mean_rates[i][ j ] + len(t.time_slice(start, end))/(qend-qstart)

	# 	# do the mean rate over trials
	# 	mean_rates[i] = mean_rates[i] / trials

	return mean_rates, stimuli



def trial_averaged_corrected_xcorrelation( sheet, folder, stimulus, parameter, start, end, bin=10.0, xlabel="", ylabel="" ):
	# following Brody1998a: "Correlations Without Synchrony"
	# shuffe-corrected cross-correlogram

	print "folder: ",folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

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

	dsv = param_filter_query( data_store, identifier='AnalogSignalList', sheet_name=sheet, st_name=stimulus, analysis_algorithm='SpikeCount' )
	dsv.print_content(full_recordings=False)

	# get S1 binned spiketrains
	box = [[-3.,-.5],[-2.,.5]]
	ids1 = select_ids_by_position(positions, sheet_ids, box=box)
	neurons1 = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)
	S1 = []
	for idd in neurons1:
		S1.append( zip([idd], [ads.get_asl_by_id(idd).magnitude for ads in dsv.get_analysis_result()]) )
	print "S1",len(S1)
	# print S1

	# get S1 binned spiketrains
	box = [[2.,-.5],[3.,.5]]
	ids2 = select_ids_by_position(positions, sheet_ids, box=box)
	neurons2 = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids2)
	S2 = []
	for idd in neurons2:
		S2.append( zip([idd], [ads.get_asl_by_id(idd).magnitude for ads in dsv.get_analysis_result()]) )
	print "S2",len(S2)
	# print S2

	# take the smallest number of ids
	num_ids = len(S1)
	if num_ids > len(S2):
		num_ids = len(S2)
	# print num_ids

	# correlation of one-vs-one cell 
	C = []
	for i in range(num_ids):
		C.append( numpy.correlate(S1[i][0][1], S2[i][0][1], "same") )

	print C

	# probably this is not good
	# P(t) for each recorded cell
	P, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter, bin=bin )
	print P.shape # (stimuli, cells)
	# we should get an average binned spiketrain per cell as P

	# to be able to subtract the right S and P
	# either we extract from these numbers the selected idd or we make more complex the function get_per_neuron_spike_count

	# then we should identify how to make the 







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
	ax.plot((mean, mean), (0,30), 'k:', linewidth=2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	barlist[0].set_color('k')
	barlist[1].set_color('k')
	barlist[2].set_color('k')
	barlist[3].set_color('k')
	barlist[4].set_color('k')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.axis([bin_edges[0], bin_edges[-1], 0, 30])
	plt.xticks(bin_edges, (10,9,8,7,6,5,4,3,2,1))
	plt.savefig( folder+"/suppression_index_"+sheet+".png", dpi=200 )
	plt.close()
	# garbage
	gc.collect()





def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.] ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_spike_count( data_store, stimulus, sheet, start, end, parameter )
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

	if not percentile:
		ax.set_ylim(ylim)

	err_max = final_sorted[1] + final_sorted[2]
	err_min = final_sorted[1] - final_sorted[2]
	ax.fill_between(final_sorted[0], err_max, err_min, color=color, alpha=0.3)

	# text
	ax.set_title( sheet )
	ax.set_xlabel( xlabel )
	if percentile:
		ylabel = "Percentile " + ylabel
	ax.set_ylabel( ylabel )
	# ax.legend( loc="lower right", shadow=False )
	# plt.show()
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+stimulus+"_"+parameter+"_"+sheet+".png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()





def perform_pairwise_scatterplot( sheet, folder_full, folder_inactive, stimulus, stimulus_band, parameter, start, end, xlabel="", ylabel="", withRegression=True, withCorrCoef=True, withCentroid=False, withLowPassIndex=False ):
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
		x_full_highest = x_full_rates[4] 
		x_inac_highest = x_inac_rates[4]
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




###################################################
# Execution

full_list = [ 
	# "ThalamoCorticalModel_data_luminance_____"
	# "ThalamoCorticalModel_data_luminance_V1_fake_____"

	# "ThalamoCorticalModel_data_contrast_____.0012"
	# "ThalamoCorticalModel_data_contrast_V1_full_____"

	# "ThalamoCorticalModel_data_temporal_____.0008"
	# "ThalamoCorticalModel_data_temporal_V1_____.0008"

	# "ThalamoCorticalModel_data_spatial_V1_full_____"
	# "ThalamoCorticalModel_data_spatial_____.0012"
	# "ThalamoCorticalModel_data_spatial_Kimura_____"

	# "ThalamoCorticalModel_data_size_V1_full_____"
	# "ThalamoCorticalModel_data_size_open_____"

	# "ThalamoCorticalModel_data_orientation_V1_full_____"
	# "ThalamoCorticalModel_data_orientation_open_____"

	"ThalamoCorticalModel_data_xcorr_open_____"
	]

inac_list = [ 
	# "ThalamoCorticalModel_data_luminance_____"

	# "ThalamoCorticalModel_data_spatial_Kimura_____"
	# "ThalamoCorticalModel_data_spatial_____.0012"

	]



sheets = ['X_ON'] #['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']



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
		# 	parameter="spatial_frequency",
		# 	start=100., 
		# 	end=10000., 
		# 	xlabel="Spatial frequency", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color="black", 
		# 	useXlog=True, 
		# 	useYlog=True, 
		# 	percentile=False 
		# )

		# # SIZE
		# # Ex: ThalamoCorticalModel_data_size_V1_full_____
		# # Ex: ThalamoCorticalModel_data_size_open_____
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

		# #ORIENTATION
		# Ex: ThalamoCorticalModel_data_orientation_V1_full_____
		# Ex: ThalamoCorticalModel_data_orientation_open_____
		# trial_averaged_tuning_curve_errorbar( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='FullfieldDriftingSinusoidalGrating',
		# 	parameter="orientation",
		# 	start=100., 
		# 	end=2000., 
		# 	xlabel="Orientation", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color="black", 
		# 	useXlog=False, 
		# 	useYlog=False, 
		# 	percentile=False 
		# )
		# perform_end_inhibition_barplot( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus="FullfieldDriftingSinusoidalGrating",
		# 	parameter='orientation',
		# 	start=100., 
		# 	end=2000., 
		# 	xlabel="Orientation bias",
		# 	ylabel="Number of cells",
		# )

		# #CROSS-CORRELATION
		trial_averaged_corrected_xcorrelation( 
			sheet=s, 
			folder=f, 
			stimulus='FlashingSquares',
			parameter="contrast",
			start=100., 
			end=10000., 
			bin=10.0,
			xlabel="correlation", 
			ylabel="firing rate (sp/s)"
		)

		# PAIRWISE
		for j,l in enumerate(inac_list):
			print j

			# # SPONTANEOUS ACTIVITY
			# # Ex: ThalamoCorticalModel_data_luminance_V1_fake_____ vs ThalamoCorticalModel_data_luminance_____
			# perform_pairwise_scatterplot( 
			# 	sheet=s, 
			# 	folder_full=f, 
			# 	folder_inactive=l,
			# 	stimulus="Null",
			# 	stimulus_band=1,
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
			# 	withLowPassIndex=True
			# )
