# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import os

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

from mozaik.tools.mozaik_parametrized import colapse




def get_per_neuron_firing_rate( datastore, stimulus, sheet, start, end, selected_parameter, parameter_list ):
	dsv = queries.param_filter_query( datastore, st_name=stimulus, sheet_name=sheet )
	# get neo segment
	segs = dsv.get_segments() # from datastore DataStoreView
	print "NEO segments: ",len(segs)

	# cutout
	# print segs[0].spiketrains[0]
	for i,s in enumerate(segs):
	    for j,t in enumerate(s.spiketrains):
	        segs[i].spiketrains[j] = t.time_slice( start*qt.ms, end*qt.ms )
	# print segs[0].spiketrains[0]

	st = [ MozaikParametrized.idd(s) for s in dsv.get_stimuli() ]
	# print "stimuli: ", len(st)
	for i,s in enumerate(st):
		# print s.name
		st[i].duration = end-start
	# print "st:", st

	# transform spike trains due to stimuly to mean_rates
	mean_rates = [numpy.array(s.mean_rates()) for s in segs]
	print "mean_rates: ",len(mean_rates)

	# join rates and stimuli
	(mean_rates, s) = colapse(mean_rates, st, parameter_list=parameter_list)
	stimuli = [getattr(i, selected_parameter) for i in s]
	rates = numpy.array(mean_rates)
	# print rates.shape #(stimulus, trial, cells)

	return rates, stimuli




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.] ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	rates, stimuli = get_per_neuron_firing_rate( data_store, stimulus, sheet, start, end, parameter, parameter_list=['trial'] )
	print rates

	# compute per-trial mean rate over cells axis=2
	collapsed_mean_rates = numpy.mean(rates, axis=2) 
	# print "Ex. collapsed_mean_rates: ", collapsed_mean_rates.shape
	# print "Ex. collapsed_mean_rates (for stimulus 0): ", collapsed_mean_rates[0]
	collapsed_std_rates = numpy.std(rates, axis=2, ddof=1) # ddof to calculate the 'corrected' sample sd = sqrt(N/(N-1)) times population sd, where N is the number of points
	# print "Ex. collapsed_std_rates (for stimulus 0): ", collapsed_std_rates[0]

	final_mean_rates = numpy.mean(collapsed_mean_rates, axis=1) 
	final_std_rates = numpy.mean(collapsed_std_rates, axis=1) # mean std
	# print "stimuli: ", stimuli	
	# print "final means and stds: ", final_mean_rates, final_std_rates
	# print sorted( zip(stimuli, final_mean_rates, final_std_rates), key=lambda entry: entry[0] ) )
	final_sorted = [ numpy.array(list(e)) for e in zip( *sorted( zip(stimuli, final_mean_rates, final_std_rates), key=lambda entry: entry[0] ) ) ]

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





def perform_pairwise_comparison( sheet, folder_full, folder_inactive, stimulus, stimulus_band, parameter, start, end, xlabel="", ylabel="", withRegression=True, withCorrCoef=True, withCentroid=False, withLowPassIndex=False ):
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


	x_full_rates, stimuli_full = get_per_neuron_firing_rate( data_store_full, stimulus, sheet, start, end, parameter, parameter_list=['trial'] )
	x_inac_rates, stimuli_inac = get_per_neuron_firing_rate( data_store_inac, stimulus, sheet, start, end, parameter, parameter_list=['trial'] )
	# print x_full_rates.shape # (5, 6, 133) # (stimuli, trials, cells)

	# compute mean rate over trials (axis=1) so to have per-cells trial-averaged stimuli
	collapsed_mean_x_full_rates = numpy.mean(x_full_rates, axis=1) 
	collapsed_mean_x_inac_rates = numpy.mean(x_inac_rates, axis=1) 
	# print "Ex. collapsed_mean_rates: ", collapsed_mean_x_full_rates.shape
	# print "Ex. collapsed_mean_rates: ", collapsed_mean_x_inac_rates.shape
	# print "Ex. collapsed_mean_rates (for stimulus 0): ", collapsed_mean_x_full_rates[0]

	x_full = collapsed_mean_x_full_rates[stimulus_band]
	# print "x_full: ", x_full.shape, x_full
	x_inac = collapsed_mean_x_inac_rates[stimulus_band]
	#!!!!!!!!!!!!! only for spontaneous test
	# x_inac = x_inac - numpy.random.uniform(low=0.0, high=1., size=(len(x_full)) ) 

	if withLowPassIndex:
		# normalized
		normalized_full = x_full / numpy.amax( x_full )
		normalized_inac = x_inac / numpy.amax( x_inac )
		# highest spatial frequency response is for [4]
		x_full_highest = collapsed_mean_x_full_rates[4] 
		x_inac_highest = collapsed_mean_x_inac_rates[4]
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
	plt.savefig( folder_inactive+"/TrialAveragedPairwiseComparison_"+stimulus+"_"+parameter+"_"+sheet+".png", dpi=200 )
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

	"ThalamoCorticalModel_data_spatial_V1_full_____"
	# "ThalamoCorticalModel_data_spatial_____.0012"
	# "ThalamoCorticalModel_data_spatial_Kimura_____"
	]

inac_list = [ 
	# "ThalamoCorticalModel_data_luminance_____"

	"ThalamoCorticalModel_data_spatial_Kimura_____"
	]



sheets = ['X_ON'] #['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']



for i,f in enumerate(full_list):
	print i,f

	for s in sheets:

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
		# 	useYlog=False, 
		# 	percentile=False 
		# )



		for j,l in enumerate(inac_list):
			print j

			# # SPONTANEOUS ACTIVITY
			# perform_pairwise_comparison( 
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

			# SPATIAL FREQUENCY
			perform_pairwise_comparison( 
				sheet=s, 
				folder_full=f, 
				folder_inactive=l,
				stimulus="FullfieldDriftingSinusoidalGrating",
				stimulus_band=0,
				parameter='spatial_frequency',
				start=100., 
				end=10000., 
				xlabel="Control",
				ylabel="PGN Inactivated",
				withRegression=False,
				withCorrCoef=False,
				withCentroid=True,
				withLowPassIndex=True
			)
