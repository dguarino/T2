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




def trial_averaged_tuning_curve_errorbar( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", percentile=False, useXlog=False, useYlog=False, ylim=[0.,100.] ):
	print folder
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	print "Checking data..."
	# Full
	dsv = queries.param_filter_query( data_store, st_name=stimulus, sheet_name=sheet )
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
	(mean_rates, s) = colapse(mean_rates, st, parameter_list=['trial'])
	stimuli = [getattr(i, parameter) for i in s]
	rates = numpy.array(mean_rates)
	#print rates.shape (stimulus, trial, cells)

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
	plt.savefig( folder+"/TrialAveragedTuningCurve_"+stimulus+"_"+sheet+".png", dpi=200 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()






###################################################
# Execution

full_list = [ 
	# "ThalamoCorticalModel_data_luminance_____"

	# "ThalamoCorticalModel_data_contrast_____.0012"
	# "ThalamoCorticalModel_data_contrast_V1_full_____"

	# "ThalamoCorticalModel_data_temporal_____.0008"
	"ThalamoCorticalModel_data_temporal_V1_____.0008"
	]



sheets = ['X_ON', 'X_OFF'] #['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']


for i,l in enumerate(full_list):
	print i,l

	for s in sheets:

		# # LUMINANCE
		# trial_averaged_tuning_curve_errorbar( 
		# 	sheet=s, 
		# 	folder=l, 
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
		# 	folder=l, 
		# 	stimulus='FullfieldDriftingSinusoidalGrating',
		# 	parameter="contrast",
		# 	start=100., 
		# 	end=10000., 
		# 	xlabel="contrast", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color="black", 
		# 	percentile=True 
		# )

		# TEMPORAL
		trial_averaged_tuning_curve_errorbar( 
			sheet=s, 
			folder=l, 
			stimulus='FullfieldDriftingSinusoidalGrating',
			parameter="temporal_frequency",
			start=100., 
			end=10000., 
			xlabel="Temporal frequency", 
			ylabel="firing rate (sp/s)", 
			color="black", 
			useXlog=True, 
			useYlog=True, 
			percentile=False 
		)



# for i,l in enumerate(full_list):
# 	# for parameter search
# 	full = [ l+"/"+f for f in os.listdir(l) if os.path.isdir(os.path.join(l, f)) ]
# 	large = [ inac_large_list[i]+"/"+f for f in os.listdir(inac_large_list[i]) if os.path.isdir(os.path.join(inac_large_list[i], f)) ]

# 	for i,f in enumerate(full):
# 		print i

# 		for s in sheets:

# 			perform_pairwise_comparison( 
# 				sheet=s, 
# 				folder_full=f, 
# 				folder_inactive=large[i],
# 				parameter='background_luminance',
# 				indices=[0,1,2], # data, trialaveraged, 3rd stimulus
# 				xlabel="sponatneous activity before cooling (spikes/s)",
# 				ylabel="sponatneous activity during cooling (spikes/s)",
# 				withRegression=True,
# 				withCorrCoef=True
# 			)