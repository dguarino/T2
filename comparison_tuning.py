# Plotting to compare single cell tuning curves in two conditions
# by having two folder/datastore
# assuming the same amount of recorded cells in the two conditions
import sys
import mozaik
import mozaik.controller
from parameters import ParameterSet

import gc
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



def select_ids_by_position(position, radius, sheet_ids, positions, reverse=False, box=[]):
	radius_ids = []
	distances = []
	min_radius = radius[0]
	max_radius = radius[1]

	for i in sheet_ids:
		a = numpy.array((positions[0][i],positions[1][i],positions[2][i]))
		l = numpy.linalg.norm(a - position)

		if len(box)>1:
			# Ex box: [ [-.3,.0], [.3,-.4] ]
			# Ex a: [ [-0.10769224], [ 0.16841423], [ 0. ] ]
			# print box[0][0], a[0], box[1][0], "      ", box[0][1], a[1], box[1][1]
			if a[0]>=box[0][0] and a[0]<=box[1][0] and a[1]>=box[0][1] and a[1]<=box[1][1]:
				radius_ids.append(i[0])
				distances.append(l)
		else:
			#print a, " - ", position

			# print "distance",l
			if abs(l)>min_radius and abs(l)<max_radius:
				# print "taken"
				radius_ids.append(i[0])
				distances.append(l)

	# sort by distance
	# print len(radius_ids)
	# print distances
	return [x for (y,x) in sorted(zip(distances,radius_ids), key=lambda pair:pair[0], reverse=reverse)]




def perform_comparison_size_tuning( sheet, reference_position, step, sizes, folder_full, folder_inactive, reverse=False, Ismaller=[2,3], Iequal=[4,5], Ilarger=[6,8], box=[] ):
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
	rowplots = 0
	max_size = 0.6

	# GET RECORDINGS BY POSITION (either step or box. In case of using box, inefficiently repetition of box-ing step times!)
	slice_ranges = numpy.arange(step, max_size+step, step)
	print "slice_ranges:",slice_ranges
	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step,cur_range]
		print col
		# get the list of all recorded neurons in X_ON
		# Full
		spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		positions1 = data_store_full.get_neuron_postions()[sheet]
		# print numpy.min(positions1), numpy.max(positions1) 
		sheet_ids1 = data_store_full.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids1)
		radius_ids1 = select_ids_by_position(reference_position, radius, sheet_ids1, positions1, reverse, box)
		neurons1 = data_store_full.get_sheet_ids(sheet_name=sheet, indexes=radius_ids1)

		# Inactivated
		spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
		positions2 = data_store_inac.get_neuron_postions()[sheet]
		sheet_ids2 = data_store_inac.get_sheet_indexes(sheet_name=sheet,neuron_ids=spike_ids2)
		radius_ids2 = select_ids_by_position(reference_position, radius, sheet_ids2, positions2, reverse, box)
		neurons2 = data_store_inac.get_sheet_ids(sheet_name=sheet, indexes=radius_ids2)

		print neurons1
		print neurons2
		if not set(neurons1)==set(neurons2):
			neurons1 = numpy.intersect1d(neurons1, neurons2)
			neurons2 = neurons1

		if len(neurons1) > rowplots:
			rowplots = len(neurons1)

		neurons_full.append(neurons1)
		neurons_inac.append(neurons2)

		print "radius_ids", radius_ids2
		print "neurons_full:", len(neurons_full[col]), neurons_full[col]
		print "neurons_inac:", len(neurons_inac[col]), neurons_inac[col]

		assert len(neurons_full[col]) > 0 , "ERROR: the number of recorded neurons is 0"

	# subplot figure creation
	plotOnlyPop = False
	print 'rowplots', rowplots
	print "Starting plotting ..."
	print "slice_ranges:", len(slice_ranges), slice_ranges
	if len(slice_ranges) >1:
		fig, axes = plt.subplots(nrows=len(slice_ranges), ncols=rowplots+1, figsize=(3*rowplots, 3*len(slice_ranges)), sharey=False)
	else:
		fig, axes = plt.subplots(nrows=2, ncols=2, sharey=False)
		plotOnlyPop=True
	print axes.shape

	p_significance = .02
	for col,cur_range in enumerate(slice_ranges):
		radius = [cur_range-step, cur_range]
		print col
		interval = str(radius[0]) +" - "+ str(radius[1]) +" deg radius"
		print interval
		axes[col,0].set_ylabel(interval+"\n\nResponse change (%)")
		print "range:",col
		if len(neurons_full[col]) < 1:
			continue

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

		print "(stimulus conditions, cells):", tc_dict1[0].values()[0][1].shape # ex. (10, 32) firing rate for each stimulus condition (10) and each cell (32)

		# Population histogram
		diff_full_inac = []
		sem_full_inac = []
		num_cells = tc_dict1[0].values()[0][1].shape[1]
		smaller_pvalue = 0.
		equal_pvalue = 0.
		larger_pvalue = 0.

		# 1. SELECT ONLY CHANGING UNITS
		all_open_values = tc_dict2[0].values()[0][1]
		all_closed_values = tc_dict1[0].values()[0][1]

		# 1.1 Search for the units that are NOT changing (within a certain absolute tolerance)
		unchanged_units = numpy.isclose(all_closed_values, all_open_values, rtol=0., atol=4.)
		# print unchanged_units.shape

		# 1.2 Reverse them into those that are changing
		changed_units = numpy.invert( unchanged_units )
		# print numpy.nonzero(changed_units)

		# 1.3 Get the indexes of all units that are changing
		changing_idxs = []
		for i in numpy.nonzero(changed_units)[0]:
			for j in numpy.nonzero(changed_units)[1]:
				if j not in changing_idxs:
					changing_idxs.append(j)
		# print sorted(changing_idxs)

		# 1.4 Get the changing units
		open_values = [ x[changing_idxs] for x in all_open_values ]
		open_values = numpy.array(open_values)
		closed_values = [ x[changing_idxs] for x in all_closed_values ]
		closed_values = numpy.array(closed_values)
		print "chosen open units:", open_values.shape
		print "chosen closed units:", closed_values.shape
		num_cells = closed_values.shape[1]

		# 2. AUTOMATIC SEARCH FOR INTERVALS
		# peak = max(numpy.argmax(closed_values, axis=0 ))
		peaks = numpy.argmax(closed_values, axis=0 )
		# peak = int( numpy.argmax( closed_values ) / closed_values.shape[1] ) # the returned single value is from the flattened array
		# print "numpy.argmax( closed_values ):", numpy.argmax( closed_values )
		print "peaks:", peaks
		# minimum = int( numpy.argmin( closed_values ) / closed_values.shape[1] )
		# minimum = min(numpy.argmin(closed_values, axis=0 ))
		minimums = numpy.argmin(closed_values, axis=0 ) +2 # +2 to get the response out of the smallest
		# print "numpy.argmin( closed_values ):", numpy.argmin( closed_values )
		print "minimums:", minimums

		# -------------------------------------
		# DIFFERENCE BETWEEN INACTIVATED AND CONTROL
		# We want to have a summary measure of the population of cells with and without inactivation.
		# Our null-hypothesis is that the inactivation does not change the activity of cells.
		# A different result will tell us that the inactivation DOES something.
		# Therefore our null-hypothesis is the result obtained in the intact system.
		# Procedure:
		# We have several stimulus sizes
		# We want to group them in three: smaller than optimal, optimal, larger than optimal
		# We do the mean response for each cell for the grouped stimuli
		#    i.e. sum the responses for each cell across stimuli in the group, divided by the number of stimuli in the group
		# We repeat for each group

		# average of all trial-averaged response for each cell for grouped stimulus size
		# we want the difference / normalized by the highest value * expressed as percentage
		# print num_cells
		# print "inac",numpy.sum(tc_dict2[0].values()[0][1][2:3], axis=0)
		# print "full",numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0)
		# print "diff",(numpy.sum(tc_dict2[0].values()[0][1][2:3], axis=0) - numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0))
		# print "diff_norm",((numpy.sum(tc_dict2[0].values()[0][1][2:3], axis=0) - numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0)) / (numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0)))
		# print "diff_norm_perc",((numpy.sum(tc_dict2[0].values()[0][1][2:3], axis=0) - numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0)) / (numpy.sum(tc_dict1[0].values()[0][1][2:3], axis=0))) * 100

		# USING PROVIDED INTERVALS
		# diff_smaller = ((numpy.sum(open_values[Ismaller[0]:Ismaller[1]], axis=0) - numpy.sum(closed_values[Ismaller[0]:Ismaller[1]], axis=0)) / numpy.sum(closed_values[Ismaller[0]:Ismaller[1]], axis=0)) * 100
		# diff_equal = ((numpy.sum(open_values[Iequal[0]:Iequal[1]], axis=0) - numpy.sum(closed_values[Iequal[0]:Iequal[1]], axis=0)) / numpy.sum(closed_values[Iequal[0]:Iequal[1]], axis=0)) * 100
		# diff_larger = ((numpy.sum(open_values[Ilarger[0]:Ilarger[1]], axis=0) - numpy.sum(closed_values[Ilarger[0]:Ilarger[1]], axis=0)) / numpy.sum(closed_values[Ilarger[0]:Ilarger[1]], axis=0)) * 100

		# USING AUTOMATIC SEARCH
		# print "open"
		# print open_values[minimums]
		# print "closed"
		# print closed_values[minimums]
		# print open_values[peaks]
		# print closed_values[peaks]

		diff_smaller = ((numpy.sum(open_values[minimums], axis=0) - numpy.sum(closed_values[minimums], axis=0)) / numpy.sum(closed_values[minimums], axis=0)) * 100
		diff_equal = ((numpy.sum(open_values[peaks], axis=0) - numpy.sum(closed_values[peaks], axis=0)) / numpy.sum(closed_values[peaks], axis=0)) * 100
		diff_larger = ((numpy.sum(open_values[Ilarger[0]:Ilarger[1]], axis=0) - numpy.sum(closed_values[Ilarger[0]:Ilarger[1]], axis=0)) / numpy.sum(closed_values[Ilarger[0]:Ilarger[1]], axis=0)) * 100
		# print "diff_smaller", diff_smaller
		# print "diff_equal", diff_smaller
		# print "diff_larger", diff_smaller

		# average of all cells
		smaller = sum(diff_smaller) / num_cells
		equal = sum(diff_equal) / num_cells
		larger = sum(diff_larger) / num_cells
		print "smaller",smaller
		print "equal", equal
		print "larger", larger

		# 0/0
		# Check using scipy
		# and we want to compare the responses of full and inactivated
		# smaller, smaller_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][0:3], axis=0)/3, numpy.sum(tc_dict1[0].values()[0][1][0:3], axis=0)/3 )
		# equal, equal_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][3:5], axis=0)/2, numpy.sum(tc_dict1[0].values()[0][1][3:5], axis=0)/2 )
		# larger, larger_pvalue = scipy.stats.ttest_rel( numpy.sum(tc_dict2[0].values()[0][1][5:], axis=0)/5, numpy.sum(tc_dict1[0].values()[0][1][5:], axis=0)/5 )
		# print "smaller, smaller_pvalue:", smaller, smaller_pvalue
		# print "equal, equal_pvalue:", equal, equal_pvalue
		# print "larger, larger_pvalue:", larger, larger_pvalue

		diff_full_inac.append( smaller )
		diff_full_inac.append( equal )
		diff_full_inac.append( larger )

		# -------------------------------------
		# Standard Error Mean calculated on the full sequence
		sem_full_inac.append( scipy.stats.sem(diff_smaller) )
		sem_full_inac.append( scipy.stats.sem(diff_equal) )
		sem_full_inac.append( scipy.stats.sem(diff_larger) )

		# print diff_full_inac
		# print sem_full_inac
		barlist = axes[col,0].bar([0.5,1.5,2.5], diff_full_inac, yerr=sem_full_inac, width=0.8)
		axes[col,0].plot([0,4], [0,0], 'k-') # horizontal 0 line
		for ba in barlist:
			ba.set_color('white')
		if smaller_pvalue < p_significance:
			barlist[0].set_color('brown')
		if equal_pvalue < p_significance:
			barlist[1].set_color('darkgreen')
		if larger_pvalue < p_significance:
			barlist[2].set_color('blue')

		# Plotting tuning curves
		x_full = tc_dict1[0].values()[0][0]
		x_inac = tc_dict2[0].values()[0][0]
		# each cell couple 
		axes[col,1].set_ylabel("Response (spikes/sec)", fontsize=10)
		for j,nid in enumerate(neurons_full[col][changing_idxs]):
			# print col,j,nid
			if len(neurons_full[col][changing_idxs])>1: # case with just one neuron in the group
				y_full = closed_values[:,j]
				y_inac = open_values[:,j]
			else:
				y_full = closed_values
				y_inac = open_values
			if not plotOnlyPop:
				axes[col,j+1].plot(x_full, y_full, linewidth=2, color='b')
				axes[col,j+1].plot(x_inac, y_inac, linewidth=2, color='r')
				axes[col,j+1].set_title(str(nid), fontsize=10)
				axes[col,j+1].set_xscale("log")

	fig.subplots_adjust(hspace=0.4)
	# fig.suptitle("All recorded cells grouped by circular distance", size='xx-large')
	fig.text(0.5, 0.04, 'cells', ha='center', va='center')
	fig.text(0.06, 0.5, 'ranges', ha='center', va='center', rotation='vertical')
	for ax in axes.flatten():
		ax.set_ylim([0,60])
		ax.set_xticks(sizes)
		ax.set_xticklabels([0.1, '', '', '', '', 1, '', 2, 4, 6])
		# ax.set_xticklabels([0.1, '', '', '', '', '', '', '', '', '', '', 1, '', '', 2, '', '', '', 4, '', 6])

	for col,_ in enumerate(slice_ranges):
		# axes[col,0].set_ylim([-.8,.8])
		axes[col,0].set_ylim([-60,60])
		axes[col,0].set_yticks([-60, -40, -20, 0., 20, 40, 60])
		axes[col,0].set_yticklabels([-60, -40, -20, 0, 20, 40, 60])
		axes[col,0].set_xlim([0,4])
		axes[col,0].set_xticks([.9,1.9,2.9])
		axes[col,0].set_xticklabels(['small', 'equal', 'larger'])
		axes[col,0].spines['right'].set_visible(False)
		axes[col,0].spines['top'].set_visible(False)
		axes[col,0].spines['bottom'].set_visible(False)

	# plt.show()
	plt.savefig( folder_inactive+"/TrialAveragedSizeTuningComparison_"+sheet+"_step"+str(step)+"_box"+str(box)+".png", dpi=100 )
	# plt.savefig( folder_full+"/TrialAveragedSizeTuningComparison_"+sheet+"_"+interval+".png", dpi=100 )
	fig.clf()
	plt.close()
	# garbage
	gc.collect()




def perform_percent_tuning( sheet, reference_position, step, sizes, folder_full, folder_inactive ):
	print folder_full
	data_store_full = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_full, 'store_stimuli' : False}),replace=True)
	data_store_full.print_content(full_recordings=False)
	print folder_inactive
	data_store_inac = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder_inactive, 'store_stimuli' : False}),replace=True)
	data_store_inac.print_content(full_recordings=False)

	# full
	spike_ids1 = param_filter_query(data_store_full, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	dsv = param_filter_query( data_store_full, st_name='DriftingSinusoidalGratingDisk', analysis_algorithm=['TrialAveragedFiringRateCutout'] )

	PlotTuningCurve(
		dsv,
		ParameterSet({
			'polar': False,
			'pool': False,
			'centered': False,
			'percent': True,
			'mean': True,
			'parameter_name' : 'radius', 
			# 'neurons': list(spike_ids1[11:12]), 
			'neurons': list(spike_ids1), 
			'sheet_name' : sheet
		}), 
		fig_param={'dpi' : 100,'figsize': (8,8)}, 
		# plot_file_name=folder_full+"/"+"SizeTuning_Grating_"+sheet+"_percent_"+str(spike_ids1[11:12])+".png"
		plot_file_name=folder_full+"/"+"SizeTuning_Grating_"+sheet+"_mean_percent.png"
	).plot({
		'*.y_lim':(0,100), 
		'*.y_label': "Response (%)",
		# '*.y_ticks':[10, 20, 30, 40, 50], 
		'*.x_ticks':[0.1, 1, 2, 4, 6], 
		'*.x_scale':'linear',
		#'*.x_scale':'log', '*.x_scale_base':2,
		'*.fontsize':24
	})

	# inactivated
	spike_ids2 = param_filter_query(data_store_inac, sheet_name=sheet).get_segments()[0].get_stored_spike_train_ids()
	print spike_ids2
	dsv = param_filter_query( data_store_inac, st_name='DriftingSinusoidalGratingDisk', analysis_algorithm=['TrialAveragedFiringRateCutout'] )

	PlotTuningCurve(
		dsv,
		ParameterSet({
			'polar': False,
			'pool': False,
			'centered': False,
			'percent': True,
			'mean': True,
			'parameter_name' : 'radius', 
			'neurons': list(spike_ids2), 
			# 'neurons': list(spike_ids2[11:12]), 
			'sheet_name' : sheet
		}), 
		fig_param={'dpi' : 100,'figsize': (8,8)}, 
		# plot_file_name=folder_inactive+"/"+"SizeTuning_Grating_"+sheet+"_percent_"+str(spike_ids2[11:12])+".png"
		plot_file_name=folder_inactive+"/"+"SizeTuning_Grating_"+sheet+"_mean_percent.png"
	).plot({
		'*.y_lim':(0,100), 
		'*.y_label': "Response (%)",
		# '*.y_ticks':[10, 20, 30, 40, 50], 
		'*.x_ticks':[0.1, 1, 2, 4, 6], 
		'*.x_scale':'linear',
		#'*.x_scale':'log', '*.x_scale_base':2,
		'*.fontsize':24
	})




def perform_time_course_fano_factor_comparison():
	# What is the added value in having a loop?
	# Compare it with Churchland_etal_2010

	pass




def perform_comparison_size_inputs( sheet, sizes, folder_full, folder_inactive, with_ppd=False ):
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
		# print "mean_inac_gsyn_e", len(mean_inac_gsyn_e), mean_inac_gsyn_e.shape
		# print "mean_inac_gsyn_i", len(mean_inac_gsyn_i), mean_inac_gsyn_i.shape

		# PSP Area response plot (as in LindstromWrobel2011)
		max_full_gsyn_e = numpy.amax(mean_full_gsyn_e, axis=1)
		max_full_gsyn_i = numpy.amax(mean_full_gsyn_i, axis=1)
		norm_full_gsyn_e = (mean_full_gsyn_e.sum(axis=1) / 10291) / max_full_gsyn_e *100
		norm_full_gsyn_i = (mean_full_gsyn_i.sum(axis=1) / 10291) / max_full_gsyn_i *100

		max_inac_gsyn_e = numpy.amax(mean_inac_gsyn_e, axis=1)
		max_inac_gsyn_i = numpy.amax(mean_inac_gsyn_i, axis=1)
		norm_inac_gsyn_e = (mean_inac_gsyn_e.sum(axis=1) / 10291) / max_full_gsyn_e *100
		norm_inac_gsyn_i = (mean_inac_gsyn_i.sum(axis=1) / 10291) / max_full_gsyn_i *100

		plt.figure()
		plt.errorbar(sizes, norm_full_gsyn_e, color='red', linewidth=2)#, xerr=0.2, yerr=0.4)
		plt.errorbar(sizes, norm_full_gsyn_i, color='blue', linewidth=2)#, xerr=0.2, yerr=0.4)
		plt.errorbar(sizes, norm_inac_gsyn_e, color='purple', linewidth=2)#, xerr=0.2, yerr=0.4)
		plt.errorbar(sizes, norm_inac_gsyn_i, color='cyan', linewidth=2)#, xerr=0.2, yerr=0.4)
		plt.xscale("log")
		plt.xticks(sizes, sizes)
		plt.ylabel("PSP (%)", fontsize=10)
		plt.xlabel("sizes", fontsize=10)
		plt.title("PSP Area response plot "+sheet)
		plt.savefig( folder_inactive+"/TrialAveragedPSP_"+sheet+".png", dpi=100 )
		plt.close()

		# Point-to-Point difference 
		if with_ppd:
			diff_e_full_inac = mean_full_gsyn_e - mean_inac_gsyn_e
			diff_i_full_inac = mean_full_gsyn_i - mean_inac_gsyn_i
			# print "diff_e_full_inac", len(diff_e_full_inac), diff_e_full_inac
			# print "diff_i_full_inac", len(diff_i_full_inac), diff_i_full_inac
			fig, axes = plt.subplots(nrows=1, ncols=num_sizes, figsize=(10*num_sizes, 10))
			print axes.shape
			# http://paletton.com/#uid=7020Q0km5KqbrV8hkPPqCEHz+z+
			for s in range(num_sizes):
				axes[s].plot(mean_full_gsyn_e[s], color='#F93026')
				axes[s].plot(mean_full_gsyn_i[s], color='#294BA8')
				axes[s].plot(mean_inac_gsyn_e[s], color='#FF7C75')
				axes[s].plot(mean_inac_gsyn_i[s], color='#7592E1')
				axes[s].plot(diff_e_full_inac[s], color='#FFC64C')
				axes[s].plot(diff_i_full_inac[s], color='#6CEA7B')
				axes[s].set_title(str(sizes[s]))
			plt.savefig( folder_inactive+"/TrialAveragedConductanceComparison_"+sheet+".png", dpi=100 )
			# plt.savefig( folder_full+"/TrialAveragedSizeTuningComparison_"+sheet+"_"+interval+".png", dpi=100 )
			plt.close()

		plt.close()
		# garbage
		gc.collect()






###################################################
# Execution
import os

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


full_list = [ 
	"latest_size_full"
	]

inac_large_list = [ 
	"latest_size_nonoverlapping"
	# "latest_size_overlapping"
	]

# CHOICE OF CELLS
# You can use either radius steps or box. The one excludes the other.
# 1. Use radius steps, to globally inspect at step distances from the center
steps = [.2] # for detail
# steps = [.8] # for all cells
# 2. Use the box (lowerleft, upperright) to have a more specific view
box = [] # empty box to make the radius choice work
box = [[-.5,.0],[.5,.5]] # 
# box = [[-.5,.3],[.5,1.]] # 

# CHOICE OF STIMULI GROUPS
Ismaller = [0,3]
Iequal   = [4,6]
Ilarger  = [6,8] # NON
# Ilarger  = [7,10] # OVER

sheets = ['X_ON'] #['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4']



for i,l in enumerate(full_list):
	# for parameter search
	full = [ l+"/"+f for f in os.listdir(l) if os.path.isdir(os.path.join(l, f)) ]
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

			# perform_comparison_size_inputs( 
			# 	sheet=s,
			# 	sizes = sizes,
			# 	folder_full=f, 
			# 	folder_inactive=large[i],
			# 	with_ppd=True
			# 	)

			for step in steps:

				# perform_percent_tuning( 
				# 	sheet=s, 
				#   reference_position=[[0.0], [0.0], [0.0]],
				# 	step=step, 
				# 	sizes = sizes, 
				# 	folder_full=f, 
				# 	folder_inactive=large[i] 
				# )

				perform_comparison_size_tuning( 
					sheet=s, 
					reference_position=[[0.0], [0.0], [0.0]],
					reverse=True, # False if overlapping, True if non-overlapping
					step=step,
					sizes = sizes,
					folder_full=f, 
					folder_inactive=large[i],
					Ismaller = Ismaller,
					Iequal   = Iequal,
					Ilarger  = Ilarger,
					box = box
				)




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
