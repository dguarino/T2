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
matplotlib.use('Agg') # for Docker

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




def SpikeTriggeredAverage(lfp_sheet, spike_sheet, folder, stimulus, parameter, ylim=[0.,100.], tip_box=[[.0,.0],[.0,.0]], radius=False, lfp_opposite=False, spike_opposite=False, addon="", color="black"):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "LFP sheet: ",lfp_sheet
	print "Spike sheet: ",spike_sheet
	import ast
	from scipy import signal
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	# data_store.print_content(full_recordings=False)

	segs = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()
	for s in segs:
		print "ann", s.annotations

	# LFP
	# 95% of the LFP signal is a result of all exc and inh cells conductances from 250um radius from the tip of the electrode (Katzner et al. 2009).
	# Therefore we include all recorded cells but account for the distance-dependent contribution weighting currents /r^2
	# We assume that the electrode has been placed in the cortical coordinates <tip>
	# Only excitatory neurons are relevant for the LFP (because of their geometry) Bartosz
	filtered_dsv = param_filter_query(data_store, sheet_name=lfp_sheet, st_name=stimulus)
	print filtered_dsv
	filtered_segs = filtered_dsv.get_segments()
	for s in filtered_segs:
		print "SpikeTriggeredAverage",s
	lfp_neurons = filtered_segs[0].get_stored_vm_ids()
	print lfp_neurons

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






###################################################
# Execution

full_list = [ 
	# "ThalamoCorticalModel_data_size_closed_vsdi_____",
	"ThalamoCorticalModel_data_size_feedforward_vsdi", 
	]


addon = ""
# sheets = ['X_ON', 'X_OFF', 'PGN', 'V1_Exc_L4', 'V1_Inh_L4']
sheets = ['V1_Exc_L4'] 

print sheets


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
					
		# stLFP: Exc iso-surround postsynaptic response to iso-center spikes
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
		SpikeTriggeredAverage(
			lfp_sheet='V1_Exc_L4', 
			spike_sheet='V1_Exc_L4', 
			folder=f, 
			stimulus='DriftingSinusoidalGratingDisk',
			parameter="radius",
			ylim=[0,50],
			lfp_opposite=False, # ISO
			spike_opposite=False, # ISO
			tip_box = [[-.6,-.6],[.6,.6]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
			radius = [1.,1.8], # surround
			addon = addon + "_center2surround",
			color = color,
		)

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
