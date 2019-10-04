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




def LFP( sheet, folder, stimulus, parameter, tip=[.0,.0,.0], sigma=0.300, ylim=[0.,-1.], addon="", color='black' ):
	import matplotlib as ml
	import quantities as pq
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet

	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_esyn_ids()
	if ids == None or len(ids)<1:
		print "No gesyn recorded.\n"
		return
	print "Recorded gesyn:", len(ids), ids

	ids = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_vm_ids()
	if ids == None or len(ids)<1:
		print "No Vm recorded.\n"
		return
	print "Recorded Vm:", len(ids), ids

	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	l4_exc_or_many = numpy.array(ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in ids]) < .1)[0]]
	ids = list(l4_exc_or_many)

	print "Recorded neurons:", len(ids), ids
	# 900 neurons over 6000 micrometers, 200 micrometers interval

	sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=ids)

	positions = data_store.get_neuron_postions()[sheet]
	print positions.shape # all 10800

	# take the positions of the ids
	ids_positions = numpy.transpose(positions)[sheet_indexes,:]
	print ids_positions.shape
	print ids_positions

	# Pre-compute distances from the LFP tip
	distances = []
	for i in range(len(ids)):
		distances.append( numpy.linalg.norm( numpy.array(ids_positions[i][0]) - numpy.array(tip) ) ) 
	distances = numpy.array(distances)
	print "distances:", len(distances), distances

	# ##############################
	# LFP
	# tip = [[x],[y],[.0]]
	# For each recorded cell:
	# Gaussianly weight it by its distance from tip
	# produce the currents
	# Divide the whole by the norm factor (area): 4 * numpy.pi * sigma

	# 95% of the LFP signal is a result of all exc and inh cells conductances from 250um radius from the tip of the electrode (Katzner et al. 2009).
	# Mostly excitatory neurons are relevant for the LFP (because of their geometry) Bartos
	# Therefore we include all recorded cells but account for the distance-dependent contribution weighting currents /r^2
	# We assume that the electrode has been placed in the cortical coordinates <tip>
	# Given that the current V1 orientation map has a pixel for each 100 um, a reasonable way to look at a neighborhood is in a radius of 300 um

	print "LFP electrode tip location (x,y) in degrees:", tip

	# Gather vm and conductances
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
	for n,idd in enumerate(ids):
		print "idd", idd
		full_vm = [s.get_vm(idd) for s in segs] # all segments
		full_gsyn_es = [s.get_esyn(idd) for s in segs]
		full_gsyn_is = [s.get_isyn(idd) for s in segs]
		print "len full_gsyn_e", len(full_gsyn_es) # segments = stimuli * trials
		print "shape gsyn_e[0]", full_gsyn_es[0].shape # stimulus lenght
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
		for st in range(num_ticks):
			mean_full_vm[st] = mean_full_vm[st] / trials
			mean_full_gsyn_e[st] = mean_full_gsyn_e[st] / trials
			mean_full_gsyn_i[st] = mean_full_gsyn_i[st] / trials

		pop_vm.append(mean_full_vm)
		pop_gsyn_e.append(mean_full_gsyn_e)
		pop_gsyn_i.append(mean_full_gsyn_i)

	pop_v = numpy.array(pop_vm)
	pop_e = numpy.array(pop_gsyn_e)
	pop_i = numpy.array(pop_gsyn_i)

	# Produce the current for each cell for this time interval, with the Ohm law:
	# I = ge(V-Ee) + gi(V+Ei)
	# where 
	# Ee is the equilibrium for exc, which is 0.0
	# Ei is the equilibrium for inh, which is -80.0
	i = pop_e*(pop_v-0.0) + pop_i*(pop_v-80.0)
	# i = pop_e*(pop_v-0.0) + 0.3*pop_i*(pop_v-80.0)
	# i = pop_e*(pop_v-0.0) # only exc
	# the LFP is the result of cells' currents divided by the distance
	sum_i = numpy.sum(i, axis=0 )
	lfp = sum_i/(4*numpy.pi*sigma) #
	lfp /= 1000. # from milli to micro
	print "LFP:", lfp.shape, lfp.mean(), lfp.min(), lfp.max()
	# print lfp
	# lfp = np.convolve(lfp, np.ones((10,))/10, mode='valid') # moving avg or running mean implemented as a convolution over steps of 10, divided by 10
	# lfp = np.convolve(lfp, np.ones((10,))/10, mode='valid') # moving avg or running mean implemented as a convolution over steps of 10, divided by 10

	#plot the LFP for each stimulus
	for s in range(num_ticks):
		# for each stimulus plot the average conductance per cell over time
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()

		ax.plot( range(0,len(lfp[s])), lfp[s], color=color, linewidth=3 )

		# ax.set_ylim([lfp.min(), lfp.max()])
		ax.set_ylim(ylim)
		ax.set_ylabel( "LFP (uV)" )
		ax.set_xlabel( "Time (us)" )
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.xaxis.set_ticks(ticks, ticks)
		ax.yaxis.set_ticks_position('left')

		# text
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseLFP_"+sheet+"_"+parameter+"_"+str(ticks[s])+"_"+addon+".svg", dpi=200, transparent=True )
		fig.clf()
		plt.close()
		# garbage
		gc.collect()




def trial_averaged_LFP_rate( sheet, folder, stimulus, parameter, start, end, xlabel="", ylabel="", color="black", ylim=[0.,100.], radius=None, addon="" ):
	print inspect.stack()[0][3]
	print "folder: ",folder
	print "sheet: ",sheet
	data_store = PickledDataStore(load=True, parameters=ParameterSet({'root_directory':folder, 'store_stimuli' : False}),replace=True)
	data_store.print_content(full_recordings=False)

	neurons = []
	neurons = param_filter_query(data_store, sheet_name=sheet, st_name=stimulus).get_segments()[0].get_stored_spike_train_ids()
	print "Recorded neurons:", len(neurons)

	### cascading requirements
	if radius:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[sheet]
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)
	####
	# if orientation:
	#	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	# 	l4_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentOrientation', sheet_name=sheet)
	# 	l4_phase = data_store.get_analysis_result(identifier='PerNeuronValue',value_name='LGNAfferentPhase', sheet_name=sheet)
	# 	# print "l4_phase", l4_phase
	# 	neurons = numpy.array([neurons[numpy.argmin([circular_dist(o,numpy.pi/2,numpy.pi) for (o,p) in zip(l4_or[0].get_value_by_id(neurons),l4_phase[0].get_value_by_id(neurons))])] ])

	print "Selected neurons:", len(neurons)#, neurons
	if len(neurons) < 1:
		return

	SpikeCount( 
		param_filter_query(data_store, sheet_name=sheet, st_name=stimulus), 
		ParameterSet({'bin_length':5, 'neurons':list(neurons), 'null':False}) 
		# ParameterSet({'bin_length':bin, 'neurons':list(neurons), 'null':False}) 
	).analyse()
	# datastore.save()
	TrialMean(
		param_filter_query( data_store, name='AnalogSignalList', analysis_algorithm='SpikeCount' ),
		ParameterSet({'vm':False, 'cond_exc':False, 'cond_inh':False})
	).analyse()

	dsvTM = param_filter_query( data_store, sheet_name=sheet, st_name=stimulus, analysis_algorithm='TrialMean' )
	# dsvTM.print_content(full_recordings=False)
	pnvsTM = [ dsvTM.get_analysis_result() ]
	# print pnvsTM
	# get stimuli from PerNeuronValues
	st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvsTM[-1]]

	asl_id = numpy.array([z.get_asl_by_id(neurons) for z in pnvsTM[-1]])
	print asl_id.shape
	# Example:
	# (8, 133, 1029)
	# 8 stimuli
	# 133 cells
	# 1029 bins

	dic = colapse_to_dictionary([z.get_asl_by_id(neurons) for z in pnvsTM[-1]], st, parameter)
	for k in dic:
		(b, a) = dic[k]
		par, val = zip( *sorted( zip(b, numpy.array(a)) ) )
		dic[k] = (par,numpy.array(val))

	stimuli = dic.values()[0][0]
	means = asl_id.mean(axis=1) # mean of
	print means.shape
	# print "means", means, "stimuli", stimuli

	#plot the LFP for each stimulus
	for s in range(0,len(means)):
		# for each stimulus plot the average conductance per cell over time
		matplotlib.rcParams.update({'font.size':22})
		fig,ax = plt.subplots()

		ax.plot( range(0,len(means[s])), means[s], color=color, linewidth=3 )

		# ax.set_ylim([lfp.min(), lfp.max()])
		# ax.set_ylim(ylim)
		ax.set_ylabel( "LFP (uV)" )
		ax.set_xlabel( "Time (us)" )
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		# text
		plt.tight_layout()
		plt.savefig( folder+"/TimecourseLFPrate_"+sheet+"_"+parameter+"_"+str(s)+"_"+addon+".svg", dpi=200, transparent=True )
		fig.clf()
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

	if radius:
		sheet_ids = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=neurons)
		positions = data_store.get_neuron_postions()[sheet]
		if radius:
			ids1 = select_ids_by_position(positions, sheet_ids, radius=radius)
		neurons = data_store.get_sheet_ids(sheet_name=sheet, indexes=ids1)

	NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
	l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=sheet)[0]
	l4_exc_or_many = numpy.array(neurons)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in neurons]) < .1)[0]]
	neurons = list(l4_exc_or_many)

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
	   plot_file_name= folder+"/TrialAveragedSensitivityNew_"+stimulus+"_"+parameter+"_"+str(sheet)+"_"+addon+"_mean.svg"
	).plot({
		# '*.y_lim':(0,30), 
		# '*.x_lim':(-10,100), 
		# '*.x_scale':'log', '*.x_scale_base':10,
		'*.fontsize':17
	})
	return






###################################################
# Execution

full_list = [ 
	# "/media/do/HANGAR/ThalamoCorticalModel_data_size_closed_vsdi_08_radius14_____",
	# "/media/do/Sauvegarde Système/ThalamoCorticalModel_data_size_closed_vsdi_____20trials",
	# "ThalamoCorticalModel_data_size_closed_vsdi_____20trials",
	# "/media/do/DATA/Deliverable/ThalamoCorticalModel_data_size_closed_vsdi_____10trials",
	# "ThalamoCorticalModel_data_size_closed_vsdi_____",
	# "ThalamoCorticalModel_data_size_Yves_____",

	# "ThalamoCorticalModel_data_interrupted_bar_ver1_____",
	# "ThalamoCorticalModel_data_interrupted_bar_ver1_ffw_____",

	"ThalamoCorticalModel_interrupted_bar_ver3_closed_____",
	# "ThalamoCorticalModel_interrupted_bar_ver3_ffw_____",
	# "ThalamoCorticalModel_int_bar_ver3_ffw_____",

	# "ThalamoCorticalModel_annulus_closed_____",
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
# sheets = ['V1_Exc_L4']
# sheets = ['V1_Inh_L4'] 
sheets = ['V1_Exc_L4', 'V1_Inh_L4'] 
# sheets = [ ['V1_Exc_L4', 'V1_Inh_L4'] ]
# sheets = ['V1_Exc_L4', 'V1_Inh_L4', 'X_OFF', 'PGN'] 



for i,f in enumerate(full_list):
	print i,f

	# color = "saddlebrown"
	color = "black"

	for s in sheets:
		print color


		# # # 
		# trial_averaged_LFP_rate(
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='DriftingSinusoidalGratingRing',
		# 	parameter="inner_appareture_radius",
		# 	start=0., 
		# 	end=2000., 
		# 	xlabel="gap", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color=color, 
		# 	radius = [.0,.1], # center
		# )

		# LFP(
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='DriftingSinusoidalGratingRing',
		# 	parameter="inner_appareture_radius",
		# 	ylim=[-4., 0.0],
		# 	tip = [.0,.0,.0], # CENTER box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	# tip_box = [[0.,0.],[0.,0.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	# tip_box = [[0.,0.],[0.,0.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	sigma = .4, # spikes from the center of spike_sheet
		# 	# sigma = 0.1, # [0.1, 0.01] # Dobiszewski_et_al2012.pdf  mm, since 1mm = 1deg in this cortical space 
		# 	# sigma = 0.25, # 0.25 Bartos et al.
		# 	addon = addon + "_center",
		# 	color = color,
		# )

		# trial_averaged_tuning_curve_errorbar( 
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='DriftingSinusoidalGratingRing',
		# 	parameter="inner_appareture_radius",
		# 	start=100., 
		# 	end=2000., 
		# 	xlabel="inner radius", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color=color, 
		# 	useXlog=False, 
		# 	useYlog=False, 
		# 	percentile=False,
		# 	# ylim=[0,50],
		# 	opposite=False, # to select cells with SAME orientation preference
		# 	radius = [.0,.7], # center
		# 	addon = addon,
		# )

		# 
		# trial_averaged_LFP_rate(
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='FlashedInterruptedBar',
		# 	parameter="gap_length",
		# 	bin=10, # us, so in ms
		# 	start=0., 
		# 	end=2000., 
		# 	xlabel="gap", 
		# 	ylabel="firing rate (sp/s)", 
		# 	color=color, 
		# 	radius = [.0,.9], # center
		# )

		trial_averaged_tuning_curve_errorbar( 
			sheet=s, 
			folder=f, 
			stimulus='FlashedInterruptedBar',
			parameter="gap_length",
			start=100., 
			end=2000., 
			xlabel="gap", 
			ylabel="firing rate (sp/s)", 
			color=color, 
			useXlog=False, 
			useYlog=False, 
			percentile=False,
			# ylim=[0,50],
			opposite=False, # to select cells with SAME orientation preference
			radius = [.0,.5], # center
			addon = addon,
		)

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

		# LFP(
		# 	sheet=s, 
		# 	folder=f, 
		# 	stimulus='FlashedInterruptedBar',
		# 	parameter="gap_length",
		# 	ylim=[-4., 0.0],
		# 	tip = [.0,.0,.0], # CENTER box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	# tip_box = [[0.,0.],[0.,0.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	# tip_box = [[0.,0.],[0.,0.]], # box in which to search a centroid for LFP to measure the conductance effect in lfp_sheet
		# 	sigma = .4, # spikes from the center of spike_sheet
		# 	# sigma = 0.1, # [0.1, 0.01] # Dobiszewski_et_al2012.pdf  mm, since 1mm = 1deg in this cortical space 
		# 	# sigma = 0.25, # 0.25 Bartos et al.
		# 	addon = addon + "_center",
		# 	color = color,
		# )
		
