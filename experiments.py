#!/usr/local/bin/ipython -i 
import numpy as np
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from mozaik.sheets.population_selector import RCSpace
from parameters import ParameterSet


def create_experiments_spontaneous(model):
  return [
      # SPONTANEOUS ACTIVITY (darkness)
      # as in LevickWilliams1964, WebbTinsleyBarracloughEastonParkerDerrington2002, (TODO: TsumotoCreutzfeldtLegendy1978)
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 )
  ]


def create_experiments_luminance(model):
  return [
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # LUMINANCE SENSITIVITY
      # as in PapaioannouWhite1972
      MeasureFlatLuminanceSensitivity(
          model, 
          luminances=[.0, .00001, .0001, .001, .01, .1, 1., 10., 100.], # as in BarlowLevick1969, SakmannCreutzfeldt1969
          # luminances=[0.0, 0.085, 0.85, 8.5, 85.0], # as in PapaioannouWhite1972
          step_duration=2*147*7,
          num_trials=6
      )
  ]


def create_experiments_contrast(model):
  return [
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # CONTRAST SENSITIVITY
      # as in DerringtonLennie1984, HeggelundKarlsenFlugsrudNordtug1989, SaulHumphrey1990, BoninManteCarandini2005
      MeasureContrastSensitivity(
          model, 
          size = 20.0,
          orientation=0., 
          spatial_frequency = 0.5, 
          temporal_frequency = 2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration = 2*147*7,
          # contrasts = [0, 10, 25, 40, 75, 100], # Bonin Mante Carandini 2005
          # contrasts = [0, 2, 4, 8, 18, 36, 50, 100], # KaplanPurpuraShapley1987
          contrasts = [0.0, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.], # LiYeSongYangZhou2011
          num_trials=6
      )
  ]


def create_experiments_spatial(model):
  return [
      #Lets kick the network up into activation
      #PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # SPATIAL FREQUENCY TUNING
      # as in SolomonWhiteMartin2002, SceniakChatterjeeCallaway2006
      MeasureFrequencySensitivity(
          model, 
          orientation=0., 
          contrasts=[80], #[25,50,100], #
          spatial_frequencies = [0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5,  2., 8.], # KimuraShimegiHaraOkamotoSato2013
          # spatial_frequencies = [0.2, 0.3],
          # spatial_frequencies=np.arange(0.0, 3., 0.2),
          temporal_frequencies=[2.0], # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=2*147*7,
          frame_duration=7,
          # square=True,
          num_trials=6
      )
  ]


def create_experiments_temporal(model):
  return [
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # TEMPORAL FREQUENCY TUNING
      # as in SaulHumphrey1990, AlittoUsrey2004
      MeasureFrequencySensitivity(
          model, 
          orientation=0., 
          contrasts=[80], 
          spatial_frequencies=[0.5], 
          temporal_frequencies=[0.05, 0.2, 1.2, 3.0, 6.4, 8, 12, 30], # AlittoUsrey2004
          #temporal_frequencies=[0.2, .8, 2.4, 6.0, 12.], # DerringtonLennie1982
          grating_duration=10*147*7,
          frame_duration=7,
          #square=True,
          num_trials=1
      )
  ]




def create_experiments_orientation(model):
  return [
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # ORIENTATION TUNING (GRATINGS)
      # as in DanielsNormanPettigrew1977, VidyasagarUrbas1982
      MeasureOrientationTuningFullfield(
          model,
          num_orientations=10, # rad: [0.0, 1.5707...]  (0, 90 deg)
          spatial_frequency=0.5,
          temporal_frequency=2.0,
          grating_duration=1*147*7,
          contrasts=[80],
          num_trials=2
      )
  ]




def create_experiments_size(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=10, # 10,
          max_size=5.0, # radius
          orientation=0., #numpy.pi/2, 
          spatial_frequency=0.5,
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7, # 1 sec
          contrasts=[80], 
          num_trials=6,
          log_spacing=True,
          with_flat=False #use also flat luminance discs
      )
  ]

def create_experiments_size_overlapping(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=10, # 10,
          max_size=5.0, # radius
          orientation=0., 
          spatial_frequency=0.5,
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7, # 1 sec
          contrasts=[80], 
          num_trials=6,
          log_spacing=True,
          with_flat=False #use also flat luminance discs
      )
  ]

def create_experiments_size_nonoverlapping(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=10, # 10,
          max_size=5.0, # radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5,
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7, # 1 sec
          contrasts=[80], 
          num_trials=6,
          log_spacing=True,
          with_flat=False #use also flat luminance discs
      )
  ]

def create_experiments_size_V1_inactivated_overlapping(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in Jones et al. 2012: inactivation of a region corresponding to 0.5 deg in cortex
      MeasureSizeTuningWithInactivation(
          model, 
          sheet_list=["V1_Exc_L4"],
          injection_configuration={
            'component':'mozaik.sheets.population_selector.RCSpace', 
            'params':{'radius':0.4, 'offset_x':0.0, 'offset_y':0.0}
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=5.0, # max radius
          orientation=0., #numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]

def create_experiments_size_V1_inactivated_nonoverlapping(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in Jones et al. 2012: inactivation of a region corresponding to 0.5 deg in cortex
      MeasureSizeTuningWithInactivation(
          model, 
          sheet_list=["V1_Exc_L4"],
          injection_configuration={
            'component':'mozaik.sheets.population_selector.RCSpace', 
            'params':{'radius':0.4, 'offset_x':0.0, 'offset_y':600.0} 
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=5.0, # max radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]




def create_experiments_brokenbar(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # CONTOUR COMPLETION
      # as in Mumford1991
      MeasureBrokenBarResponse(
          model, 
          relative_luminance = 0., # 0. is dark, 1.0 is double the background luminance
          orientation = 0., # 0, pi
          width = 0.5, # cpd
          length = 10., # cpd, length of the bar
          flash_duration = 300., # ms, duration of the bar presentation
          gap_radius = .5, # degrees, radius of the gap
          x = .0, # degrees, x location of the center of the bar
          y = .0, # degrees, y location of the center of the bar
          duration = 100*7,
          frame_duration = 7, # ms
          num_trials = 1
      ),
  ]





def create_experiments_correlation(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # CONTOUR COMPLETION
      # as in SillitoJonesGersteinWest1994, SillitoCudeiroMurphy1993
      MeasureFeatureInducedCorrelation(
          model, 
          contrast=80, 
          spatial_frequencies=[0.5],
          separation=1.5, #[1,2,3,4] # SillitoJonesGersteinWest1994 (at 13deg eccentricity) # degrees of visual field
          temporal_frequency=2.0,
          exp_duration=1*147*7,
          frame_duration=7,
          num_trials=2,
      ),
  ]



# LIFELONG SPARSENESS
# as in RathbunWarlandUsrey2010, AndolinaJonesWangSillito2007
# stimulation as Size Tuning

