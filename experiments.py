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
          # luminances=[.0, .00001, .0001, .001, .01, .1, 1., 10., 100.], # as in BarlowLevick1969, SakmannCreutzfeldt1969
          luminances=[0.0, 0.085, 0.85, 8.5, 85.0], # as in PapaioannouWhite1972
          step_duration=2*147*7,
          num_trials=2
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
          orientation=numpy.pi/2, 
          spatial_frequency = 0.5, 
          temporal_frequency = 2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration = 2*147*7,
          contrasts = [0, 10, 25, 40, 75, 100], # Bonin Mante Carandini 2005
          # contrasts = [0, 2, 4, 8, 18, 36, 50, 100], # KaplanPurpuraShapley1987
          num_trials=2
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
          orientation=numpy.pi/2, 
          contrasts=[80], #[25,50,100], #
          spatial_frequencies = [0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.5,  2., 8.], # KimuraShimegiHaraOkamotoSato2013
          # spatial_frequencies = [0.2, 0.3],
          # spatial_frequencies=np.arange(0.0, 3., 0.2),
          temporal_frequencies=[2.0], # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=2*147*7,
          frame_duration=7,
          # square=True,
          num_trials=5
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
          orientation=numpy.pi/2, 
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


def create_experiments_size(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=10, 
          max_size=6.0, # max radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True,
          with_flat=False #use also flat luminance discs
      )
  ]


def create_experiments_size_V1_inactivated_small(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in Jones et al. 2012: inactivation of a region corresponding to 0.5 deg in cortex
      MeasureSizeTuningWithInactivation(
          model, 
          sheet_list=["V1_Exc_L4"],
          injection_configuration={
            'component':'mozaik.sheets.population_selector.RCSpace', 
            'params':{'radius':0.15, 'offset_x':0.0, 'offset_y':0.0}
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=6.0, # max radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]
def create_experiments_size_V1_inactivated_large(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in Jones et al. 2012: inactivation of a region corresponding to 0.5 deg in cortex
      MeasureSizeTuningWithInactivation(
          model, 
          sheet_list=["V1_Exc_L4"],
          injection_configuration={
            'component':'mozaik.sheets.population_selector.RCSpace', 
            'params':{'radius':0.3, 'offset_x':0.0, 'offset_y':0.0}
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=6.0, # max radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]

def create_experiments_size_V1_inactivated_large_nonoverlapping(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in Jones et al. 2012: inactivation of a region corresponding to 0.5 deg in cortex
      MeasureSizeTuningWithInactivation(
          model, 
          sheet_list=["V1_Exc_L4"],
          injection_configuration={
            'component':'mozaik.sheets.population_selector.RCSpace', 
            'params':{'radius':0.3, 'offset_x':0.0, 'offset_y':1000.0} # at 1.6 deg right (see or_map_6x6)
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=6.0, # max radius
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]



# ------------------------------------------



def create_experiments_combined(model):
  # return create_experiments_luminance(model) + create_experiments_contrast(model) + create_experiments_spatial(model) #+ create_experiments_temporal(model)
  return create_experiments_spatial(model) + create_experiments_temporal(model)



# ------------------------------------------



def create_experiments_orientation(model):
  return [
      #Lets kick the network up into activation
      PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # ORIENTATION TUNING (GRATINGS)
      # as in DanielsNormanPettigrew1977, VidyasagarUrbas1982
      MeasureOrientationTuningFullfield(
          model,
          num_orientations=2, # rad: [0.0, 1.5707...]  (0, 90 deg)
          spatial_frequency=0.8, #0.5,
          temporal_frequency=2.0, #8.0,
          grating_duration=2*147*7,
          contrasts=[100],
          num_trials=2
      )
  ]


  # #Spontaneous Activity 
  # NoStimulation(model,duration=2*2*5*3*8*7),
  # # Measure orientation tuning with full-filed sinusoidal gratins
  # MeasureOrientationTuningFullfield(model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=2*147*7,contrasts=[100],num_trials=2),

  # # Measure response to natural image with simulated eye movement
  # MeasureNaturalImagesWithEyeMovement(model,stimulus_duration=2*147*7,num_trials=2),



def create_experiments_correlation(model):
  return [
      #Lets kick the network up into activation
      PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # CONTOUR COMPLETION
      # as in SillitoJonesGersteinWest1994, SillitoCudeiroMurphy1993
      # By default, for this experiment only, the visual space ('size' parameter in the SpatioTemporalFilterRetinaLGN_default file)
      # is reduced to a flat line in order to have an horizontal distribution of neurons.
      # A separation distance is established and the experimental protocol finds the closest neurons to the distance specified.
      MeasureFeatureInducedCorrelation(
          model, 
          contrast=80, 
          spatial_frequencies=[0.5],
          separation=6,
          temporal_frequency=8.0,
          exp_duration=147*7,
          frame_duration=7,
          num_trials=8,
      ),
  ]



# LIFELONG SPARSENESS
# as in RathbunWarlandUsrey2010, AndolinaJonesWangSillito2007
# stimulation as Size Tuning

# IMAGES WITH EYEMOVEMENT
#MeasureNaturalImagesWithEyeMovement(
#  model,
#  stimulus_duration=147*7 *3,
#  num_trials=5
#),

# GRATINGS WITH EYEMOVEMENT
# MeasureDriftingSineGratingWithEyeMovement(
#   model, 
#   spatial_frequency=0.8,
#   temporal_frequency=2,
#   stimulus_duration=147*7,
#   num_trials=10,
#   contrast=100
# ),
