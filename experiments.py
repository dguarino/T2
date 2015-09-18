#!/usr/local/bin/ipython -i 
import numpy as np
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet

    
def create_experiments_spontaneous(model):
  return [
      # SPONTANEOUS ACTIVITY (darkness)
      # as in LevickWilliams1964, WebbTinsleyBarracloughEastonParkerDerrington2002, (TODO: TsumotoCreutzfeldtLegendy1978)
      #Lets kick the network up into activation
      PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
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
          # luminances=[0.01, 20.0, 50.0, 100.0],
          luminances=[0.01, 0.1, 1.0, 10.0, 20.0, 50.0, 70.0, 100.0],
          step_duration=147*7,
          num_trials= 54
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
          temporal_frequency = 2.5, 
          grating_duration = 147*7,
          contrasts = [80], #[0, 25, 50, 75, 100],
          num_trials = 1
      )
  ]


def create_experiments_spatial(model):
  return [
      #Lets kick the network up into activation
      #PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # SPATIAL FREQUENCY TUNING (with different contrasts)
      # as in SolomonWhiteMartin2002, SceniakChatterjeeCallaway2006
      MeasureFrequencySensitivity(
          model, 
          orientation=numpy.pi/2, 
          contrasts=[80], #[25,50,100], #
          # spatial_frequencies = [0.1, 0.2, 0.5, 1., 2.], #[0.16, 0.24, 0.3, 0.4, 0.5, 0.8, 1., 2.],
          spatial_frequencies=np.arange(0.0, 3., 0.2),
          temporal_frequencies=[2.5],
          grating_duration=147*7,
          frame_duration=7,
          # square=True,
          num_trials=14
      )
  ]


def create_experiments_temporal(model):
  return [
      #Lets kick the network up into activation
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # TEMPORAL FREQUENCY TUNING (with different contrasts)
      # as in SaulHumphrey1990, AlittoUsrey2004
      MeasureFrequencySensitivity(
          model, 
          orientation=numpy.pi/2, 
          contrasts=[80], #[25,50,100], #
          spatial_frequencies=[0.5], #[0.1, 0.5, 0.9], 
          temporal_frequencies=[0.8, 1.2, 2.5, 5.1, 6.4, 8.0, 12., 16.],
          grating_duration=147*7,
          frame_duration=7,
          #square=True,
          num_trials=14
      )
  ]


def create_experiments_size(model):
  return [
      #Lets kick the network up into activation
      #PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=10, 
          max_size=10.0, 
          orientation=numpy.pi/2, 
          spatial_frequency=0.5, #0.8
          temporal_frequency=2.5,
          grating_duration=147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=2,
          log_spacing=True,
          with_flat=False #use also flat luminance discs
      )
  ]


def create_experiments_orientation(model):
  return [
      #Lets kick the network up into activation
      PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      NoStimulation( model, duration=147*7 ),
      # ORIENTATION TUNING (GRATINGS)
      # as in DanielsNormanPettigrew1977, VidyasagarUrbas1982
      MeasureOrientationTuningFullfield(
          model,
          num_orientations=8,
          spatial_frequency=0.5,
          temporal_frequency=8.0,
          grating_duration=147*7,
          contrasts=[80],
          num_trials=4
      )
      # MeasureOrientationTuningFullfield(model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=2*147*7,contrasts=[100],num_trials=2),
  ]


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


# ------------------------------------------

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
