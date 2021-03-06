#!/usr/local/bin/ipython -i 
import numpy as np
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from mozaik.sheets.population_selector import RCSpace
from parameters import ParameterSet


def create_experiments_spontaneous(model):
  return [
      # SPONTANEOUS ACTIVITY
      # as in LevickWilliams1964, WebbTinsleyBarracloughEastonParkerDerrington2002, (TODO: TsumotoCreutzfeldtLegendy1978)
      # NoStimulation( model, duration=147*7 ), (darkness)
      # PoissonNetworkKick(model,duration=8*8*7,drive_period=200.0,sheet_list=["V1_Exc_L4","V1_Inh_L4"],stimulation_configuration={'component' : 'mozaik.sheets.population_selector.RCRandomPercentage','params' : {'percentage' : 100.0}},lambda_list=[400.0,400.0,400.0,400.0],weight_list=[0.001,0.001,0.001,0.001]),
      MeasureSpontaneousActivity( model, num_trials=6, duration=147*7 )
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
          # spatial_frequencies = [0.07, 0.5, 1.], # test
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
          num_orientations=1, # Horizontal-only. Num_orientations is converted to factors of rad (numpy.pi/num_orientations*i)
          # num_orientations=10, # rad: [0.0, 1.5707...]  (0, 90 deg)
          spatial_frequency=0.5,
          temporal_frequency=2.0,
          grating_duration=1*147*7,
          contrasts=[80],
          num_trials=15
      )
  ]


def create_experiments_size(model):
  return [
      NoStimulation( model, duration=147*7 ),
      # MeasureSpontaneousActivity( model, num_trials=6, duration=147*7 ),

      # SIZE TUNING
      # as in ClelandLeeVidyasagar1983, BoninManteCarandini2005
      MeasureSizeTuning(
          model, 
          num_sizes=1, # 10,
          # max_size=0.1, # radius, Yves's test of response at RF-sized stimuli
          # max_size=1.4, # radius 
          max_size=2., # radius, large-field
          orientations=[0.0], 
          # orientations=[0.0, numpy.pi/8, numpy.pi/4, 3*numpy.pi/8, numpy.pi/2], 
          spatial_frequency=0.5,
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7, # 1 sec
          contrasts=[80], 
          num_trials=15, #6
          log_spacing=False, # True
          with_flat=False #use also flat luminance discs
      )

      # NoStimulation( model, duration=147*7 )
      # MeasureSpontaneousActivity( model, num_trials=6, duration=147*7 ),
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
          orientations=[0.], 
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
          orientations=[numpy.pi/2], 
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
            'component':'mozaik.sheets.population_selector.RCAnnulus', 
            'params':{'inner_radius':0.0, 'outer_radius':0.4}
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=5.0, # max radius
          orientations=[0.], #numpy.pi/2, 
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
            'component':'mozaik.sheets.population_selector.RCAnnulus', 
            'params':{'inner_radius':0.4, 'outer_radius':1.0} 
          },
          injection_current=-.5, # nA
          num_sizes=10, 
          max_size=5.0, # max radius
          orientations=[numpy.pi/2], 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=1*147*7,
          contrasts=[80], #40,100  to look for contrast-dependent RF expansion
          num_trials=6,
          log_spacing=True
      )
  ]



def create_experiments_annulus(model):
  return [
      NoStimulation( model, duration=147*7 ),
      MeasureAnnulusTuning(
          model, 
          num_sizes=10, 
          max_inner_diameter=3., # deg
          outer_diameter=10.0, # max radius
          orientation=0, 
          log_spacing=False, 
          spatial_frequency=0.5, #
          temporal_frequency=2.0, # optimal for LGN: 8. # optimal for V1: 2.
          grating_duration=2*147*7,
          contrasts=[100], #40,100  to look for contrast-dependent RF expansion
          num_trials=10
      )
  ]



def create_interrupted_bar(model):
  return [
      # NoStimulation( model, duration=147*7 ),

      # ## Control
      # MapResponseToInterruptedBarStimulus(
      #   model,
      #   x=0,
      #   y=0,
      #   length= 20,
      #   width= 1/0.8/4.0,
      #   orientation=0,
      #   max_offset= 1/0.8/2.0 * 1.5,
      #   steps= 8,
      #   duration= 700,
      #   flash_duration= 500, 
      #   relative_luminances= [1.0],
      #   num_trials= 10,
      #   gap_lengths= [0.0]
      # )

      # ## version 1
      # MapResponseToInterruptedBarStimulus(
      #   model,
      #   x=0,
      #   y=0,
      #   length= 1/0.8/2.0 * 12.0,
      #   width= 1/0.8/4.0,
      #   orientation=0,
      #   max_offset= 1/0.8/2.0 * 1,
      #   steps= 3,
      #   duration= 700,
      #   flash_duration= 500, 
      #   relative_luminances= [1.0],
      #   num_trials= 1,
      #   gap_lengths= [0.0,0.1,0.2,0.4,0.8,1.6,3.2]
      # )

      # ## version 2
      # MapResponseToInterruptedBarStimulus(
      #   model,
      #   x=0,
      #   y=0,
      #   length= 1/0.8/2.0 * 12.0,
      #   width= 1/0.8/4.0,
      #   orientation=0,
      #   max_offset= 1/0.8/2.0 * 1,
      #   steps= 3,
      #   duration= 700,
      #   flash_duration= 500, 
      #   relative_luminances= [0.55],
      #   num_trials= 1,
      #   gap_lengths= [0.0,0.1,0.2,0.4,0.8,1.6,3.2]
      # )

      # ## version 3
      # MapResponseToInterruptedBarStimulus(
      #   model,
      #   x=0,
      #   y=0,
      #   length= 20,
      #   width= 1/0.8/4.0,
      #   orientation=0,
      #   max_offset= 1/0.8/2.0 * 1,
      #   steps= 1,
      #   duration= 2*143*7,
      #   flash_duration= 2*143*7, 
      #   relative_luminances= [1.0],
      #   num_trials= 10,
      #   gap_lengths=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.6,3.2]
      # )

      ## version latest
      MapResponseToInterruptedBarStimulus(
        model,
        x=0,
        y=0,
        length= 20,
        width= 1/0.8/4.0,
        orientation=0,
        max_offset= 2.4*1/0.8/4.0,
        steps= 3,
        duration= 2*143*7,
        flash_duration= 2*143*7, 
        relative_luminances= [1.0],
        num_trials= 10,
        gap_lengths=[0.4],
        disalignment=[0,0.05,0.1,0.2,0.3,0.4],
      )

  ]


# LIFELONG SPARSENESS
# as in RathbunWarlandUsrey2010, AndolinaJonesWangSillito2007
# stimulation as Size Tuning

