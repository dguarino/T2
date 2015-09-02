import numpy
import mozaik
import pylab
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.circ_stat import circular_dist
from mozaik.controller import Global

# import sys
# sys.path.append('/home/antolikjan/dev/pkg/mozaik/mozaik/contrib')
# import Kremkow_plots
# from Kremkow_plots import *


logger = mozaik.getMozaikLogger()



def perform_analysis_and_visualization( data_store, withPGN=False, withV1=False ):
    # Gather indexes
    analog_Xon_ids = sorted( param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_vm_ids() )
    analog_Xoff_ids = sorted( param_filter_query(data_store,sheet_name="X_OFF").get_segments()[0].get_stored_vm_ids() )
    print "analog_Xon_ids: ",analog_Xon_ids
    print "analog_Xoff_ids: ",analog_Xoff_ids

    analog_PGN_ids = None
    if withPGN:
        analog_PGN_ids = sorted( param_filter_query(data_store,sheet_name="PGN").get_segments()[0].get_stored_vm_ids() )
        print "analog_PGN_ids: ",analog_PGN_ids

    analog_ids = None
    if withV1:
        analog_ids = sorted( param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_vm_ids() )
        spike_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_spike_train_ids()
        print "analog_ids: ",analog_ids
        print "spike_ids: ",spike_ids
        if False:
            sheets = list(set(data_store.sheets()) & set(['V1_Exc_L4','V1_Inh_L4']))
            exc_sheets = list(set(data_store.sheets()) & set(['V1_Exc_L4']))
            l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')
            l4_exc_phase = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentPhase', sheet_name = 'V1_Exc_L4')
            l4_exc = analog_ids[numpy.argmin([circular_dist(o,numpy.pi/2,numpy.pi)  for (o,p) in zip(l4_exc_or[0].get_value_by_id(analog_ids),l4_exc_phase[0].get_value_by_id(analog_ids))])]
            l4_inh_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Inh_L4')
            l4_inh_phase = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentPhase', sheet_name = 'V1_Inh_L4')
            l4_inh = analog_ids_inh[numpy.argmin([circular_dist(o,numpy.pi/2,numpy.pi)  for (o,p) in zip(l4_inh_or[0].get_value_by_id(analog_ids_inh),l4_inh_phase[0].get_value_by_id(analog_ids_inh))])]
            l4_exc_or_many = numpy.array(l4_exc_or[0].ids)[numpy.nonzero(numpy.array([circular_dist(o,0,numpy.pi)  for (o,p) in zip(l4_exc_or[0].values,l4_exc_phase[0].values)]) < 0.1)[0]]
            l4_exc_or_many = list(set(l4_exc_or_many) &  set(spike_ids))
            orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(data_store,st_name='FullfieldDriftingSinusoidalGrating',st_contrast=100).get_stimuli()]))                
            l4_exc_or_many_analog = numpy.array(analog_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or[0].get_value_by_id(i),0,numpy.pi)  for i in analog_ids]) < 0.1)[0]]
            l4_inh_or_many_analog = numpy.array(analog_ids_inh)[numpy.nonzero(numpy.array([circular_dist(l4_inh_or[0].get_value_by_id(i),0,numpy.pi)  for i in analog_ids_inh]) < 0.1)[0]]
    
    # NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
    # data_store.print_content(full_ADS=True)

    ##############
    # ANALYSES
    # perform_analysis_luminance( data_store, withPGN, withV1 )
    # perform_analysis_contrast_frequency( data_store, withPGN, withV1 )
    perform_analysis_size( data_store, withPGN, withV1 )
    # perform_analysis_and_visualization_subcortical_conn(data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids)
    # perform_analysis_and_visualization_cortical_or_conn( data_store )

    ##############
    # PLOTTING
    activity_plot_param =    {
           'frame_rate' : 5,  
           'bin_width' : 5.0, 
           'scatter' :  True,
           'resolution' : 0
    }
    plot_overview( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids )
    # plot_luminance_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)
    # plot_contrast_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)
    # plot_spatial_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)
    # plot_temporal_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)
    plot_size_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids )


#########################################################################################
#########################################################################################
#########################################################################################




def perform_analysis_and_visualization_subcortical_conn(data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None):
    # CONNECTIVITY PLOT
    # LGN On -> PGN: 'LGN_PGN_ConnectionOn'
    ConnectivityPlot(
        data_store,
        ParameterSet({
            'neuron': analog_Xon_ids[0],  # the target neuron whose connections are to be displayed
            'reversed': False,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
            'sheet_name': 'X_ON',  # for neuron in which sheet to display connectivity
        }),
        fig_param={'dpi':100, 'figsize': (10,12)},
        plot_file_name='LGN_On_'+str(analog_Xon_ids[0])+'_outgoing.png'
    ).plot()    
    # LGN Off -> PGN: 'LGN_PGN_ConnectionOff'
    ConnectivityPlot(
        data_store,
        ParameterSet({
            'neuron': analog_Xoff_ids[0],  # the target neuron whose connections are to be displayed
            'reversed': False,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
            'sheet_name': 'X_OFF',  # for neuron in which sheet to display connectivity
        }),
        fig_param={'dpi':100, 'figsize': (10,12)},
        plot_file_name='LGN_Off_'+str(analog_Xoff_ids[0])+'_outgoing.png'
    ).plot()    
    # PGN lateral: 'PGN_PGN_Connection'
    ConnectivityPlot(
        data_store,
        ParameterSet({
            'neuron': analog_PGN_ids[0],  # the target neuron whose connections are to be displayed
            'reversed': False,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
            'sheet_name': 'PGN',  # for neuron in which sheet to display connectivity
        }),
        fig_param={'dpi':100, 'figsize': (24,12)},
        plot_file_name='PGN_Connections.png'
    ).plot()    
    # PGN -> LGN On: 'PGN_LGN_ConnectionOn'
    ConnectivityPlot(
        data_store,
        ParameterSet({
            'neuron': analog_Xon_ids[0],  # the target neuron whose connections are to be displayed
            'reversed': True,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
            'sheet_name': 'X_ON',  # for neuron in which sheet to display connectivity
        }),
        fig_param={'dpi':100, 'figsize': (10,12)},
        plot_file_name='LGN_On_'+str(analog_Xon_ids[0])+'_incoming.png'
    ).plot()    
    # PGN -> LGN On: 'PGN_LGN_ConnectionOff'
    ConnectivityPlot(
        data_store,
        ParameterSet({
            'neuron': analog_Xoff_ids[0],  # the target neuron whose connections are to be displayed
            'reversed': True,  # False: outgoing connections from the given neuron are shown. True: incoming connections are shown
            'sheet_name': 'X_OFF',  # for neuron in which sheet to display connectivity
        }),
        fig_param={'dpi':100, 'figsize': (10,12)},
        plot_file_name='LGN_Off_'+str(analog_Xoff_ids[0])+'_incoming.png'
    ).plot()



def perform_analysis_and_visualization_cortical_or_conn(data_store):
    analog_ids = sorted( param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_vm_ids() )
    spike_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_spike_train_ids()
    print "analog_ids: ",analog_ids
    print "spike_ids: ",spike_ids

    NeuronAnnotationsToPerNeuronValues( data_store, ParameterSet({}) ).analyse()
    l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name='V1_Exc_L4')[0]

    l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in spike_ids]) < 0.1)[0]]

    dsv = param_filter_query(data_store,identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation')
    mozaik.visualization.plotting.ConnectivityPlot(data_store,ParameterSet({'neuron' : l4_exc_or_many[0], 'reversed' : True,'sheet_name' : 'V1_Exc_L4'}),pnv_dsv=dsv,fig_param={'dpi' : 100,'figsize': (34,12)},plot_file_name='ExcConnections1.png').plot()
    mozaik.visualization.plotting.ConnectivityPlot(data_store,ParameterSet({'neuron' : l4_exc_or_many[1], 'reversed' : True,'sheet_name' : 'V1_Exc_L4'}),pnv_dsv=dsv,fig_param={'dpi' : 100,'figsize': (34,12)},plot_file_name='ExcConnections2.png').plot()
    mozaik.visualization.plotting.ConnectivityPlot(data_store,ParameterSet({'neuron' : l4_exc_or_many[2], 'reversed' : True,'sheet_name' : 'V1_Exc_L4'}),pnv_dsv=dsv,fig_param={'dpi' : 100,'figsize': (34,12)},plot_file_name='ExcConnections3.png').plot()
    mozaik.visualization.plotting.ConnectivityPlot(data_store,ParameterSet({'neuron' : l4_exc_or_many[3], 'reversed' : True,'sheet_name' : 'V1_Exc_L4'}),pnv_dsv=dsv,fig_param={'dpi' : 100,'figsize': (34,12)},plot_file_name='ExcConnections4.png').plot()
    mozaik.visualization.plotting.ConnectivityPlot(data_store,ParameterSet({'neuron' : l4_exc_or_many[4], 'reversed' : True,'sheet_name' : 'V1_Exc_L4'}),pnv_dsv=dsv,fig_param={'dpi' : 100,'figsize': (34,12)},plot_file_name='ExcConnections5.png').plot()
    


def perform_analysis_luminance( data_store, withPGN=False, withV1=False ):
    dsv0_Xon = param_filter_query( data_store, st_name='Null', sheet_name='X_ON' )  
    TrialAveragedFiringRate( dsv0_Xon, ParameterSet({}) ).analyse()
    dsv0_Xoff = param_filter_query( data_store, st_name='Null', sheet_name='X_OFF' )  
    TrialAveragedFiringRate( dsv0_Xoff, ParameterSet({}) ).analyse()
    if withPGN:
        dsv0_PGN = param_filter_query( data_store, st_name='Null', sheet_name='PGN' )  
        TrialAveragedFiringRate( dsv0_Xoff, ParameterSet({}) ).analyse()
    if withV1:
        dsv0_V1e = param_filter_query( data_store, st_name='Null', sheet_name='V1_Exc_L4' )  
        TrialAveragedFiringRate( dsv0_V1e, ParameterSet({}) ).analyse()
        # dsv0_V1i = param_filter_query( data_store, st_name='Null', sheet_name='V1_Inh_L4' )  
        # TrialAveragedFiringRate( dsv0_V1i, ParameterSet({}) ).analyse()



def perform_analysis_contrast_frequency( data_store, withPGN=False, withV1=False ):
    dsv10 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='X_ON' )  
    TrialAveragedFiringRate( dsv10, ParameterSet({}) ).analyse()
    dsv11 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='X_OFF' )  
    TrialAveragedFiringRate( dsv11, ParameterSet({}) ).analyse()
    if withPGN:
        dsv12 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='PGN' )  
        TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()
    if withV1:
        dsv1_V1e = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='V1_Exc_L4' )  
        TrialAveragedFiringRate( dsv1_V1e, ParameterSet({}) ).analyse()
        # dsv1_V1i = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='V1_Inh_L4' )  
        # TrialAveragedFiringRate( dsv1_V1i, ParameterSet({}) ).analyse()



def perform_analysis_size( data_store, withPGN=False, withV1=False ):
    dsv10 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='X_ON' )  
    TrialAveragedFiringRate( dsv10, ParameterSet({}) ).analyse()
    dsv11 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='X_OFF' )  
    TrialAveragedFiringRate( dsv11, ParameterSet({}) ).analyse()
    # dsv12 = param_filter_query( data_store, st_name='FlatDisk', sheet_name='X_ON' )  
    # TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()
    # dsv13 = param_filter_query( data_store, st_name='FlatDisk', sheet_name='X_OFF' )  
    # TrialAveragedFiringRate( dsv13, ParameterSet({}) ).analyse()
    if withPGN:
        dsv12 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='PGN' )  
        TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()
    if withV1:
        dsv11 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='V1_Exc_L4' )  
        TrialAveragedFiringRate( dsv11, ParameterSet({}) ).analyse()



#########################################################################################
#########################################################################################
#########################################################################################

def plot_luminance_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against luminance levels              
    dsv = param_filter_query( data_store, st_name='Null', analysis_algorithm=['TrialAveragedFiringRate'] )
    PlotTuningCurve(
       dsv,
       ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'background_luminance', 
            'neurons': list(analog_Xon_ids[0:1]), 
            'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="FlatLuminanceSensitivity_LGN_On.png"
    ).plot({
       '*.y_lim':(0,30), 
       # '*.x_lim':(-10,100), 
       '*.fontsize':17
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'background_luminance', 
            'neurons': list(analog_Xoff_ids[0:1]), 
            'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="FlatLuminanceSensitivity_LGN_Off.png"
    ).plot({
       '*.y_lim':(0,30), 
       # '*.x_lim':(-10,100), 
       '*.fontsize':17
    })
    if analog_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'background_luminance', 
                'neurons': list(analog_ids), 
                'sheet_name' : 'V1_Exc_L4'
           }), 
           fig_param={'dpi' : 100,'figsize': (30,8)}, 
           plot_file_name="FlatLuminanceSensitivity_V1e.png"
        ).plot({
           '*.y_lim':(0,30), 
           # '*.x_lim':(-10,100), 
           '*.fontsize':17
        })



def plot_contrast_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # CONTRAST SENSITIVITY analog_LGNon_ids
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    PlotTuningCurve(
       dsv,
       ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'contrast', 
            'neurons': list(analog_Xon_ids[0:1]), 
            'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="ContrastSensitivity_LGN_On.png"
    ).plot({
       '*.y_lim':(0,100), 
       # '*.x_scale':'log', '*.x_scale_base':10,
       '*.fontsize':17
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'contrast', 
            'neurons': list(analog_Xoff_ids[0:1]), 
            'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="ContrastSensitivity_LGN_Off.png"
    ).plot({
       '*.y_lim':(0,100), 
       # '*.x_scale':'log', '*.x_scale_base':10,
       '*.fontsize':17
    })
    if analog_PGN_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'contrast', 
                'neurons': list(analog_PGN_ids[0:1]), 
                'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="ContrastSensitivity_PGN.png"
        ).plot({
           '*.y_lim':(0,100), 
           # '*.x_scale':'log', '*.x_scale_base':10,
           '*.fontsize':17
        })
    if analog_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'contrast', 
                'neurons': list(analog_ids[0:1]), 
                'sheet_name' : 'V1_Exc_L4'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="ContrastSensitivity_V1e.png"
        ).plot({
           '*.y_lim':(0,100), 
           # '*.x_scale':'log', '*.x_scale_base':10,
           '*.fontsize':17
        })



def plot_spatial_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against spatial frequencies
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    # dsv = param_filter_query( data_store, st_name='FullfieldDriftingSquareGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    # dsv.print_content(full_ADS=True)
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'spatial_frequency', 
           'neurons': list(analog_Xon_ids[0:1]), 
           'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="SpatialFrequencyTuning_LGN_On.png"
    ).plot({
       '*.y_lim':(0,100), 
       #'*.x_scale':'log', '*.x_scale_base':2,
       '*.fontsize':17
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'spatial_frequency', 
           'neurons': list(analog_Xoff_ids[0:1]), 
           'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="SpatialFrequencyTuning_LGN_Off.png"
    ).plot({
       '*.y_lim':(0,100), 
       #'*.x_scale':'log', '*.x_scale_base':2,
       '*.fontsize':17
    })
    if analog_PGN_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
               'polar': False,
               'pool': False,
               'centered': False,
               'mean': False,
               'parameter_name' : 'spatial_frequency', 
               'neurons': list(analog_PGN_ids[0:1]), 
               'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_PGN.png"
        ).plot({
           # '*.y_lim':(0,100), 
           #'*.x_scale':'log', '*.x_scale_base':2,
           '*.fontsize':17
        })
    if analog_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
               'polar': False,
               'pool': False,
               'centered': False,
               'mean': False,
               'parameter_name' : 'spatial_frequency', 
               'neurons': list(analog_ids[0:1]), 
               'sheet_name' : 'V1_Exc_L4'
           }), 
           fig_param={'dpi' : 50,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_V1e.png"
        ).plot({
           '*.y_lim':(0,100), 
           '*.x_scale':'log', '*.x_scale_base':2,
           '*.fontsize':17
        })



def plot_temporal_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xon_ids[0:1]), 
           'sheet_name' : 'X_ON'
      }), 
      fig_param={'dpi' : 100,'figsize': (8,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_On.png"
    ).plot({
        '*.y_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':2,
        '*.fontsize':27
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xoff_ids[0:1]), 
           'sheet_name' : 'X_OFF'
      }), 
      fig_param={'dpi' : 100,'figsize': (8,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_Off.png"
    ).plot({
        '*.y_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':2,
        '*.fontsize':27
    })
    if analog_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
               'polar': False,
               'pool': False,
               'centered': False,
               'mean': False,
               'parameter_name' : 'temporal_frequency', 
               'neurons': list(analog_ids), 
               'sheet_name' : 'V1_Exc_L4'
          }), 
          fig_param={'dpi' : 100,'figsize': (30,8)}, 
          plot_file_name="TemporalFrequencyTuning_V1e.png"
        ).plot({
            '*.y_lim':(0,60), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':27
        })



def plot_size_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against sizes
    dsv = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', analysis_algorithm=['TrialAveragedFiringRate'] )
    PlotTuningCurve(
        dsv,
        ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'radius', 
            'neurons': list(analog_Xon_ids[0:1]), 
            'sheet_name' : 'X_ON'
        }), 
        fig_param={'dpi' : 100,'figsize': (8,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_On.png"
    ).plot({
        '*.y_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':2,
        '*.fontsize':17
    })
    PlotTuningCurve(
        dsv,
        ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'radius', 
            'neurons': list(analog_Xoff_ids[0:1]), 
            'sheet_name' : 'X_OFF'
        }), 
        fig_param={'dpi' : 100,'figsize': (8,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_Off.png"
    ).plot({
        '*.y_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':2,
        '*.fontsize':17
    })
    if analog_PGN_ids:
        PlotTuningCurve(
            dsv,
            ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'radius', 
                'neurons': list(analog_PGN_ids[0:1]), 
                'sheet_name' : 'PGN'
            }), 
            fig_param={'dpi' : 100,'figsize': (8,8)}, 
            plot_file_name="SizeTuning_Grating_PGN.png"
        ).plot({
            '*.y_lim':(0,200), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':17
        })
    if analog_ids:
        PlotTuningCurve(
            dsv,
            ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'radius', 
                'neurons': list(analog_ids[0:1]), 
                'sheet_name' : 'V1_Exc_L4'
            }), 
            fig_param={'dpi' : 100,'figsize': (8,8)}, 
            plot_file_name="SizeTuning_Grating_l4_exc.png"
        ).plot({
            '*.y_lim':(0,100), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':17
        })
    # dsv = param_filter_query( data_store, st_name='FlatDisk', analysis_algorithm=['TrialAveragedFiringRate'] )
    # PlotTuningCurve(
    #    dsv,
    #    ParameterSet({
    #         'polar': False,
    #         'pool': False,
    #         'centered': False,
    #         'mean': False,
    #         'parameter_name' : 'radius', 
    #         'neurons': list(analog_Xon_ids[0:1]), 
    #         'sheet_name' : 'X_ON'
    #    }), 
    #     fig_param={'dpi' : 100,'figsize': (8,8)}, 
    #    plot_file_name="SizeTuning_Disk_LGN_On.png"
    # ).plot({
    #    #'*.y_lim':(0,50), 
    #    '*.x_scale':'log', '*.x_scale_base':2,
    #    '*.fontsize':17
    # })
    # PlotTuningCurve(
    #    dsv,
    #    ParameterSet({
    #         'polar': False,
    #         'pool': False,
    #         'centered': False,
    #         'mean': False,
    #         'parameter_name' : 'radius', 
    #         'neurons': list(analog_Xoff_ids[0:1]), 
    #         'sheet_name' : 'X_OFF'
    #    }), 
    #     fig_param={'dpi' : 100,'figsize': (8,8)}, 
    #    plot_file_name="SizeTuning_Disk_LGN_Off.png"
    # ).plot({
    #    #'*.y_lim':(0,50), 
    #    '*.x_scale':'log', '*.x_scale_base':2,
    #    '*.fontsize':17
    # })
    # PlotTuningCurve(
    #    dsv,
    #    ParameterSet({
    #         'polar': False,
    #         'pool': False,
    #         'centered': False,
    #         'mean': False,
    #         'parameter_name' : 'radius', 
    #         'neurons': list(analog_ids), 
    #         'sheet_name' : 'V1_Exc_L4'
    #    }), 
    #     fig_param={'dpi' : 100,'figsize': (8,8)}, 
    #    plot_file_name="SizeTuning_Disk_l4_exc.png"
    # ).plot({
    #    '*.y_lim':(0,100), 
    #    '*.x_scale':'log', '*.x_scale_base':2,
    #    '*.fontsize':17
    # })



#--------------------
# LIFELONG SPARSENESS
# # per neuron FanoFactor level
# dsv = param_filter_query(data_store, analysis_algorithm=['Analog_MeanSTDAndFanoFactor'], sheet_name=['X_ON'], value_name='FanoFactor(VM)')   
# PerNeuronValuePlot(
#    dsv,
#    ParameterSet({'cortical_view':True}),
#    fig_param={'dpi' : 100,'figsize': (6,6)}, 
#    plot_file_name="FanoFactor_LGN_On.png"
# ).plot({
#    '*.x_axis' : None, 
#    '*.fontsize':17
# })
# # # per neuron Activity Ratio
# dsv = param_filter_query(data_store, analysis_algorithm=['TrialAveragedSparseness'], sheet_name=['X_ON'], value_name='Sparseness')   
# PerNeuronValuePlot(
#     dsv,
#     ParameterSet({'cortical_view':True}),
#     fig_param={'dpi' : 100,'figsize': (6,6)}, 
#     plot_file_name="Sparseness_LGN_On.png"
# ).plot({
#     '*.x_axis' : None, 
#     '*.fontsize':17
# })

# #-------------------
# # ORIENTATION TUNING
# dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
# PlotTuningCurve( 
#   dsv, 
#   ParameterSet({
#         'polar': False,
#         'pool': False,
#         'centered': False,
#         'mean': False,
#         'parameter_name':'orientation', 
#         'neurons':list(analog_Xon_ids), 
#         'sheet_name':'X_ON'
#   }), 
#   fig_param={'dpi' : 100,'figsize': (30,8)}, 
#   plot_file_name="OrientationTuning_LGN_On.png"
# ).plot({
#     '*.y_lim' : (0,100),
#     '*.fontsize':17
# })
# PlotTuningCurve( 
#   dsv, 
#   ParameterSet({
#         'polar': False,
#         'pool': False,
#         'centered': False,
#         'mean': False,
#         'parameter_name':'orientation', 
#         'neurons':list(analog_Xoff_ids), 
#         'sheet_name':'X_OFF'
#   }), 
#   fig_param={'dpi' : 100,'figsize': (30,8)}, 
#   plot_file_name="OrientationTuning_LGN_Off.png"
# ).plot({
#     '*.y_lim' : (0,100),
#     '*.fontsize':17
# })

# # V1        
# dsv = param_filter_query(data_store,st_name=['InternalStimulus'])        
# OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[0], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (28,12)},plot_file_name='SSExcAnalog.png').plot()
# OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Inh_L4', 'neuron' : analog_ids_inh[0], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (28,12)},plot_file_name='SSInhAnalog.png').plot()    

# if False:            
#     TrialToTrialVariabilityComparison(data_store,ParameterSet({}),plot_file_name='TtTVar.png').plot()

#     dsv = param_filter_query(data_store,st_name='NaturalImageWithEyeMovement')            
#     KremkowOverviewFigure(dsv,ParameterSet({'neuron' : l4_exc,'sheet_name' : 'V1_Exc_L4'}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name='NMOverview.png').plot()            

# if True:
#     dsv = param_filter_query(data_store,st_name='FullfieldDriftingSinusoidalGrating',st_contrast=100)    
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : l4_exc, 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc2.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[0], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc3.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[1], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc4.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[2], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc5.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[3], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc6.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[4], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc7.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[5], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc8.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[6], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc9.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_ids[7], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name="Exc10.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})

    
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Inh_L4', 'neuron' : l4_inh, 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (14,12)},plot_file_name="Inh.png").plot({'Vm_plot.y_lim' : (-67,-56),'Conductance_plot.y_lim' : (0,35.0)})
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'X_ON', 'neuron' : sorted(param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_esyn_ids())[0], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (14,12)},plot_file_name="LGN0On.png").plot()
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'X_OFF', 'neuron' : sorted(param_filter_query(data_store,sheet_name="X_OFF").get_segments()[0].get_stored_esyn_ids())[0], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (14,12)},plot_file_name="LGN0Off.png").plot()
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'X_ON', 'neuron' : sorted(param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_esyn_ids())[1], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (14,12)},plot_file_name="LGN1On.png").plot()
#     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'X_OFF', 'neuron' : sorted(param_filter_query(data_store,sheet_name="X_OFF").get_segments()[0].get_stored_esyn_ids())[1], 'sheet_activity' : {}}),fig_param={'dpi' : 100,'figsize': (14,12)},plot_file_name="LGN1Off.png").plot()

# # tuning curves
# dsv = param_filter_query(data_store,st_name='FullfieldDriftingSinusoidalGrating',analysis_algorithm=['TrialAveragedFiringRate','Analog_F0andF1'])    
# PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'orientation', 'neurons': list(analog_ids), 'sheet_name' : 'V1_Exc_L4','centered'  : True,'mean' : False,'polar' : False, 'pool' : False}),fig_param={'dpi' : 100,'figsize': (25,12)},plot_file_name="TCExc.png").plot({'TuningCurve F0_Inh_Cond.y_lim' : (0,180) , 'TuningCurve F0_Exc_Cond.y_lim' : (0,80)})
# PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'orientation', 'neurons': list(analog_ids_inh), 'sheet_name' : 'V1_Inh_L4','centered'  : True,'mean' : False,'polar' : False, 'pool' : False}),fig_param={'dpi' : 100,'figsize': (25,12)},plot_file_name="TCInh.png").plot({'TuningCurve F0_Inh_Cond.y_lim' : (0,180) , 'TuningCurve F0_Exc_Cond.y_lim' : (0,80)})

# dsv = param_filter_query(data_store,st_name='FullfieldDriftingSinusoidalGrating',analysis_algorithm=['Analog_MeanSTDAndFanoFactor'])    
# PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'orientation', 'neurons': list(analog_ids), 'sheet_name' : 'V1_Exc_L4','centered'  : True,'mean' : False,'polar' : False, 'pool' : False}),fig_param={'dpi' : 100,'figsize': (25,12)},plot_file_name="TCExcA.png").plot()
# PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'orientation', 'neurons': list(analog_ids_inh), 'sheet_name' : 'V1_Inh_L4','centered'  : True,'mean' : False,'polar' : False, 'pool' : False}),fig_param={'dpi' : 100,'figsize': (25,12)},plot_file_name="TCInhA.png").plot()

#-----------
# ## CONTOUR COMPLETION
## Square Grating
# dsv = param_filter_query( data_store, st_name='FullfieldDriftingSquareGrating', analysis_algorithm=['TrialAveragedCorrectedCrossCorrelation'] )
# PerNeuronPairAnalogSignalListPlot(
#     dsv,
#     ParameterSet({
#         'sheet_name': 'X_ON'
#     }),
#     fig_param={'dpi' : 100,'figsize': (14,14)}, 
#     plot_file_name='SquareGrating_XCorr_LGN_On.png'
# ).plot({
#     '*.y_lim':(-30,30), 
#     '*.fontsize':17
# })
# PerNeuronPairAnalogSignalListPlot(
#     dsv,
#     ParameterSet({
#         'sheet_name' : 'V1_Exc_L4', 
#     }),
#     fig_param={'dpi' : 100,'figsize': (14,14)}, 
#     plot_file_name='SquareGrating_XCorr_V1e.png'
# ).plot({
#     '*.y_lim':(-30,30), 
#     '*.fontsize':17
# })
# Flashing squares
# dsv = param_filter_query( data_store, st_name='FlashingSquares', analysis_algorithm=['TrialAveragedCorrectedCrossCorrelation'] )
# PerNeuronPairAnalogSignalListPlot(
#     dsv,
#     ParameterSet({
#         'sheet_name': 'X_ON'
#     }),
#     fig_param={'dpi' : 100,'figsize': (14,14)}, 
#     plot_file_name='FlashingSquare_XCorr_LGN_On.png'
# ).plot({
#     '*.y_lim':(-30,30), 
#     '*.fontsize':17
# })
# PerNeuronPairAnalogSignalListPlot(
#     dsv,
#     ParameterSet({
#         'sheet_name' : 'V1_Exc_L4', 
#     }),
#     fig_param={'dpi' : 100,'figsize': (14,14)}, 
#     plot_file_name='FlashingSquare_XCorr_V1e.png'
# ).plot({
#     '*.y_lim':(-30,30), 
#     '*.fontsize':17
# })






#########################################################################################
# ---- OVERVIEW ----
def plot_overview( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # RETINA
    OverviewPlot(
       data_store,
       ParameterSet({
           # 'centered': False,
           # 'mean': False,
           'spontaneous': False,
           'sheet_name' : 'X_OFF', 
           'neuron' : analog_Xoff_ids[0], 
           'sheet_activity' : {}
       }),
       fig_param={'dpi' : 100,'figsize': (19,12)},
       plot_file_name="LGN_Off.png"
    ).plot({
        'Vm_plot.*.y_lim' : (-100,-40),
        '*.fontsize':7
    })

    OverviewPlot(
       data_store,
       ParameterSet({
           # 'centered': False,
           # 'mean': False,
           'spontaneous': False,
           'sheet_name' : 'X_ON', 
           'neuron' : analog_Xon_ids[0], 
           'sheet_activity' : {}
       }),
       fig_param={'dpi':100, 'figsize':(19,12)},
       plot_file_name="LGN_On.png"
    ).plot({
        'Vm_plot.*.y_lim' : (-100,-40),
        '*.fontsize':7
    })

    if analog_PGN_ids:
        OverviewPlot(
           data_store,
           ParameterSet({
               'spontaneous': False,
               'sheet_name' : 'PGN', 
               'neuron' : analog_PGN_ids[0], 
               'sheet_activity' : {}
           }),
           fig_param={'dpi' : 100,'figsize': (19,12)},
           plot_file_name="PGN.png"
        ).plot({
            'Vm_plot.*.y_lim' : (-100,-40),
            '*.fontsize':7
        })

    if analog_ids:
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[0], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_0.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[1], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_1.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[2], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_2.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[3], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_3.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[4], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_4.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[5], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_5.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[6], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_6.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[7], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_7.png").plot({'Vm_plot.*.y_lim':(-67,-56), 'Conductance_plot.y_lim':(0,35.0), '*.fontsize':7})

