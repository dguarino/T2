import numpy
import mozaik
import pylab
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.TrialAveragedFiringRateCutout import TrialAveragedFiringRateCutout
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



def perform_analysis_and_visualization( data_store, atype='contrast', withPGN=False, withV1=False ):
    # Gather indexes
    # analog_RGC_Xon_ids = sorted( param_filter_query(data_store,sheet_name="Retina_X_ON").get_segments()[0].get_stored_vm_ids() )
    # analog_RGC_Xoff_ids = sorted( param_filter_query(data_store,sheet_name="Retina_X_OFF").get_segments()[0].get_stored_vm_ids() )
    analog_Xon_ids = sorted( param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_vm_ids() )
    analog_Xoff_ids = sorted( param_filter_query(data_store,sheet_name="X_OFF").get_segments()[0].get_stored_vm_ids() )
    # print "analog_RGC_Xon_ids: ",analog_RGC_Xon_ids
    # print "analog_RGC_Xoff_ids: ",analog_RGC_Xoff_ids
    print "analog_Xon_ids: ",analog_Xon_ids
    print "analog_Xoff_ids: ",analog_Xoff_ids

    analog_PGN_ids = None
    if withPGN:
        analog_PGN_ids = sorted( param_filter_query(data_store,sheet_name="PGN").get_segments()[0].get_stored_vm_ids() )
        print "analog_PGN_ids: ",analog_PGN_ids

    analog_ids = None
    spike_ids = None
    if withV1:
        analog_ids = sorted( param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_vm_ids() )
        spike_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_spike_train_ids()
        analog_ids_inh = param_filter_query(data_store,sheet_name="V1_Inh_L4").get_segments()[0].get_stored_esyn_ids()
        spike_ids_inh = param_filter_query(data_store,sheet_name="V1_Inh_L4").get_segments()[0].get_stored_spike_train_ids()
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
    if atype == 'luminance':
        perform_analysis_luminance( data_store, withPGN, withV1 )

    if atype in ['contrast', 'spatial_frequency', 'temporal_frequency', 'orientation']:
        perform_analysis_full_field( data_store, withPGN, withV1 )

    if atype == 'size':
        perform_analysis_size( data_store, withPGN, withV1 )

    if atype == 'subcortical_conn':
        perform_analysis_and_visualization_subcortical_conn(data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids)

    if atype == 'cortical_or_conn':
        perform_analysis_and_visualization_cortical_or_conn( data_store )

    ##############
    # PLOTTING
    activity_plot_param =    {
           'frame_rate' : 5,  
           'bin_width' : 5.0, 
           'scatter' :  True,
           'resolution' : 0
    }

    # Overview
    plot_overview( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids )

    # Tuning
    if atype == 'luminance':
        plot_luminance_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)

    if atype == 'contrast':
        plot_contrast_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)

    if atype == 'size':
        plot_size_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids )

    if atype == 'spatial_frequency':
        plot_spatial_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)

    if atype == 'temporal_frequency':
        plot_temporal_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)

    if atype == 'orientation':
        plot_orientation_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids, analog_ids)


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
    # TrialAveragedFiringRate( dsv0_Xon, ParameterSet({}) ).analyse()
    TrialAveragedFiringRateCutout( dsv0_Xon, ParameterSet({}) ).analyse(start=100, end=1000)
    dsv0_Xoff = param_filter_query( data_store, st_name='Null', sheet_name='X_OFF' )  
    # TrialAveragedFiringRate( dsv0_Xoff, ParameterSet({}) ).analyse()
    TrialAveragedFiringRateCutout( dsv0_Xoff, ParameterSet({}) ).analyse(start=100, end=1000)

    if withPGN:
        dsv0_PGN = param_filter_query( data_store, st_name='Null', sheet_name='PGN' )  
        # TrialAveragedFiringRate( dsv0_PGN, ParameterSet({}) ).analyse()
        TrialAveragedFiringRateCutout( dsv0_PGN, ParameterSet({}) ).analyse(start=100, end=1000)
    if withV1:
        dsv0_V1e = param_filter_query( data_store, st_name='Null', sheet_name='V1_Exc_L4' )  
        TrialAveragedFiringRate( dsv0_V1e, ParameterSet({}) ).analyse()



def perform_analysis_full_field( data_store, withPGN=False, withV1=False ):
    dsv10 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='X_ON' )  
    # TrialAveragedFiringRate( dsv10, ParameterSet({}) ).analyse()
    TrialAveragedFiringRateCutout( dsv10, ParameterSet({}) ).analyse(start=100, end=1000)
    dsv11 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='X_OFF' )  
    # TrialAveragedFiringRate( dsv11, ParameterSet({}) ).analyse()
    TrialAveragedFiringRateCutout( dsv11, ParameterSet({}) ).analyse(start=100, end=1000)
    if withPGN:
        dsv12 = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='PGN' )  
        # TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()
        TrialAveragedFiringRateCutout( dsv12, ParameterSet({}) ).analyse(start=100, end=1000)
    if withV1:
        dsv1_V1e = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='V1_Exc_L4' )  
        TrialAveragedFiringRate( dsv1_V1e, ParameterSet({}) ).analyse()
        # dsv1_V1i = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name='V1_Inh_L4' )  
        # TrialAveragedFiringRate( dsv1_V1i, ParameterSet({}) ).analyse()
        NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()



def perform_analysis_size( data_store, withPGN=False, withV1=False ):
    dsv10 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='X_ON' )  
    TrialAveragedFiringRate( dsv10, ParameterSet({}) ).analyse()
    dsv11 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='X_OFF' )  
    TrialAveragedFiringRate( dsv11, ParameterSet({}) ).analyse()
    dsv12 = param_filter_query( data_store, st_name='FlatDisk', sheet_name='X_ON' )  
    TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()

    if withPGN:
        dsv12 = param_filter_query( data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name='PGN' )  
        TrialAveragedFiringRate( dsv12, ParameterSet({}) ).analyse()

    if withV1:
        analog_ids = sorted( param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_vm_ids() )
        spike_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_spike_train_ids()
        analog_ids_inh = param_filter_query(data_store,sheet_name="V1_Inh_L4").get_segments()[0].get_stored_esyn_ids()
        spike_ids_inh = param_filter_query(data_store,sheet_name="V1_Inh_L4").get_segments()[0].get_stored_spike_train_ids()

        # NeuronAnnotationsToPerNeuronValues(data_store,ParameterSet({})).analyse()
        # l4_exc_or = data_store.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = 'V1_Exc_L4')[0]

        # l4_exc_or_many = numpy.array(spike_ids)[numpy.nonzero(numpy.array([circular_dist(l4_exc_or.get_value_by_id(i),0,numpy.pi)  for i in spike_ids]) < 0.1)[0]]

        # idx4 = data_store.get_sheet_indexes(sheet_name='V1_Exc_L4',neuron_ids=l4_exc_or_many)

        # x = data_store.get_neuron_postions()['V1_Exc_L4'][0][idx4]
        # y = data_store.get_neuron_postions()['V1_Exc_L4'][1][idx4]
        # center4 = l4_exc_or_many[numpy.nonzero(numpy.sqrt(numpy.multiply(x,x)+numpy.multiply(y,y)) < 0.1)[0]]
        
        # analog_center4 = set(center4).intersection(analog_ids)

        TrialAveragedFiringRate(param_filter_query(data_store,sheet_name=['V1_Exc_L4','V1_Inh_L4'],st_name='DriftingSinusoidalGratingDisk'),ParameterSet({})).analyse()
            
        # print len(l4_exc_or_many)

        # dsv = param_filter_query(data_store,st_name='DriftingSinusoidalGratingDisk',analysis_algorithm=['TrialAveragedFiringRate'])    
        # PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'radius', 'neurons': list(center4), 'sheet_name' : 'V1_Exc_L4','centered'  : False,'mean' : False, 'polar' : False, 'pool'  : False}),plot_file_name='SizeTuningExcL4.png',fig_param={'dpi' : 100,'figsize': (32,7)}).plot()
        # PlotTuningCurve(dsv,ParameterSet({'parameter_name' : 'radius', 'neurons': list(center4), 'sheet_name' : 'V1_Exc_L4','centered'  : False,'mean' : True, 'polar' : False, 'pool'  : False}),plot_file_name='SizeTuningExcL4M.png',fig_param={'dpi' : 100,'figsize': (32,7)}).plot()
        # data_store.save()
        
        # if True:
        #     dsv = param_filter_query(data_store,st_name=['DriftingSinusoidalGratingDisk'])    
        #     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_center4[0], 'sheet_activity' : {}, 'spontaneous' : True}),fig_param={'dpi' : 100,'figsize': (28,12)},plot_file_name='Overview_ExcL4_1.png').plot()
        #     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_center4[1], 'sheet_activity' : {}, 'spontaneous' : True}),fig_param={'dpi' : 100,'figsize': (28,12)},plot_file_name='Overview_ExcL4_2.png').plot()
        #     OverviewPlot(dsv,ParameterSet({'sheet_name' : 'V1_Exc_L4', 'neuron' : analog_center4[2], 'sheet_activity' : {}, 'spontaneous' : True}),fig_param={'dpi' : 100,'figsize': (28,12)},plot_file_name='Overview_ExcL4_3.png').plot()



#########################################################################################
#########################################################################################
#########################################################################################

def plot_luminance_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against luminance levels              
    # dsv = param_filter_query( data_store, st_name='Null', analysis_algorithm=['TrialAveragedFiringRate'] )
    dsv = param_filter_query( data_store, st_name='Null', analysis_algorithm=['TrialAveragedFiringRateCutout'] )
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
       '*.y_lim':(0,60), 
       '*.x_lim':(0,100), 
       '*.x_scale':'log', '*.x_scale_base':10,
       '*.fontsize':17,
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': True,
            'parameter_name' : 'background_luminance', 
            'neurons': list(analog_Xon_ids), 
            'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="FlatLuminanceSensitivity_LGN_On_mean.png"
    ).plot({
       '*.y_lim':(0,60), 
       '*.x_lim':(0,100), 
       '*.x_scale':'log', '*.x_scale_base':10,
       '*.fontsize':17,
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
        '*.y_lim':(0,60), 
        '*.x_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':10,
        '*.fontsize':17,
    })
    PlotTuningCurve(
        dsv,
        ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': True,
            'parameter_name' : 'background_luminance', 
            'neurons': list(analog_Xoff_ids), 
            'sheet_name' : 'X_OFF'
        }), 
        fig_param={'dpi' : 100,'figsize': (8,8)}, 
        plot_file_name="FlatLuminanceSensitivity_LGN_Off_mean.png"
    ).plot({
        '*.y_lim':(0,60), 
        '*.x_lim':(0,100), 
        '*.x_scale':'log', '*.x_scale_base':10,
        '*.fontsize':17,
    })
    if analog_PGN_ids:
        PlotTuningCurve(
           dsv,
           ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': False,
                'parameter_name' : 'background_luminance', 
                'neurons': list(analog_PGN_ids[0:1]), 
                'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="FlatLuminanceSensitivity_PGN.png"
        ).plot({
           #'*.y_lim':(0,60), 
           # '*.x_scale':'log', '*.x_scale_base':10,
           '*.fontsize':17
        })
        PlotTuningCurve(
           dsv,
           ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': True,
                'parameter_name' : 'background_luminance', 
                'neurons': list(analog_PGN_ids), 
                'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="FlatLuminanceSensitivity_PGN_mean.png"
        ).plot({
           #'*.y_lim':(0,60), 
           # '*.x_scale':'log', '*.x_scale_base':10,
           '*.fontsize':17
        })



def plot_contrast_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # CONTRAST SENSITIVITY analog_LGNon_ids
    # dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRateCutout'] )
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
           '*.y_lim':(0,60), 
           # '*.x_scale':'log', '*.x_scale_base':10,
           '*.fontsize':17
        })



def plot_spatial_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against spatial frequencies
    # dsv = param_filter_query( data_store, st_name='FullfieldDriftingSquareGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRateCutout'] )
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
           'parameter_name' : 'spatial_frequency', 
           'neurons': list(analog_Xoff_ids[0:1]), 
           'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="SpatialFrequencyTuning_LGN_Off.png"
    ).plot({
       '*.y_lim':(0,100), 
       '*.x_scale':'log', '*.x_scale_base':2,
       '*.fontsize':17
    })
    # PlotTuningCurve(
    #    dsv,
    #    ParameterSet({
    #        'polar': False,
    #        'pool': False,
    #        'centered': False,
    #        'mean': False,
    #        'parameter_name' : 'spatial_frequency', 
    #        'neurons': list(analog_Xon_ids[0:5]), 
    #        'sheet_name' : 'X_ON'
    #    }), 
    #    fig_param={'dpi' : 100,'figsize': (40,8)}, 
    #    plot_file_name="SpatialFrequencyTuning_LGN_On_5.png"
    # ).plot({
    #    '*.y_lim':(0,100), 
    #    '*.x_scale':'log', '*.x_scale_base':2,
    #    '*.fontsize':17
    # })
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': True,
           'parameter_name' : 'spatial_frequency', 
           'neurons': list(analog_Xon_ids), 
           'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="SpatialFrequencyTuning_LGN_On_mean.png"
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
           'mean': True,
           'parameter_name' : 'spatial_frequency', 
           'neurons': list(analog_Xoff_ids), 
           'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (8,8)}, 
       plot_file_name="SpatialFrequencyTuning_LGN_Off_mean.png"
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
               'parameter_name' : 'spatial_frequency', 
               'neurons': list(analog_PGN_ids[0:1]), 
               'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_PGN.png"
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
               'mean': True,
               'parameter_name' : 'spatial_frequency', 
               'neurons': list(analog_PGN_ids), 
               'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_PGN_mean.png"
        ).plot({
           '*.y_lim':(0,100), 
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
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_V1e.png"
        ).plot({
           '*.y_lim':(0,60), 
           '*.x_scale':'log', '*.x_scale_base':2,
           '*.fontsize':17
        })
        PlotTuningCurve(
           dsv,
           ParameterSet({
               'polar': False,
               'pool': False,
               'centered': False,
               'mean': True,
               'parameter_name' : 'spatial_frequency', 
               'neurons': list(analog_ids), 
               'sheet_name' : 'V1_Exc_L4'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="SpatialFrequencyTuning_V1e_mean.png"
        ).plot({
           '*.y_lim':(0,30), 
           '*.x_scale':'log', '*.x_scale_base':2,
           '*.fontsize':17
        })



def plot_temporal_frequency_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRateCutout'] )
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
        '*.fontsize':17
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xon_ids[0:5]), 
           'sheet_name' : 'X_ON'
      }), 
      fig_param={'dpi' : 100,'figsize': (40,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_On_5.png"
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
           'mean': True,
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xon_ids), 
           'sheet_name' : 'X_ON'
      }), 
      fig_param={'dpi' : 100,'figsize': (8,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_On_mean.png"
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
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xoff_ids[0:1]), 
           'sheet_name' : 'X_OFF'
      }), 
      fig_param={'dpi' : 100,'figsize': (8,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_Off.png"
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
           'mean': True,
           'parameter_name' : 'temporal_frequency', 
           'neurons': list(analog_Xoff_ids), 
           'sheet_name' : 'X_OFF'
      }), 
      fig_param={'dpi' : 100,'figsize': (8,8)}, 
      plot_file_name="TemporalFrequencyTuning_LGN_Off_mean.png"
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
               'parameter_name' : 'temporal_frequency', 
               'neurons': list(analog_PGN_ids[0:1]), 
               'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="TemporalFrequencyTuning_PGN.png"
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
               'mean': True,
               'parameter_name' : 'temporal_frequency', 
               'neurons': list(analog_PGN_ids), 
               'sheet_name' : 'PGN'
           }), 
           fig_param={'dpi' : 100,'figsize': (8,8)}, 
           plot_file_name="TemporalFrequencyTuning_PGN_mean.png"
        ).plot({
           '*.y_lim':(0,100), 
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
               'parameter_name' : 'temporal_frequency', 
               'neurons': list(analog_ids[0:1]), 
               'sheet_name' : 'V1_Exc_L4'
          }), 
          fig_param={'dpi' : 100,'figsize': (8,8)}, 
          plot_file_name="TemporalFrequencyTuning_V1e.png"
        ).plot({
            '*.y_lim':(0,60), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':17
        })



def plot_size_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    # firing rate against disk
    # dsv = param_filter_query( data_store, st_name='FlatDisk', analysis_algorithm=['TrialAveragedFiringRate'] )
    # PlotTuningCurve(
    #     dsv,
    #     ParameterSet({
    #         'polar': False,
    #         'pool': False,
    #         'centered': False,
    #         'mean': False,
    #         'parameter_name' : 'radius', 
    #         'neurons': list(analog_Xon_ids[0:1]), 
    #         'sheet_name' : 'X_ON'
    #     }), 
    #     fig_param={'dpi' : 100,'figsize': (8,8)}, 
    #     plot_file_name="SizeTuning_Disk_LGN_On.png"
    # ).plot({
    #     '*.y_lim':(0,150), 
    #     '*.x_scale':'log', '*.x_scale_base':2,
    #     '*.fontsize':17
    # })
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
    PlotTuningCurve(
        dsv,
        ParameterSet({
            'polar': False,
            'pool': False,
            'centered': False,
            'mean': False,
            'parameter_name' : 'radius', 
            'neurons': list(analog_Xon_ids[0:9]), 
            'sheet_name' : 'X_ON'
        }), 
        fig_param={'dpi' : 100,'figsize': (80,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_On_10.png"
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
            'neurons': list(analog_Xoff_ids[0:9]), 
            'sheet_name' : 'X_OFF'
        }), 
        fig_param={'dpi' : 100,'figsize': (80,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_Off_10.png"
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
            'mean': True,
            'parameter_name' : 'radius', 
            'neurons': list(analog_Xon_ids), 
            'sheet_name' : 'X_ON'
        }), 
        fig_param={'dpi' : 100,'figsize': (8,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_On_mean.png"
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
            'mean': True,
            'parameter_name' : 'radius', 
            'neurons': list(analog_Xoff_ids), 
            'sheet_name' : 'X_OFF'
        }), 
        fig_param={'dpi' : 100,'figsize': (8,8)}, 
        plot_file_name="SizeTuning_Grating_LGN_Off_mean.png"
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
            '*.y_lim':(0,100), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':17
        })
        # PlotTuningCurve(
        #     dsv,
        #     ParameterSet({
        #         'polar': False,
        #         'pool': False,
        #         'centered': False,
        #         'mean': False,
        #         'parameter_name' : 'radius', 
        #         'neurons': list(analog_PGN_ids[0:48]), 
        #         'sheet_name' : 'PGN'
        #     }), 
        #     fig_param={'dpi' : 100,'figsize': (384,8)}, 
        #     plot_file_name="SizeTuning_Grating_PGN_48.png"
        # ).plot({
        #     '*.y_lim':(0,100), 
        #     '*.x_scale':'log', '*.x_scale_base':2,
        #     '*.fontsize':17
        # })
        PlotTuningCurve(
            dsv,
            ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': True,
                'parameter_name' : 'radius', 
                'neurons': list(analog_PGN_ids), 
                'sheet_name' : 'PGN'
            }), 
            fig_param={'dpi' : 100,'figsize': (8,8)}, 
            plot_file_name="SizeTuning_Grating_PGN_mean.png"
        ).plot({
            '*.y_lim':(0,100), 
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
            '*.y_lim':(0,60), 
            '*.x_scale':'log', '*.x_scale_base':2,
            '*.fontsize':17
        })
        PlotTuningCurve(
            dsv,
            ParameterSet({
                'polar': False,
                'pool': False,
                'centered': False,
                'mean': True,
                'parameter_name' : 'radius', 
                'neurons': list(analog_ids), 
                'sheet_name' : 'V1_Exc_L4'
            }), 
            fig_param={'dpi' : 100,'figsize': (8,8)}, 
            plot_file_name="SizeTuning_Grating_l4_exc_mean.png"
        ).plot({
            '*.y_lim':(0,60), 
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



#-------------------
# ORIENTATION TUNING

def plot_orientation_tuning( data_store, analog_Xon_ids, analog_Xoff_ids, analog_PGN_ids=None, analog_ids=None ):
    dsv = param_filter_query( data_store, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=['TrialAveragedFiringRate'] )
    dsv.print_content(full_ADS=True)
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': True,
           'parameter_name' : 'orientation', 
           'neurons': list(analog_Xon_ids[:]), 
           'sheet_name' : 'X_ON'
       }), 
       fig_param={'dpi' : 100,'figsize': (240,8)}, 
       plot_file_name="OrientationTuning_LGN_On_mean.png"
    ).plot({
       '*.fontsize':17
    })
    PlotTuningCurve(
       dsv,
       ParameterSet({
           'polar': False,
           'pool': False,
           'centered': False,
           'mean': False,
           'parameter_name' : 'orientation', 
           'neurons': list(analog_Xoff_ids[:]), 
           'sheet_name' : 'X_OFF'
       }), 
       fig_param={'dpi' : 100,'figsize': (240,8)}, 
       plot_file_name="OrientationTuning_LGN_Off_mean.png"
    ).plot({
       '*.fontsize':17
    })
    if analog_PGN_ids:
        # PlotTuningCurve(
        #    dsv,
        #    ParameterSet({
        #        'polar': False,
        #        'pool': False,
        #        'centered': False,
        #        'mean': False,
        #        'parameter_name' : 'orientation', 
        #        'neurons': list(analog_PGN_ids[:]), 
        #        'sheet_name' : 'PGN'
        #    }), 
        #    fig_param={'dpi' : 100,'figsize': (16,8)}, 
        #    plot_file_name="OrientationTuning_PGN_mean.png"
        # ).plot({
        #    '*.fontsize':17
        # })
        # if analog_ids:
        # PlotTuningCurve(
        #    dsv,
        #    ParameterSet({
        #        'polar': False,
        #        'pool': False,
        #        'centered': False,
        #        'mean': False,
        #        'parameter_name' : 'orientation', 
        #        'neurons': list(analog_ids[0:9]), 
        #        'sheet_name' : 'V1_Exc_L4'
        #    }), 
        #    fig_param={'dpi' : 100,'figsize': (80,8)}, 
        #    plot_file_name="OrientationTuning_V1e_1.png"
        # ).plot({
        #    '*.fontsize':17
        # })
        # PlotTuningCurve(
        #    dsv,
        #    ParameterSet({
        #        'polar': False,
        #        'pool': False,
        #        'centered': False,
        #        'mean': False,
        #        'parameter_name' : 'orientation', 
        #        'neurons': list(analog_ids[10:19]), 
        #        'sheet_name' : 'V1_Exc_L4'
        #    }), 
        #    fig_param={'dpi' : 100,'figsize': (80,8)}, 
        #    plot_file_name="OrientationTuning_V1e_2.png"
        # ).plot({
        #    '*.fontsize':17
        # })
        # PlotTuningCurve(
        #    dsv,
        #    ParameterSet({
        #        'polar': False,
        #        'pool': False,
        #        'centered': False,
        #        'mean': False,
        #        'parameter_name' : 'orientation', 
        #        'neurons': list(analog_ids[19:29]), 
        #        'sheet_name' : 'V1_Exc_L4'
        #    }), 
        #    fig_param={'dpi' : 100,'figsize': (80,8)}, 
        #    plot_file_name="OrientationTuning_V1e_3.png"
        # ).plot({
        #    '*.fontsize':17
        # })
        # dsv = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=['V1_Exc_L4'], value_name='LGNAfferentOrientation')   
        dsv = param_filter_query(data_store, sheet_name=['V1_Exc_L4'], value_name='LGNAfferentOrientation')   
        PerNeuronValuePlot(dsv,ParameterSet({"cortical_view":True}), plot_file_name='ORSet.png').plot()
        dsv = param_filter_query(data_store, sheet_name=['V1_Exc_L4'], value_name='Firing rate')   
        PerNeuronValuePlot(dsv,ParameterSet({"cortical_view":True}), plot_file_name='FR.png').plot()



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
        # 'Vm_plot.*.y_lim' : (-100,40),
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
        # 'Vm_plot.*.y_lim' : (-100,40),
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
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[0], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_0.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[1], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_1.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[2], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_2.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[3], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_3.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[4], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_4.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[5], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_5.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[6], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_6.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})
        OverviewPlot( data_store, ParameterSet({'spontaneous':False, 'sheet_name':'V1_Exc_L4', 'neuron':analog_ids[7], 'sheet_activity':{}}), fig_param={'dpi':100,'figsize':(19,12)}, plot_file_name="V1_Exc_L4_7.png").plot({'Vm_plot.*.y_lim':(-80,-50), 'Conductance_plot.y_lim':(0,100.0), '*.fontsize':7})

