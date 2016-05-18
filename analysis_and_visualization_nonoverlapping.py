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


logger = mozaik.getMozaikLogger()



def perform_analysis_and_visualization( data_store, atype='contrast', withPGN=False, withV1=False ):
    # Gather indexes
    spike_Xon_ids = param_filter_query(data_store,sheet_name="X_ON").get_segments()[0].get_stored_spike_train_ids()
    spike_Xoff_ids = param_filter_query(data_store,sheet_name="X_OFF").get_segments()[0].get_stored_spike_train_ids()
    print "spike_Xon_ids: ",spike_Xon_ids
    print "spike_Xoff_ids: ",spike_Xoff_ids

    # NON-OVERLAPPING indexes
    # we want to have only the spike_Xon_ids that lie outside of the excitatory zone overlapping with the cortical one
    position_Xon = data_store.get_neuron_postions()['X_ON']
    position_Xoff = data_store.get_neuron_postions()['X_OFF']
    Xon_sheet_ids = data_store.get_sheet_indexes(sheet_name='X_ON',neuron_ids=spike_Xon_ids)
    Xoff_sheet_ids = data_store.get_sheet_indexes(sheet_name='X_OFF',neuron_ids=spike_Xoff_ids)
    nonoverlap_Xon_ids = []
    nonoverlap_Xoff_ids = []
    for i in Xon_sheet_ids:
        a = numpy.array((position_Xon[0][i],position_Xon[1][i],position_Xon[2][i]))
        l = numpy.linalg.norm(a - 0.0)
        if l>0.75 and l<2.:
            nonoverlap_Xon_ids.append(i[0])
            # print a
    for i in Xoff_sheet_ids:
        a = numpy.array((position_Xoff[0][i],position_Xoff[1][i],position_Xoff[2][i]))
        l = numpy.linalg.norm(a - 0.0)
        if l>0.75 and l<2.:
            nonoverlap_Xoff_ids.append(i[0])
    nonoverlap_Xon_ids = data_store.get_sheet_ids(sheet_name='X_ON',indexes=nonoverlap_Xon_ids)
    nonoverlap_Xoff_ids = data_store.get_sheet_ids(sheet_name='X_OFF',indexes=nonoverlap_Xoff_ids)
    print "nonoverlap_Xon_ids: ",nonoverlap_Xon_ids
    print "nonoverlap_Xoff_ids: ",nonoverlap_Xoff_ids
