import numpy as np
import pickle
import neo
import glob

# loading a neo file
folder = 	"ThalamoCorticalModel_interrupted_bar_ver3_closed_____/" # example on my machine
# folder = 	"/media/do/HANGAR/Deliverable/ThalamoCorticalModel_data_orientation_closed_____/" # example on my machine

# ds = pickle.load( open(folder+"datastore.recordings.pickle", "rb") )
# print ds
# print ds.name, ds.description, ds.segments

filenames = [ x for x in glob.glob(folder+"*.pickle") ]
# print filenames

for name in filenames:
	print name

	seg = pickle.load( open(name, "rb") )

	if 'V1_Exc_L4' in seg.description or 'V1_Inh_L4' in seg.description:
		# print
		# print seg.name, seg.description

		# print
		# print len(seg.spiketrains)
		# for st in seg.spiketrains:
		# 	print st.name, st.description, st.annotations

		for a in seg.analogsignalarrays:
			if a.name == 'v':
				print
				print a.description, a.annotations
				# print a

	# vm = seg.filter(name = 'v')[0]