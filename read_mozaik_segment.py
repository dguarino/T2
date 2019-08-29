import numpy as np
import pickle
import neo
import glob

# loading a neo file
folder = "ThalamoCorticalModel_data_interrupted_bar_ver2_____/" # example on my machine
# folder = 	"/media/do/HANGAR/Deliverable/ThalamoCorticalModel_data_orientation_closed_____/" # example on my machine

filenames = [ x for x in glob.glob(folder+"*.pickle") ]
# print filenames

for name in filenames:
	print name

	seg = pickle.load( open(name, "rb") )

	if 'V1_Exc_L4' in seg.description or 'V1_Inh_L4' in seg.description:
		print
		print seg.description

		print
		print len(seg.spiketrains)
		for st in seg.spiketrains:
			print st.annotations

		print
		for a in seg.analogsignalarrays:
			print
			print a.name, a.annotations
			# print a

	# vm = seg.filter(name = 'v')[0]