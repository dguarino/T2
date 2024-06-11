import numpy as np
import pickle
import neo
import glob

# loading a neo file
folder = "Deliverable/ThalamoCorticalModel_data_contrast_closed_____/" # example on my machine

filenames = [ x for x in glob.glob(folder+"*.pickle") ]
print(filenames)

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