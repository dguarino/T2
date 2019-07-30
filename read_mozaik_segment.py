import numpy as np
import pickle
import neo

# loading a neo file
folder = 	"/media/do/HANGAR/Deliverable/ThalamoCorticalModel_data_orientation_closed_____/" # example on my machine
filename = folder + "Segment35.pickle"
seg = pickle.load( open(filename, "rb") )

print
print seg.name

print
print seg.description

print
print len(seg.spiketrains)
for st in seg.spiketrains:
	print st.annotations
	
print
print seg.analogsignals

# vm = seg.filter(name = 'v')[0]