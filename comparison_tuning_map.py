import numpy
import scipy.stats
import pylab
import matplotlib.pyplot as plt
import glob
import os


directory = "CombinationParamSearch_focused_nonoverlapping"
xvalues = [40, 70, 90, 110, 140]
yvalues = [100, 130, 150, 170, 200]
ticks = [0,1,2,3,4]


# directory = "CombinationParamSearch_large_nonoverlapping"
# xvalues = [30, 50, 70, 90]
# yvalues = [150, 200, 250, 300]
# ticks = [0,1,2,3]



filenames = [ x for x in glob.glob(directory+"/*.csv") ]
print filenames


def normalize(a, axis=-1, order=2):
	l2 = numpy.atleast_1d( numpy.linalg.norm(a, order, axis) )
	l2[l2==0] = 1
	return a/ numpy.expand_dims(l2, axis)


colors = numpy.zeros( (len(xvalues),len(yvalues)) )
alpha = numpy.zeros( (len(xvalues),len(yvalues)) )
for name in filenames:
	print name
	mapname = os.path.splitext(name)[0]+'.png'
	print mapname

	# cycle over lines
	with open(name,'r') as csv:
		for i,line in enumerate(csv): 
			print line
			print eval(line)
			xvalue = eval(line)[0]
			yvalue = eval(line)[1]
			s = eval(line)[2]
			print xvalue, yvalue, s
			fit = numpy.polyfit([0,1,2], s, 1)
			if numpy.amin(s) < -10.: # tolerance on the smallest value
				fit = [0., 0.]
			if fit[0] < 0.:
				fit = [0., 0.]
			print s, fit
			colors[xvalues.index(xvalue)][yvalues.index(yvalue)] = fit[0] # if fit[0]>0. else 0. # slope
			alpha[xvalues.index(xvalue)][yvalues.index(yvalue)] = fit[1] # in

	print colors
	# alpha = numpy.absolute( normalize(alpha) )
	# alpha = normalize(alpha)
	print alpha

	plt.figure()
	ca = plt.imshow(colors, interpolation='nearest', cmap='coolwarm')
	# ca = plt.contourf(colors, cmap='coolwarm')
	cbara = plt.colorbar(ca, ticks=[numpy.amin(colors), 0, numpy.amax(colors)])
	cbara.set_label('Regression Slope')
	# cb = plt.contour(alpha, cmap='brg')
	# cbarb = plt.colorbar(cb, ticks=[numpy.amin(alpha), 0, numpy.amax(alpha)])
	# print cbarb.set_ticklabels([numpy.amin(alpha), 0, numpy.amax(alpha)])
	# cbarb.set_label('Regression Intercept')
	plt.xticks(ticks, xvalues)
	plt.yticks(ticks, yvalues)
	plt.xlabel('V1-PGN arborization radius')
	plt.ylabel('PGN-LGN arborization radius')
	plt.savefig( mapname, dpi=300 )
	plt.close()
	# plt.show()