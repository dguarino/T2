import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt



def meshgrid3D(x, y, z):
    """A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    mult_fact = numpy.ones((len(x), len(y), len(z)))
    nax = numpy.newaxis
    return x[:, nax, nax] * mult_fact, \
           y[nax, :, nax] * mult_fact, \
           z[nax, nax, :] * mult_fact


# stt
def stRF_2d(x, y, t, p):
    """
    x, y, and t should all be 3D arrays, produced by meshgrid3D.
    If we need to optimize, it would be quicker to do G() on a 1D t array and
    F_2d() on 2D x and y arrays, and then multiply them, as Jens did in his
    original implementation.
    Timing gives 0.44 s for Jens' implementation, and 2.9 s for this one.
    """

    tmc = G(t, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2)
    tms = G(t-p.td, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2)

    fcm = F_2d(x, y, p.Ac, p.sigma_c)
    fsm = F_2d(x, y, p.As, p.sigma_s)

    # Linear Receptive Field
    #rf = (fcm*tmc - fsm*tms)/(fcm - fsm).max()
    rf = (fcm*tmc - fsm*tms)
    # rf = (fcm - fsm)

    x_res = x[1,0,0] - x[0,0,0]
    fcm_area = fcm[:,:,0].sum()*x_res*x_res
    center_area = 2*numpy.pi*p.sigma_c*p.sigma_c*p.Ac
    assert abs(fcm_area - center_area)/max(fcm_area,center_area) < 0.01, "Synthesized center of RF doesn't fit the supplied sigma and amplitude (%f-%f=%f), check visual field size and model size!" % (fcm_area, center_area, abs(fcm_area - center_area))
    fsm_area = fsm[:,:,0].sum()*x_res*x_res
    surround_area = 2*numpy.pi*p.sigma_s*p.sigma_s*p.As
    assert abs(fsm_area - surround_area)/max(fsm_area,surround_area) < 0.01, "Synthesized surround of RF doesn't fit the supplied sigma and amplitude (%f-%f=%f), check visual field size and model size!" % (fsm_area, surround_area, abs(fsm_area - surround_area))
    #AssertionError: Synthesized center of RF doesn't fit the supplied sigma and amplitude (0.528030-1.570796=1.042766), check visual field size and model size!

    if p.subtract_mean:
        for i in xrange(0,numpy.shape(rf)[2]): # lets normalize each time slice separately
            rf[:,:,i] = rf[:,:,i] - rf[:,:,i].mean()
        #rf = rf - rf.mean()

    return rf


def G(t, K1, K2, c1, c2, t1, t2, n1, n2):
    p1 = K1 * ((c1*(t - t1))**n1 * np.exp(-c1*(t - t1))) / ((n1**n1) * np.exp(-n1))
    p2 = K2 * ((c2*(t - t2))**n2 * np.exp(-c2*(t - t2))) / ((n2**n2) * np.exp(-n2))
    p3 = p1 - p2
    return p3


# original AllenFreeman2006, with Linsenmeier et al. 1982
K1= 1*1.05 # transient height
c1= 0.14/1 # AllenFreeman2006
n1= 7.0 # transient center
t1= -6.0 # transient offset (ms)
K2= 1*0.7 # AllenFreeman2006
c2= 0.12 # AllenFreeman2006
n2= 8.0 # AllenFreeman2006
t2= -0.6 # susteined offset (ms)


# x = numpy.linspace(0.0, width - dx, width/dx) + dx/2.0 - width/2.0
# y = numpy.linspace(0.0, height - dy, height/dy) + dx/2.0 - height/2.0
# # t is the time at the beginning of each timestep
# t = numpy.arange(0.0, duration, dt)
# X, Y, T = meshgrid3D(y, x, t)  # x,y are reversed because (x,y) <--> (j,i)

vals = []
for T in range(300):
	vals.append( G(T, K1, K2, c1, c2, t1, t2, n1, n2) )



# # tuning curve
plt.figure()
plt.xlabel( 'time' )
plt.ylabel( 'value' )
plt.plot( vals, '-' )
plt.savefig('temporal.png')
plt.show()

