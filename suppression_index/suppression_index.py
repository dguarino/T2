import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt


##################################
# INACTIVE
inactive_data = np.loadtxt( open("suppression_inactive.csv","rb"), delimiter=",", skiprows=6)
# print inactive_data
# print inactive_data.shape

# difference between 2deg and 6deg (first and second column)
inactive_suppression = inactive_data[:,0] - inactive_data[:,1]
# print inactive_suppression # by cell
inactive_hist, inactive_bin_edges = np.histogram(inactive_suppression, bins=10)
inactive_hist = inactive_hist[::-1] # reversed
inactive_bin_edges = inactive_bin_edges[::-1]
# print inactive_hist
# print inactive_bin_edges

width = inactive_bin_edges[1] - inactive_bin_edges[0]
center = (inactive_bin_edges[:-1] + inactive_bin_edges[1:]) / 2
plt.bar(center, inactive_hist, align='center', width=width, facecolor='white')
plt.xlabel('Index of end-inhibition (open)')
plt.ylabel('Number of cells')
plt.axis([inactive_bin_edges[0], inactive_bin_edges[-1], 0, 10])
plt.xticks(inactive_bin_edges, (10,9,8,7,6,5,4,3,2,1))
plt.savefig( "suppression_index_inactive.png", dpi=200 )
plt.close()


##################################
# ACTIVE
active_data = np.loadtxt( open("suppression_active.csv","rb"), delimiter=",", skiprows=6)
active_suppression = active_data[:,0] - active_data[:,1]
print active_suppression # by cell
active_hist, active_bin_edges = np.histogram(active_suppression, bins=10)
print active_hist
print active_bin_edges
active_hist = active_hist[::-1] # reversed
active_bin_edges = active_bin_edges[::-1]
print active_hist
print active_bin_edges

width = active_bin_edges[1] - active_bin_edges[0]
center = (active_bin_edges[:-1] + active_bin_edges[1:]) / 2
plt.bar(center, active_hist, align='center', width=width, facecolor='white')
plt.xlabel('Index of end-inhibition (closed)')
plt.ylabel('Number of cells')
plt.axis([active_bin_edges[0], active_bin_edges[-1], 0, 10])
plt.xticks(active_bin_edges, (10,9,8,7,6,5,4,3,2,1))
plt.savefig( "suppression_index_active.png", dpi=200 )
plt.close()
# plt.show()