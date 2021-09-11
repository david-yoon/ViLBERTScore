#-*- coding: utf-8 -*-
import numpy as np

'''
print stats such as mean, max, std
data        : list(numpy) of data
'''
def print_stats(data, print_coverage=False):
    print("\t#:\t\t"  + '{:.0f}'.format(len(data))) 
    print("\tmean:\t\t" + '{:.3f}'.format(np.average(data)))
    print("\tmedian:\t\t" + '{:.3f}'.format(np.median(data)))
    print("\tstd:\t\t"  + '{:.3f}'.format(np.std(data)))
    print("\tmax:\t\t"  + '{:.3f}'.format(np.max(data)))
    print("\tmin:\t\t"  + '{:.3f}'.format(np.min(data)))
    if (print_coverage) :
        print("\t95.xx coverage: " + '{:.3f}'.format(np.average(data) +  2*np.std(data)))
        print("\t99.73 coverage: " + '{:.3f}'.format(np.average(data) +  3*np.std(data)))
        print("\t99.95 coverage: " + '{:.3f}'.format(np.average(data) +  3.5*np.std(data)))
        print("\t99.99 coverage: " + '{:.3f}'.format(np.average(data) +  4*np.std(data)))
    print('')