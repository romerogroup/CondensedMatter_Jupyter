#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from ase.units import Bohr
import os.path
from collections import namedtuple

def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=0,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=2,
                     xticks=None,
                     title=None):
    """
    kslist:  [iband, kx]
    ekslist: [iband, energy]
    wkslist: [iband, weight]
    """
    if axis is None:
        fig, a = plt.subplots()
        plt.tight_layout(pad=2.19)
        plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
    else:
        a = axis
    if title is not None:
        a.set_title(title)

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - 66,
                  np.array(ekslist).flatten().max() + 66)

    if wkslist is not None:
        for i in range(len(kslist)): # number of bands
            x = kslist[i]            # bandx
            y = ekslist[i]           # band energy
            lwidths = np.array(wkslist[i]) * width
            #lwidths=np.ones(len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lc = LineCollection(
                    segments,
                    linewidths=[2] * len(x),
                    colors=[
                        colorConverter.to_rgba(
                            color, alpha=np.abs(lwidth / (width + 0.001)))
                        for lwidth in lwidths
                    ])

            a.add_collection(lc)
    plt.ylabel('Frequency (cm$^{-1}$)')
    if axis is None:
        for ks, eks in zip(kslist, ekslist):
            plt.plot(ks, eks, color='gray', linewidth=0.001)
        a.set_xlim(0, xmax)
        a.set_ylim(yrange)
        if xticks is not None:
            plt.xticks(xticks[1], xticks[0])
        for x in xticks[1]:
            plt.axvline(x, color='gray', linewidth=0.5)
        if efermi is not None:
            plt.axhline(linestyle='--', color='black')
    return a


