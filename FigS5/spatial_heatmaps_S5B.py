import numpy as np
import cv2
from PIL import Image
import scanpy as sc
import sys
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import warnings
import matplotlib.pyplot as plt
import os
from itertools import compress
import glob
from multiprocessing import Pool
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
import seaborn as sns
import matplotlib.colors
from itertools import repeat
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


adata = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_notscaled.h5ad')
adata.obs['sample'][adata.obs['sample']=='GW5900'] = 'UMB5900'
adata1 = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_no_counts_locs_rotated.h5ad')
adata.obsm['spatial'] = adata1.obsm['spatial']
ncol=2

dict1 = {'UMB1367-P': 'UMB1367-P1', 'UMB1367-OP': 'UMB1367-O1', 'UMB1117-FP': 'UMB1117-F1a', 'UMB1117-FP-2': 'UMB1117-F1b', 'UMB1117-B-1': 'UMB1117-F2a',
         'UMB1117-B-2': 'UMB1117-F2b', 'UMB1117-E-dorsal': 'UMB1117-P1', 'UMB1117-E-lateral': 'UMB1117-T1', 'UMB1117-G': 'UMB1117-O1',
         'FB080-BD-230310': 'FB080-F1', 'FB080-C': 'FB080-F2a', 'FB080-C-2': 'FB080-F2b', 'FB080-F-dorsal-3': 'FB080-P1b', 'FB080-F-dorsal4': 'FB080-P1a',
         'FB080-F-lateral': 'FB080-P2', 's': 'FB080-T1', 'FB080-BA17': 'FB080-O1a', 'FB080-BA17-2': 'FB080-O1b', 'FB080-BA17-3': 'FB080-O1c',
         'FB080-BA17-4': 'FB080-O1d', 'FB121-GW20-3A': 'FB121-F1', 'FB121-GW20-4A': 'FB121-F2', 'FB121-5A-GW20': 'FB121-P1', 'FB121-5C-rerun': 'FB121-T1',
         'FB121-6A-230221': 'FB121-P2', 'FB121-BA17-GW20': 'FB121-O1', 'FB123-R2-2': 'FB123-F1', 'FB123-R3': 'FB123-F2', 'FB123-R4': 'FB123-F3', 'FB123-R5':
         'FB123-P1', 'FB123-R6': 'FB123-O1', 'FB123-R6-2': 'FB123-O2', 'UMB5900-BA9': 'UMB5900-BA9', 'UMB5900-BA4': 'UMB5900-BA4', 'UMB5900-BA123': 'UMB5900-BA123',
         'UMB5900-BA40': 'UMB5900-BA40a', 'UMB5900-BA40-2': 'UMB5900-BA40b', 'UMB5900-BA22': 'UMB5900-BA22', 'UMB5900-BA17': 'UMB5900-BA18'}

dict2 = dict((v,k) for k,v in dict1.items())

def make_plot(sample, region, gene, cmap):
    sample1 = dict2[sample+'-'+region]
    sample = sample1.split('-',1)[0]
    region = sample1.split('-',1)[1]
    if os.path.exists('/Users/kylecoleman/data/walsh/all/clustering2/annotations/'+sample+'_'+region+'/increase_dimensions.txt'):
      d = {}
      with open('/Users/kylecoleman/data/walsh/all/clustering2/annotations/'+sample+'_'+region+'/increase_dimensions.txt') as f:
        for line in f:
          (key, val) = line.split(':')
          d[key] = int(val)
      if 'height' in d:
        height = d['height']
      else:
        height=0
      if 'width' in d:
        width=d['width']
      else: 
        width=0
    else:
      height=0
      width=0
    adata1 = adata[(adata.obs['sample']==sample) & (adata.obs.region==region)].copy()
    sc.pp.scale(adata1, zero_center = True, max_value = 6)
    vmin = adata1.X.min()
    vmax = adata1.X.max()
    plot = sc.pl.embedding(adata1, basis="spatial", color = gene, show = False, s=2, color_map = cmap, alpha=1, colorbar_loc = None);
    #plot = sc.pl.embedding(adata1, basis="spatial", color = gene, show = False, s=2, color_map = cmap, alpha=1, vmin=vmin, vmax=vmax, colorbar_loc = None);
    norm = matplotlib.colors.Normalize(vmin=adata1[:,gene].X.min(), vmax=adata1[:,gene].X.max());
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); 
    sm.set_array([]);
    plt.colorbar(sm, location = 'top', orientation = 'horizontal', label = gene, shrink = 0.3);
    plt.axis('off');
    plot.set_aspect('equal');
    #handles, labels = plt.gca().get_legend_handles_labels();
    #order = [labels.index(i) for i in types];    
    #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'center', fontsize=2, ncol = ncol, bbox_to_anchor=(1.0,1.0), markerscale=0.25);
    #plot.legend(loc = 'center', fontsize=3, ncol = 1, bbox_to_anchor=(0.95,1), markerscale=0.5);
    plot.get_figure().gca().set_title('');
    scalebar = ScaleBar(1, "um", fixed_value=500, location = 'lower right');
    plot.add_artist(scalebar);
    plt.tight_layout();
    plt.savefig(dict1[sample + '-' + region].replace('-', '_') + '_'+gene+ '.png', dpi=500); plt.clf()


samples = ['FB121-P1']

genes = ['SATB2', 'BCL11B', 'TBR1']

#cmap1 = LinearSegmentedColormap.from_list("mycmap1", ['lightgrey', 'green'])
#cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['lightgrey', 'magenta'])
#cmap = [cmap1]*4 + [cmap2]*4
cmap = ['YlGnBu']*8

def main():
  for sample in samples:
    with Pool(8) as pool:
      pool.starmap(make_plot, zip(repeat(sample.split('-')[0]), repeat(sample.split('-')[1]), genes, cmap))

if __name__=="__main__":
    main()

