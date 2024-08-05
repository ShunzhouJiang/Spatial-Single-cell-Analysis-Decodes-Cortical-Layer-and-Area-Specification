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


adata = sc.read('/Users/kylecoleman/data/walsh/all/clustering_ind/scshc/gw20.h5ad')
locs = np.load('/Users/kylecoleman/data/walsh/all/clustering_ind/scshc/gw20_locs_rotated.npy')
adata.obs.H3_cluster[~np.isnan(adata.obs.H3_cluster)] = adata.obs.H3_cluster[~np.isnan(adata.obs.H3_cluster)].astype('int').astype('str')
adata.obs['H3'] = adata.obs[['H2_annotation', 'H3_cluster']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
adata.obsm['spatial'] = locs
ncol=2
gw20 = ['FB080', 'FB121']

dict1 = {'UMB1367-P': 'UMB1367-P1', 'UMB1367-OP': 'UMB1367-O1', 'UMB1117-FP': 'UMB1117-F1a', 'UMB1117-FP-2': 'UMB1117-F1b', 'UMB1117-B-1': 'UMB1117-F2a',
         'UMB1117-B-2': 'UMB1117-F2b', 'UMB1117-E-dorsal': 'UMB1117-P1', 'UMB1117-E-lateral': 'UMB1117-T1', 'UMB1117-G': 'UMB1117-O1',
         'FB080-BD-230310': 'FB080-F1', 'FB080-C': 'FB080-F2a', 'FB080-C-2': 'FB080-F2b', 'FB080-F-dorsal-3': 'FB080-P1b', 'FB080-F-dorsal4': 'FB080-P1a',
         'FB080-F-lateral': 'FB080-P2', 'FB080-F-ventral': 'FB080-T1', 'FB080-BA17': 'FB080-O1a', 'FB080-BA17-2': 'FB080-O1b', 'FB080-BA17-3': 'FB080-O1c',
         'FB080-BA17-4': 'FB080-O1d', 'FB121-GW20-3A': 'FB121-F1', 'FB121-GW20-4A': 'FB121-F2', 'FB121-5A-GW20': 'FB121-P1', 'FB121-5C-rerun': 'FB121-T1',
         'FB121-6A-230221': 'FB121-P2', 'FB121-BA17-GW20': 'FB121-O1', 'FB123-R2-2': 'FB123-F1', 'FB123-R3': 'FB123-F2', 'FB123-R4': 'FB123-F3', 'FB123-R5':
         'FB123-P1', 'FB123-R6': 'FB123-O1', 'FB123-R6-2': 'FB123-O2', 'UMB5900-BA9': 'UMB5900-BA9', 'UMB5900-BA4': 'UMB5900-BA4', 'UMB5900-BA123': 'UMB5900-BA123',
         'UMB5900-BA40': 'UMB5900-BA40a', 'UMB5900-BA40-2': 'UMB5900-BA40b', 'UMB5900-BA22': 'UMB5900-BA22', 'UMB5900-BA17': 'UMB5900-BA18'}

dict2 = dict((v,k) for k,v in dict1.items())


regions_dict = {'UMB1367':[{'P':('flip_horizontal',), 'OP':('no_change',)}], 
'UMB1117':[{'FP':('rotate_180',), 'FP-2':('rotate_180',), 'B-1':('no_change',),'B-2':('rotate_180',), 'E-dorsal':('flip_horizontal',), 'E-lateral':('flip_vertical',), 'G':('no_change',)}], 
'FB080': [{'BD-230310':('flip_horizontal','rotate_90'), 'C':('rotate_90',), 'C-2':('rotate_270',), 'F-dorsal-3':('flip_horizontal',), 'F-dorsal4':('flip_horizontal',), 'F-lateral':('rotate_90','flip_vertical'),
'F-ventral':('flip_vertical',), 'BA17':('rotate_270',), 'BA17-2':('rotate_270',), 'BA17-3':('no_change',), 'BA17-4': ('no_change',)}],
'FB121':[{'GW20-3A':('flip_horizontal',), 'GW20-4A':('no_change',), '5A-GW20':('flip_vertical',), '6A-230221':('flip_horizontal',), '5C-rerun':('flip_horizontal',),'BA17-GW20':('rotate_180',)}],
'FB123':[{'R2-2':('no_change',), 'R3':('rotate_270',), 'R4':('no_change',), 'R5':('flip_vertical',), 'R6':('rotate_90',), 'R6-2':('rotate_180',)}], 
'UMB5900':[{'BA9':('rotate_270',), 'BA4':('no_change',),'BA123':('no_change',), 'BA40':('no_change',), 'BA40-2':('no_change',), 'BA22':('no_change',), 'BA17':('no_change',)}]}

gw_dict = {'15':('UMB1367', 'UMB1117'), '20':('FB080','FB121'),'22':('FB123',),'34':('UMB5900',)}
gw = '20'

types1 = ['EN-IT-L3_0', 'EN-IT-L3/4_3', 'EN-ET-L5/6_4', 'EN-ET-SP-2_1', 'EN-IT-L3_3', 'EN-IT-L3/4_0', 'EN-ET-L5/6_3', 'EN-ET-SP-2_3']
colors1 = ['#FF97EE', '#FF00FF', '#BC00BC', '#A800A8', '#ACFC64', '#00FF00', '#00C800', '#007E00']


def make_plot(j):
    types = [types1[j]]
    colors = [colors1[j]]         
    fig, axes = plt.subplots(2,11, figsize=(30,10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(len(gw_dict[gw])):
      for k in range(len(regions_dict[gw_dict[gw][i]][0])):
        region = list(regions_dict[gw_dict[gw][i]][0].keys())[k]
        sample1 = gw_dict[gw][i] + '-'+ region
        sample2 = dict1[sample1]
        sample = sample1.split('-',1)[0]
        region = sample1.split('-',1)[1]
        adata1 = adata[(adata.obs['sample']==sample) &(adata.obs.region==region)].copy()
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata1.obs.H3.unique(), types), ['grey']*len(np.setdiff1d(adata1.obs.H3.unique(), types)))))
        sc.pl.embedding(adata1, basis="spatial", color = 'H3', groups = types, palette=color_dict, alpha=1, show=False, ax = axes[i,k])
        axes[i,k].set_aspect('equal')
        axes[i,k].set_title(sample2.split('-',1)[1], size=30, loc= 'center', va = 'top')
        axes[i,k].set_xlabel('')
        axes[i,k].set_ylabel('')
        axes[i,k].get_legend().remove()
        #scalebar = ScaleBar(1, "um", fixed_value=500, location = 'lower right');
        #axes[i,k].add_artist(scalebar);
    for i in range(len(gw_dict[gw])):
      for k in range(11):
        axes[i,k].axis('off')
    for ax, row in zip(axes[:,0] , list(gw_dict[gw])):
      ax.axis('on')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.set_ylabel(row, rotation=0, size=30)
      ax.yaxis.set_label_coords(-0.5, .5)  
    plt.tight_layout();
    plt.suptitle('GW20: ' + types1[j], size = 35)
    plt.savefig(types1[j].replace('-', '_').replace('/', '_') + '.png', bbox_inches = 'tight', dpi=500)
    plt.clf()


def main():
  with Pool(8) as pool:
    pool.map(make_plot, list(range(len(types1))))

if __name__=="__main__":
    main()

