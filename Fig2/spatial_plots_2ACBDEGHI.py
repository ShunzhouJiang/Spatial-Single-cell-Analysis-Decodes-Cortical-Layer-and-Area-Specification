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

Image.MAX_IMAGE_PIXELS = None
adata = sc.read('merscope_855_no_counts_locs_rotated.h5ad')
adata.obs['sample'][adata.obs['sample']=='GW5900'] = 'UMB5900'

gw15 = ['UMB1367', 'UMB1117']
gw20 = ['FB080', 'FB121']
gw22 = ['FB123']
gw34 = ['UMB5900']


def make_plot(sample, region, fig):
    os.makedirs(sample+'_'+region, exist_ok=True)
    adata1 = adata[(adata.obs['sample']==sample) & (adata.obs.region==region)].copy()
    if fig=='layer_def':
      ncol=1
      if sample in gw15:
        types = ['EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        colors = ['#0FF008', '#0832F0', '#F008EC']
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
      elif sample in gw20:
        types = ['EN-IT-L2/3-A1', 'EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        colors = ["#F0A608", "#0FF008", '#0832F0', '#F008EC']
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
      elif sample in gw22:
        types = ['EN-L2-1', 'EN-IT-L3-A', 'EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        colors = ["#EC2814", "#F0A608", "#0FF008", '#0832F0', '#F008EC']
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
      elif sample in gw34:
        types = ['EN-L2-4', 'EN-IT-L3-late', 'EN-IT-L4-late', 'EN-ET-L5-1', 'EN-IT-L6-late']
        colors = ["#EC2814", "#F0A608", "#0FF008", '#0832F0', '#F008EC']
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
    elif fig=='en_et':
        ncol=2
        types1 = ['EN-ET-L5-1', 'EN-ET-L6-V1', 'EN-ET-L5/6', 'EN-ET-L6-A', 'EN-ET-SP-early3', 'EN-ET-L6-early4', 'EN-ET-L6-P', 'EN-ET-SP-early4', 'EN-ET-SP-early2', 'EN-ET-SP-3', 'EN-ET-SP-A', 'EN-ET-SP-early1', 'EN-ET-L6-early1', 'EN-ET-L6-early3', 'EN-ET-SP-early5', 'EN-ET-SP-2', 'EN-ET-L6-early2', 'EN-ET-SP-1', 'EN-ET-SP-5', 'EN-ET-SP-4', 'EN-ET-SP-P2', 'EN-ET-SP-P1', 'EN-ET-SP-V1T2', 'EN-ET-L6-early5', 'EN-ET-SP-V1T1']
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939']
        color_dict = dict(zip(types1, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types1), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types1)))))      
        #cmap = ListedColormap(np.vstack((tab20(np.linspace(0,1,20)),tab20b(np.linspace(0,1,20)))))
        types = list(adata1.obs[adata1.obs.H1_annotation=='EN-ET'].H3_annotation.unique())
        types = [ele for ele in types1 if ele in types]
        #colors = [cmap(i) for i in range(25)]
        #color_dict = dict(zip(types, [matplotlib.colors.rgb2hex(i) for i in colors]))
        #color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
    elif fig=='en_it_deep':
        ncol=2
        types = ['EN-IT-L4-1', 'EN-IT-Hip', 'EN-IT-L6-2', 'EN-IT-L5-1', 'EN-IT-L4/5-early', 'EN-IT-L4/5-late', 'EN-IT-L4/5-1', 'EN-IT-L6-1', 'EN-IT-L6-late', 'EN-IT-L5/6-P']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        #tab10 = cm.get_cmap('tab10')
        #colors = [matplotlib.colors.rgb2hex(tab10(i)) for i in range(10)]
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))
    elif fig=='en_it_upper':
        ncol=2
        types = ['EN-IT-L3-V1', 'EN-IT-L2/3-A2', 'EN-IT-L3-A', 'EN-IT-L3/4-1', 'EN-IT-L2/3-A1', 'EN-IT-L3/4-early', 'EN-IT-L4-A', 'EN-IT-L4-V1', 'EN-IT-L3-late', 'EN-IT-L3/4-P', 'EN-IT-L3/4-P2', 'EN-IT-L3-P', 'EN-IT-L4-late', 'EN-IT-L3/4-T']
        #types = list(adata.obs[adata.obs.H2_annotation.isin(['EN-IT-L2/3', 'EN-IT-L4','EN-IT-L3/4'])].H3_annotation.unique())
        #colors = [matplotlib.colors.rgb2hex(tab20(i)) for i in range(14)]
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2']
        color_dict = dict(zip(types, colors))
        color_dict.update(dict(zip(np.setdiff1d(adata.obs.H3_annotation.unique(), types), ['grey']*len(np.setdiff1d(adata.obs.H3_annotation.unique(), types)))))  
    plot = sc.pl.embedding(adata1, basis="spatial", color = 'H3_annotation', groups = types, show = False, s=2, palette = color_dict, alpha=1); plt.axis('off');
    plot.set_aspect('equal');
    handles, labels = plt.gca().get_legend_handles_labels();
    order = [labels.index(i) for i in types];    
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'center', fontsize=2, ncol = ncol, bbox_to_anchor=(1.0,1.0), markerscale=0.25);
    #plot.legend(loc = 'center', fontsize=3, ncol = 1, bbox_to_anchor=(0.95,1), markerscale=0.5);
    plot.get_figure().gca().set_title('');
    scalebar = ScaleBar(1, "um", fixed_value=500, location = 'lower right');
    plot.add_artist(scalebar);
    plt.tight_layout();
    plt.savefig(sample+'_'+region + '/' + sample+'_'+region+'_'+fig+'.png', dpi=500); plt.clf()


figs = ['layer_def', 'en_et', 'en_it_deep', 'en_it_upper']


sections = ['UMB1117_FP', 'UMB1367_OP', 'FB080_BA17-3','FB121_GW20-3A', 'FB123_R2-2', 'FB123_R3','FB123_R4','FB123_R5','FB123_R6-2','UMB5900_BA9','UMB5900_BA17']

def main():
  for i in sections:
    with Pool(4) as pool:
      pool.starmap(make_plot, zip(repeat(i.split('_')[0]), repeat(i.split('_')[1]), figs))

if __name__=="__main__":
    main()

