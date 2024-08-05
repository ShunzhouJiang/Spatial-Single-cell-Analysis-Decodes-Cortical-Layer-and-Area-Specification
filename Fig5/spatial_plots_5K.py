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

cutoff = int(sys.argv[1])
os.makedirs('cutoff' + str(cutoff), exist_ok = True)

adata = sc.read('/Users/kylecoleman/Library/CloudStorage/Box-Box/MERFISH Collaboration 3/Figures/Fig6/ENVI_imputation/mers_imputed_20.h5ad')
adata2 = sc.AnnData(adata.obsm['imputation'])
adata2.obs = adata.obs
adata2.obsm['spatial'] = adata.obsm['spatial']
del adata

gw = 'gw20'
gw_dict = {'gw15': ('UMB1367', 'UMB1117'), 'gw20': ('FB080', 'FB121'), 'gw22': ('FB123', ), 'gw34': ('UMB5900', )}

regions_dict = {'UMB1367':[{'P':('flip_horizontal',), 'OP':('no_change',)}], 
'UMB1117':[{'FP':('rotate_180',), 'FP-2':('rotate_180',), 'B-1':('no_change',),'B-2':('rotate_180',), 'E-dorsal':('flip_horizontal',), 'E-lateral':('flip_vertical',), 'G':('no_change',)}], 
'FB080': [{'BD-230310':('flip_horizontal','rotate_90'), 'C':('rotate_90',), 'C-2':('rotate_270',), 'F-dorsal-3':('flip_horizontal',), 'F-dorsal4':('flip_horizontal',), 'F-lateral':('rotate_90','flip_vertical'),
'F-ventral':('flip_vertical',), 'BA17':('rotate_270',), 'BA17-2':('rotate_270',), 'BA17-3':('no_change',), 'BA17-4': ('no_change',)}],
'FB121':[{'GW20-3A':('flip_horizontal',), 'GW20-4A':('no_change',), '5A-GW20':('flip_vertical',), '6A-230221':('flip_horizontal',), '5C-rerun':('flip_horizontal',),'BA17-GW20':('rotate_180',)}],
'FB123':[{'R2-2':('no_change',), 'R3':('rotate_270',), 'R4':('no_change',), 'R5':('flip_vertical',), 'R6':('rotate_90',), 'R6-2':('rotate_180',)}], 
'UMB5900':[{'BA9':('rotate_270',), 'BA4':('no_change',),'BA123':('no_change',), 'BA40':('no_change',), 'BA40-2':('no_change',), 'BA22':('no_change',), 'BA17':('no_change',)}]}

regions_dict = {k: regions_dict[k] for k in gw_dict[gw]}


def rotate(adata, section, region, rotation):
  cells = (adata.obs['sample']==section) &(adata.obs.region==region)
  if rotation=='no_change':
    pass
  elif rotation=='rotate_90':
    adata.obsm['spatial'][cells] = np.stack((adata.obsm['spatial'][:,1][cells],adata.obsm['spatial'][:,0][cells]), axis=-1)
    adata.obsm['spatial'][:,1][cells] = max(adata.obsm['spatial'][:,1][cells])-adata.obsm['spatial'][:,1][cells]
  elif rotation=='rotate_180':
    adata.obsm['spatial'][:,0][cells] = max(adata.obsm['spatial'][:,0][cells])-adata.obsm['spatial'][:,0][cells]
    adata.obsm['spatial'][:,1][cells] = max(adata.obsm['spatial'][:,1][cells])-adata.obsm['spatial'][:,1][cells]
  elif rotation=='rotate_270':
    adata.obsm['spatial'][cells] = np.stack((adata.obsm['spatial'][:,1][cells],adata.obsm['spatial'][:,0][cells]), axis=-1)
    adata.obsm['spatial'][:,0][cells] = max(adata.obsm['spatial'][:,0][cells])-adata.obsm['spatial'][:,0][cells]
  elif rotation=='flip_horizontal':
    adata.obsm['spatial'][:,0][cells] = max(adata.obsm['spatial'][:,0][cells])-adata.obsm['spatial'][:,0][cells]
  elif rotation=='flip_vertical':
    adata.obsm['spatial'][:,1][cells] = max(adata.obsm['spatial'][:,1][cells])-adata.obsm['spatial'][:,1][cells]
  else:
    warnings.warn('rotation ' + rotation + ' not valid')
  return adata

for i in list(regions_dict.keys()):
  for k in list(regions_dict[i][0].keys()):
    rotations = regions_dict[i][0][k]
    for rotation in rotations:
      cells = (adata2.obs['sample']==i) &(adata2.obs.region==k)
      if sum(cells)>0:
        adata2 = rotate(adata2, i, k, rotation)



#locs = np.load('/Users/kylecoleman/data/walsh/all/clustering_ind/scshc/gw20_locs_rotated.npy')
#adata.obs.H3_cluster[~np.isnan(adata.obs.H3_cluster)] = adata.obs.H3_cluster[~np.isnan(adata.obs.H3_cluster)].astype('int').astype('str')
#adata.obs['H3'] = adata.obs[['H2_annotation', 'H3_cluster']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
#adata.obsm['spatial'] = locs
ncol=2

dict1 = {'UMB1367-P': 'UMB1367-P1', 'UMB1367-OP': 'UMB1367-O1', 'UMB1117-FP': 'UMB1117-F1a', 'UMB1117-FP-2': 'UMB1117-F1b', 'UMB1117-B-1': 'UMB1117-F2a',
         'UMB1117-B-2': 'UMB1117-F2b', 'UMB1117-E-dorsal': 'UMB1117-P1', 'UMB1117-E-lateral': 'UMB1117-T1', 'UMB1117-G': 'UMB1117-O1',
         'FB080-BD-230310': 'FB080-F1', 'FB080-C': 'FB080-F2a', 'FB080-C-2': 'FB080-F2b', 'FB080-F-dorsal-3': 'FB080-P1b', 'FB080-F-dorsal4': 'FB080-P1a',
         'FB080-F-lateral': 'FB080-P2', 'FB080-F-ventral': 'FB080-T1', 'FB080-BA17': 'FB080-O1a', 'FB080-BA17-2': 'FB080-O1b', 'FB080-BA17-3': 'FB080-O1c',
         'FB080-BA17-4': 'FB080-O1d', 'FB121-GW20-3A': 'FB121-F1', 'FB121-GW20-4A': 'FB121-F2', 'FB121-5A-GW20': 'FB121-P1', 'FB121-5C-rerun': 'FB121-T1',
         'FB121-6A-230221': 'FB121-P2', 'FB121-BA17-GW20': 'FB121-O1', 'FB123-R2-2': 'FB123-F1', 'FB123-R3': 'FB123-F2', 'FB123-R4': 'FB123-F3', 'FB123-R5':
         'FB123-P1', 'FB123-R6': 'FB123-O1', 'FB123-R6-2': 'FB123-O2', 'UMB5900-BA9': 'UMB5900-BA9', 'UMB5900-BA4': 'UMB5900-BA4', 'UMB5900-BA123': 'UMB5900-BA123',
         'UMB5900-BA40': 'UMB5900-BA40a', 'UMB5900-BA40-2': 'UMB5900-BA40b', 'UMB5900-BA22': 'UMB5900-BA22', 'UMB5900-BA17': 'UMB5900-BA18'}

dict2 = dict((v,k) for k,v in dict1.items())

def make_plot(sample, region, gene):
    if sum(adata2.var.index==gene)==0:
        return
    
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
    adata1 = adata2[(adata2.obs['sample']==sample) & (adata2.obs.region==region)].copy()
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.scale(adata1, zero_center=True, max_value=cutoff)
    #sc.pp.scale(adata1, zero_center=True)
    #vmin = adata1.X.min()
    #vmax = adata1.X.max()
    plot = sc.pl.embedding(adata1, basis="spatial", color = gene, show = False, s=2, color_map = cmap, alpha=1, colorbar_loc = None);
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
    plt.savefig('cutoff' + str(cutoff) + '/' + dict1[sample + '-' + region].replace('-', '_') + '_'+gene+ '.png', dpi=500); plt.clf()


#samples_genes = {'FB080-O1c': ['ABI3BP', 'PDZRN4', 'FLRT2', 'TATA2', 'NR1D1', 'IL1RAP'],
#                 'FB121-F1': ['PDZRN3', 'DLX6', 'DLX6-AS1', 'ADARB2', 'ERBB4', 'NRXN3', 'DLX2', 'ZNF536', 'PRKCA', 'THRB', 'TSHZ1', 'PBX3', 'MEIS2', 'CALB2',
#                              'CDCA7L', 'SYNPR', 'SP8', 'CASZ1', 'FOXP4', 'SP8']}

#samples_genes = {'FB080-O1c': ['CHRM2'], 'FB121-F1': ['CHRM2']}
#cmap1 = LinearSegmentedColormap.from_list("mycmap1", ['lightgrey', 'green'])
#cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['lightgrey', 'magenta'])
#cmap = [cmap1]*4 + [cmap2]*4
cmap = 'YlGnBu'

#samples = ['FB080-O1c', 'FB121-F1']
genes = list(adata2.var.index)
samples_genes = {'FB080-O1c': genes, 'FB121-F1': genes}

def main():
  for sample in list(samples_genes.keys()):
    with Pool(8) as pool:
      pool.starmap(make_plot, zip(repeat(sample.split('-')[0]), repeat(sample.split('-')[1]), samples_genes[sample]))

if __name__=="__main__":
    main()

