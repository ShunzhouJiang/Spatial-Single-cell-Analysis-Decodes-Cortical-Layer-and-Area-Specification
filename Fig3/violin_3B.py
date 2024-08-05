import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress
import glob
import os
from multiprocessing import Pool
import seaborn as sns

adata = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_no_counts_locs_rotated.h5ad')
adata.obs['sample'][adata.obs['sample']=='GW5900'] = 'UMB5900'
adata.obs['image'] = np.nan
adata1 = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_notscaled.h5ad')
adata1.X = np.exp(adata1.X.A)-1
sc.pp.normalize_total(adata1)

def find_genes(section, region, image):
  obs = pd.read_csv('/Users/kylecoleman/data/walsh/all/clustering2/annotations_completed_cortical_dist/' + section + '_' + region + '/' + image + '_obs_cp.csv', index_col = 0)
  obs['cp_dist'] = np.sqrt(obs['cp_dist'])
  adata2 = adata1[(adata1.obs['sample']==section) & (adata1.obs.region==region)].copy()
  adata2.obs.index = [i.split('-')[0] for i in adata2.obs.index]
  adata3 = adata2[obs.index.astype('str')].copy()
  adata3.obs['cp_dist'] = np.array(obs.cp_dist)
  ge = adata3.X
  ge1 = ge.round()
  ge1 = ge1.astype('int')
  dist_all = [np.concatenate([adata3.obs.cp_dist[i]*np.ones(ge1[:,j][i]) for i in range(len(adata3.obs.cp_dist)) if ge1[:,j][i]>0]) for j in range(ge1.shape[1])]
  dist40 = [dist_all[i] for i in np.array([np.quantile(j,0.75)-np.quantile(j,0.25) for j in dist_all]).argsort()[:40]]
  genes = adata1.var.index[np.array([np.quantile(i,0.75)-np.quantile(i,0.25) for i in dist_all]).argsort()[:40]]
  dict1 = dict(zip(genes, dist40))
  genes1 = [[i]*len(dict1[i]) for i in dict1.keys()]
  genes1 = [x for xs in genes1 for x in xs]
  dict1 = dict(zip(genes, dist40))
  genes1 = [[i]*len(dict1[i]) for i in dict1.keys()]
  genes1 = [x for xs in genes1 for x in xs]
  df1 = pd.DataFrame(genes1)
  df1.columns = ['gene']
  df1['cp_dist'] = np.hstack(dict1.values())
  genes2 = list(df1.groupby('gene').aggregate('median').sort_values(by='cp_dist').index)
  return genes2

def make_heatmap(section,region, image, genes):
  obs = pd.read_csv('/Users/kylecoleman/data/walsh/all/clustering2/annotations_completed_cortical_dist/' + section + '_' + region + '/' + image + '_obs_cp.csv', index_col = 0)
  obs['cp_dist'] = np.sqrt(obs['cp_dist'])
  adata2 = adata1[(adata1.obs['sample']==section) & (adata1.obs.region==region)].copy()
  adata2.obs.index = [i.split('-')[0] for i in adata2.obs.index]
  adata3 = adata2[obs.index.astype('str')].copy()
  adata3.obs['cp_dist'] = np.array(obs.cp_dist)
  adata3 = adata3[:,genes].copy()
  ge = adata3.X
  ge1 = ge.round()
  ge1 = ge1.astype('int')
  dist40 = [np.concatenate([adata3.obs.cp_dist[i]*np.ones(ge1[:,j][i]) for i in range(len(adata3.obs.cp_dist)) if ge1[:,j][i]>0]) for j in range(ge1.shape[1])]
  #dist40 = [dist_all[i] for i in
  #dist40 = [dist_all[i] for i in np.array([np.quantile(j,0.75)-np.quantile(j,0.25) for j in dist_all]).argsort()[:40]]  
  #genes = adata1.var.index[np.array([np.quantile(i,0.75)-np.quantile(i,0.25) for i in dist_all]).argsort()[:40]]
  dict1 = dict(zip(genes, dist40))
  genes1 = [[i]*len(dict1[i]) for i in dict1.keys()]
  genes1 = [x for xs in genes1 for x in xs]
  df1 = pd.DataFrame(genes1)
  df1.columns = ['gene']
  df1['cp_dist'] = np.hstack(dict1.values())
  layer_types = ['EN-L2-1', 'EN-IT-L3-A', 'EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
  l2_3 = (np.quantile(obs[obs.H3_annotation==layer_types[0]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.75))/2
  l3_4 = (np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.75))/2
  l4_5 = (np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.75))/2
  l5_6 = (np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[4]].cp_dist, 0.75))/2
  l = [l2_3,l3_4,l4_5,l5_6]
  order = genes
  plt.figure(figsize=(25,5));
  plot = sns.violinplot(x='gene', y='cp_dist', hue='gene', data=df1, order=order, density_norm='width', inner = None, dodge=False, cut=0); plot.legend().remove();
  [plot.axhline(i, linestyle = '--') for i in l];
  plt.xticks(rotation=90, fontsize=9); plt.yticks(fontsize=9); plot.set(xlabel=None); plot.set_ylabel('Cortical Depth', fontsize=20); plt.ylim(0,1); plt.tight_layout();
  plt.savefig(section + '_' + region + '_' + image + '_gene_violin1.png', dpi=200, bbox_to_inches = 'tight', pad_inches=0)

def main():
  genes = find_genes('FB123', 'R2-2', 'A-PFC')
  make_heatmap('FB123', 'R6-2', 'A-Occi', genes)
