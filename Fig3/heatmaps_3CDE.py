import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress
import glob
import os
from multiprocessing import Pool

adata = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_no_counts_locs_rotated.h5ad')
adata.obs['sample'][adata.obs['sample']=='GW5900'] = 'UMB5900'
adata.obs['image'] = np.nan
adata1 = sc.read('/Users/kylecoleman/data/walsh/all/clustering2/merscope_855_notscaled.h5ad')
adata1.X = np.exp(adata1.X.A)-1
sc.pp.normalize_total(adata1)
#sc.pp.log1p(adata1)
adata.X = adata1.X
del adata1
#cbln2 = np.load('/Users/kylecoleman/data/walsh/all/clustering2/cbln2_notscaled.npy')
#adata = adata[:,'CBLN2'].copy()
#adata.X = cbln2
#sc.pp.scale(adata.X)
dir = '/Users/kylecoleman/data/walsh/all/clustering2/fig3'
gw15 = ['UMB1367', 'UMB1117']
gw20 = ['FB080', 'FB121']
gw22 = ['FB123']
gw34 = ['UMB5900']

#layers = ['mz', 'l2', 'l3', 'l4', 'l5', 'l6', 'sp', 'iz', 'osvz', 'isvz', 'vz']


def mean_exp(adata, section, image, layers):
  layer_dict = dict(zip(layers, [np.nan]*len(layers)))
  os.chdir(dir+'/'+section)
  sample1 = section.split('_')[0]
  region = section.split('_')[1]
  adata2 = adata[(adata.obs['sample']==sample1) & (adata.obs.region==region)].copy()
  #sc.pp.scale(adata1)
  #full_index = np.load(section+'_'+image+'_index.npy').astype('str')
  #adata1 = adata1[full_index].copy()
  for layer in layers:
    layer_index = np.load(section+'_'+image+'_'+layer+'_index.npy').astype('str')
    adata3 = adata2[layer_index].copy()
    if len(layer_index)>0:
      layer_dict[layer] = adata3.X.mean()
    else:
      layer_dict[layer] = np.nan
  return layer_dict



def cp_annotation(section, image):
      os.chdir(dir+'/'+section)
      #sample1 = section.split('_')[0]
      #region = section.split('_')[1]
      #adata1 = adata[(adata.obs['sample']==sample1) & (adata.obs.region==region)].copy()
      obs = pd.read_csv('/Users/kylecoleman/data/walsh/all/clustering2/annotations_completed_cortical_dist/'+section+'/'+image+'_obs_cp.csv', index_col = 0)
      obs.cp_dist = np.sqrt(obs.cp_dist)
      if section.split('_')[0] in gw15:
        cp_layers = ['l4','l5','l6']
        layer_types = ['EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        l4_5 = (np.quantile(obs[obs.H3_annotation==layer_types[0]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.75))/2
        l5_6 = (np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.75))/2
        l = [l4_5,l5_6,0]
      elif section.split('_')[0] in gw20:
        cp_layers = ['l3','l4','l5','l6']
        layer_types = ['EN-IT-L2/3-A1', 'EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        l3_4 = (np.quantile(obs[obs.H3_annotation==layer_types[0]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.75))/2
        l4_5 = (np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.75))/2
        l5_6 = (np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.75))/2
        l = [l3_4,l4_5,l5_6,0]
      elif section.split('_')[0] in gw22:
        cp_layers = ['l2','l3','l4','l5','l6']
        layer_types = ['EN-L2-1', 'EN-IT-L3-A', 'EN-IT-L4-1', 'EN-ET-L5-1', 'EN-IT-L6-1']
        l2_3 = (np.quantile(obs[obs.H3_annotation==layer_types[0]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.75))/2
        l3_4 = (np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.75))/2
        l4_5 = (np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.75))/2
        l5_6 = (np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[4]].cp_dist, 0.75))/2
        l = [l2_3,l3_4,l4_5,l5_6,0]
      elif section.split('_')[0] in gw34:
        cp_layers = ['l2','l3','l4','l5','l6']
        layer_types = ['EN-L2-4', 'EN-IT-L3-late', 'EN-IT-L4-late', 'EN-ET-L5-1', 'EN-IT-L6-late']
        l2_3 = (np.quantile(obs[obs.H3_annotation==layer_types[0]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.75))/2
        l3_4 = (np.quantile(obs[obs.H3_annotation==layer_types[1]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.75))/2
        l4_5 = (np.quantile(obs[obs.H3_annotation==layer_types[2]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.75))/2
        l5_6 = (np.quantile(obs[obs.H3_annotation==layer_types[3]].cp_dist, 0.25)+np.quantile(obs[obs.H3_annotation==layer_types[4]].cp_dist, 0.75))/2
        l = [l2_3,l3_4,l4_5,l5_6,0]
  
      layer_ann = [cp_layers[np.where((i-l)>=0)[0][0]] for i in np.array(obs.cp_dist)]
      layer_list = ['l2','l3','l4','l5','l6']
      [np.save(section+'_'+image+'_'+j+'_index.npy', np.array(list(compress(obs.index, [i==j for i in layer_ann])))) for j in layer_list]

dirs = os.listdir('/Users/kylecoleman/data/walsh/all/clustering2/annotations_completed_cortical_dist')
if '.DS_Store' in dirs:
  dirs.remove('.DS_Store')

#dirs.remove('FB121_6A-230221')
dirs.remove('UMB1117_G')
dirs.remove('FB080_BA17-2')

os.chdir('/Users/kylecoleman/data/walsh/all/clustering2/annotations_completed_cortical_dist')  
dict1 = dict(zip(dirs,[[i.split('/')[1].split('_')[0] for i in glob.glob(j+'/*.csv')] for j in dirs]))
dict1['FB121_6A-230221'] = ['A-Par']

order = ['PFC', 'PMC', 'M1', 'S1', 'Par', 'Temp', 'Occi', 'V2', 'V1']


sections_all = [[i]*len(dict1[i]) for i in dict1.keys()]
sections_all = [x for xs in sections_all for x in xs]

def make_heatmap(gene):
  print(gene)
  adata1 = adata[:,gene].copy()
  layers = ['l2', 'l3', 'l4', 'l5', 'l6', 'sp', 'iz', 'osvz', 'isvz', 'vz']
  #layers = ['mz', 'l2', 'l3', 'l4', 'l5', 'l6', 'sp', 'iz', 'osvz', 'isvz', 'vz']
  sections = list(compress(sections_all, [sum([i.startswith(j) for j in gw15]) for i in sections_all]))
  _, idx = np.unique(sections, return_index = True)
  sections_unique = list(np.array(sections)[np.sort(idx)])
  images = [dict1[i] for i in sections_unique]
  images = [x for xs in images for x in xs]
  #[cp_annotation(i,j) for i,j in zip(sections, images)]
  gw15_dict = [mean_exp(adata1,i,j,layers) for i,j in zip(sections, images)]
  gw15_df = pd.DataFrame(gw15_dict).transpose()
  images = [i.split('-')[1] if (len(i.split('-')) > 1) else i for i in images]
  gw15_df.columns = images
  images_ordered = [i for j in order for i in images if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw15_df = gw15_df[images_ordered]
  #gw15_df = (gw15_df - gw15_df.min().min()) /(gw15_df.max().max() - gw15_df.min().min())
  #sns.heatmap(gw15_df); plt.show()
  #dir='/Users/kylecoleman/data/walsh/all/clustering2/fig3/gw20'
  sections = list(compress(sections_all, [sum([i.startswith(j) for j in gw20]) for i in sections_all]))
  _, idx = np.unique(sections, return_index = True)
  sections_unique = list(np.array(sections)[np.sort(idx)])
  images = [dict1[i] for i in sections_unique]
  images = [x for xs in images for x in xs]
  #[cp_annotation(i,j) for i,j in zip(sections, images)]
  gw20_dict = [mean_exp(adata1,i,j,layers) for i,j in zip(sections, images)]
  gw20_df = pd.DataFrame(gw20_dict).transpose()
  images = [i.split('-')[1] if (len(i.split('-')) > 1) else i for i in images]
  gw20_df.columns = images
  images_ordered = [i for j in order for i in images if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw20_df = gw20_df[images_ordered]
  #gw20_df = (gw20_df - gw20_df.min().min()) /(gw20_df.max().max() - gw20_df.min().min())
  #sns.heatmap(gw20_df); plt.show()
  #dir='/Users/kylecoleman/data/walsh/all/clustering2/fig3/gw22'
  sections = list(compress(sections_all, [sum([i.startswith(j) for j in gw22]) for i in sections_all]))
  _, idx = np.unique(sections, return_index = True)
  sections_unique = list(np.array(sections)[np.sort(idx)])
  images = [dict1[i] for i in sections_unique]
  images = [x for xs in images for x in xs]
  #[cp_annotation(i,j) for i,j in zip(sections, images)]
  gw22_dict = [mean_exp(adata1,i,j,layers) for i,j in zip(sections, images)]
  gw22_df = pd.DataFrame(gw22_dict).transpose()
  images = [i.split('-')[1] if (len(i.split('-')) > 1) else i for i in images]
  gw22_df.columns = images
  images_ordered = [i for j in order for i in images if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw22_df = gw22_df[images_ordered]
  #gw22_df = (gw22_df - gw22_df.min().min()) /(gw22_df.max().max() - gw22_df.min().min())
  #sns.heatmap(gw22_df); plt.show()
  layers = ['l2', 'l3', 'l4', 'l5', 'l6']
  #dir='/Users/kylecoleman/data/walsh/all/clustering2/fig3/gw34'
  sections = list(compress(sections_all, [sum([i.startswith(j) for j in gw34]) for i in sections_all]))
  _, idx = np.unique(sections, return_index = True)
  sections_unique = list(np.array(sections)[np.sort(idx)])
  images = [dict1[i] for i in sections_unique]
  images = [x for xs in images for x in xs]
  #[cp_annotation(i,j) for i,j in zip(sections, images)]
  gw34_dict = [mean_exp(adata1,i,j,layers) for i,j in zip(sections, images)]
  gw34_df = pd.DataFrame(gw34_dict).transpose()
  images = [i.split('-')[1] if (len(i.split('-')) > 1) else i for i in images]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw34_df.columns = images
  images_ordered = [i for j in order for i in images if i==j]
  gw34_df = gw34_df[images_ordered]
  #gw34_df = (gw34_df - gw34_df.min().min()) /(gw34_df.max().max() - gw34_df.min().min())
  gw34_df = pd.concat((gw34_df, pd.DataFrame(np.nan, index = ['sp', 'iz', 'osvz', 'isvz', 'vz'], columns = images)))
  #sns.heatmap(gw34_df); plt.show()
  grid_max = max(gw15_df.max().max(), gw20_df.max().max(), gw22_df.max().max(), gw34_df.max().max())
  grin_min = min(gw15_df.min().min(), gw20_df.min().min(), gw22_df.min().min(), gw34_df.min().min())
  gw15_df = (gw15_df - grin_min) /(grid_max - grin_min)
  gw20_df = (gw20_df - grin_min) /(grid_max - grin_min)
  gw22_df = (gw22_df - grin_min) /(grid_max - grin_min)
  gw34_df = (gw34_df - grin_min) /(grid_max - grin_min)
  gw15_df = gw15_df.groupby(level=0, axis=1).agg(np.mean)
  gw20_df = gw20_df.groupby(level=0, axis=1).agg(np.mean)
  gw22_df = gw22_df.groupby(level=0, axis=1).agg(np.mean)
  gw34_df = gw34_df.groupby(level=0, axis=1).agg(np.mean)
  images_ordered = [i for j in order for i in gw15_df.columns if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw15_df = gw15_df[images_ordered]
  images_ordered = [i for j in order for i in gw20_df.columns if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw20_df = gw20_df[images_ordered]
  images_ordered = [i for j in order for i in gw22_df.columns if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw22_df = gw22_df[images_ordered]
  images_ordered = [i for j in order for i in gw34_df.columns if i==j]
  _, idx = np.unique(images_ordered, return_index = True)
  images_ordered = list(np.array(images_ordered)[np.sort(idx)])
  gw34_df = gw34_df[images_ordered]
  grid_max = max(gw15_df.max().max(), gw20_df.max().max(), gw22_df.max().max(), gw34_df.max().max())
  grin_min = min(gw15_df.min().min(), gw20_df.min().min(), gw22_df.min().min(), gw34_df.min().min())
  gw15_df = (gw15_df - grin_min) /(grid_max - grin_min)
  gw20_df = (gw20_df - grin_min) /(grid_max - grin_min)
  gw22_df = (gw22_df - grin_min) /(grid_max - grin_min)
  gw34_df = (gw34_df - grin_min) /(grid_max - grin_min)
  fig, axs = plt.subplots(figsize = (19,10), ncols=5, gridspec_kw=dict(width_ratios=[5,6,5,6,1]));
  sns.heatmap(gw15_df, cbar = False, ax = axs[0], cmap = 'rainbow', vmin=0, vmax=1); axs[0].set_title('GW15', size=25); 
  sns.heatmap(gw20_df, yticklabels = False, cbar = False, ax = axs[1], cmap = 'rainbow', vmin=0, vmax=1); axs[1].set_title('GW20', size=25);
  sns.heatmap(gw22_df, yticklabels = False, cbar = False, ax = axs[2], cmap = 'rainbow', vmin=0, vmax=1); axs[2].set_title('GW22', size=25);
  sns.heatmap(gw34_df, yticklabels = False, cbar = False, ax = axs[3], cmap = 'rainbow', vmin=0, vmax=1); axs[3].set_title('GW34', size=25);
  fig.colorbar(axs[0].collections[0], cax=axs[4]);
  plt.title(gene, fontsize=20);
  axs[0].tick_params(axis='both', which='major', labelsize=15)
  axs[1].tick_params(axis='both', which='major', labelsize=15)
  axs[2].tick_params(axis='both', which='major', labelsize=15)
  axs[3].tick_params(axis='both', which='major', labelsize=15)
  plt.tight_layout();
  plt.savefig('/Users/kylecoleman/data/walsh/all/clustering2/fig3/heatmaps_norm_mean/' + gene + '.png', dpi=500);
  plt.clf()

pool = Pool(10)
pool.map(make_heatmap,adata.var.index)


