from PIL import Image
import numpy as np
import scanpy as sc
import cv2
import multiprocessing
from relative_height_utils import *
from functools import partial
import os
from itertools import product
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.sparse
import glob
import pandas as pd

dir = '/project/kyle/xuyu/all/clustering2/annotations/'
obs = pd.read_csv('merscope_855_obs.csv', index_col = 0)
sections = []
for f in os.listdir(dir):
  if not f.startswith('.'):
    sections.append(f)

for section in sections:
  print(section)
  os.chdir(dir+section)
  d = {}
  with open("direction.txt") as f:
    for line in f:
      (key, val) =  line.split(': ')
      d[key] = val.rstrip('\n')

  order_dict = {}
  with open("order.txt") as f:
    for line in f:
      (key, val) =  line.split(': ')
      order_dict[key] = val.rstrip('\n')

  images = []
  for file in os.listdir('./'):
    if file.endswith(".png"):
      images.append(file)
  for image in images:
    print(image)
    image1 = image.split('.')[0].split('_')[-1]  
    dist_calculator = DistCalculator()
    dist_calculator.calculate_pts(section,image,d[image1], order_dict[image1])
    #np.save('x_cr.npy', dist_calculator.x_cr)
    #scipy.sparse.save_npz('mz.npz', dist_calculator.mz)
    #scipy.sparse.save_npz('vz.npz', dist_calculator.vz)
    #np.save('o_cr.npy', dist_calculator.o_cr)
    pool = Pool(24)
    #calc_dist1 = partial(calc_dist,pts=pts)
    #dist = dist_calculator.calc_dist_parallel()
    dist = pool.map(dist_calculator.calc_dist,range(dist_calculator.num_coords))
    #dist = pool.starmap(calc_dist, product(range(coords.shape[0]),[pts]))
    #dist = pool.starmap(calc_dist, zip(range(n_pts),pts*n_pts))
    dist = np.array(dist)
    sample,region = section.split('_')
    obs1 = obs[(obs['sample']==sample)&(obs.region==region)]
    index = np.load(section + '_' + image1 + '_index.npy')
    index = index[~np.isnan(dist)]
    dist = dist[~np.isnan(dist)]
    obs2 = obs1.loc[index]
    obs2['vz_mz_dist'] = dist
    obs2.to_csv(image1+'_obs.csv')

    

