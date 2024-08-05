from PIL import Image
import numpy as np
import scanpy as sc
import cv2
import multiprocessing
from cortical_depth_utils import *
from functools import partial
import os
from itertools import product
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.sparse
import pandas as pd

dir = '/project/kyle/xuyu/all/clustering2/annotations/'
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
    dist_calculator = DistCalculator(section,image,d[image1], order_dict[image1])
    dist_calculator.mz_vz_lines()
    dist_calculator.cp_line1()
    dist_calculator.cp_line2()
    cp_mask = dist_calculator.generate_mask('cp')
    dist_calculator.find_index_pts(mask=cp_mask, cp = True)
    pool = Pool(24)
    dist = pool.map(dist_calculator.calc_cp_dist,range(dist_calculator.num_coords))
    dist = np.array(dist)

