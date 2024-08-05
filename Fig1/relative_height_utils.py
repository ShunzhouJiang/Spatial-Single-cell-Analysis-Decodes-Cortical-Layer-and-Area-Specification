import numpy as np
import cv2
from PIL import Image
import scanpy as sc
import sys
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import warnings
import os
import multiprocessing
from functools import partial
from itertools import repeat
from scipy.sparse import csr_matrix

Image.MAX_IMAGE_PIXELS = None


class DistCalculator():

  def calculate_pts(self,section,image,direction,order):
    self.image = image
    if os.path.exists('increase_dimensions.txt'):
      d = {}
      with open("increase_dimensions.txt") as f:
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


    edges = np.array(Image.open(image))
    edges = edges[0:-1,0:-1]
    mask = binary_fill_holes(edges)
    image1 = image.split('.')[0].split('_')[-1]
    edges_corners = np.array(Image.open('corners/'+image.split('.')[0]+'-corners.png'))

    corners = cv2.goodFeaturesToTrack(edges_corners,4,0.01,10)
    corners = corners.reshape((4,2)).astype('int')
    corners_rc = corners[:,[1,0]]
    corners[:,1] = mask.shape[0]-corners[:,1]
    
    corners1 = corners[corners[:,1].argsort()]
    order = np.array([i for i in order])

    vz_coords = corners1[np.where(order=='v')[0],:]
    mz_coords = corners1[np.where(order=='m')[0],:]


    if direction=='left':
      #corners1 = corners[corners[:,0].argsort()]
      #vz_coords = corners1[0:2,:]
      vz_coords = vz_coords[vz_coords[:,1].argsort()[::-1][:2]]
      #mz_coords = corners1[2:,:]
      mz_coords = mz_coords[mz_coords[:,1].argsort()[::-1][:2]] 

    if direction=='right':
      #corners1 = corners[corners[:,0].argsort()[::-1][:4]]
      #vz_coords = corners1[0:2,:]
      vz_coords = vz_coords[vz_coords[:,1].argsort()]
      #mz_coords = corners1[2:,:]
      mz_coords = mz_coords[mz_coords[:,1].argsort()]

    if direction=='down':
      #corners1 = corners[corners[:,1].argsort()]
      #vz_coords = corners1[0:2,:]
      vz_coords = vz_coords[vz_coords[:,0].argsort()]          
      #mz_coords = corners1[2:,:]
      mz_coords = mz_coords[mz_coords[:,0].argsort()]          
 
    if direction=='up': 
      #corners1 = corners[corners[:,1].argsort()[::-1][:4]]          
      #vz_coords = corners1[0:2,:]
      vz_coords = vz_coords[vz_coords[:,0].argsort()]
      #mz_coords = corners1[2:,:]
      mz_coords = mz_coords[mz_coords[:,0].argsort()]


    #np.save('vz_coords.npy', vz_coords)
    #np.save('mz_coords.npy', mz_coords)

    vz_coord1,vz_coord2 = vz_coords.astype('int')
    mz_coord1,mz_coord2 = mz_coords.astype('int')


    m1 = (mz_coord1[1]-vz_coord1[1])/(mz_coord1[0]-vz_coord1[0])
    m2 = (mz_coord2[1]-vz_coord2[1])/(mz_coord2[0]-vz_coord2[0])

    x = (vz_coord1[1]-vz_coord2[1]-m1*vz_coord1[0]+m2*vz_coord2[0])/(m2-m1)
    y = vz_coord1[1] + m1*(x-vz_coord1[0])
    o = (int(np.round(x)),int(np.round(y)))

    coords = np.where(mask!=0)
    coords = np.stack((coords[0], coords[1]),1)

    adata = sc.read('/project/kyle/xuyu/all/clustering2/merscope_855_no_counts.h5ad')
    sample = section.split('_')[0]
    region = section.split('_')[1]
    adata1 = adata[(adata.obs['sample']==sample) & (adata.obs.region==region)].copy()
    coords1 = adata1.obsm['spatial']-adata1.obsm['spatial'].min(0)
    coords1[:,0] = coords1[:,0]+height/2
    coords1[:,1] = coords1[:,1]+width/2

    dt = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    coords2 = np.in1d(coords1.view(dt).reshape(-1), coords.view(dt).reshape(-1))
    adata2 = adata1[coords2].copy()
    adata2.obsm['spatial'] = coords1[coords2]
    coords = adata2.obsm['spatial']
    self.coords_rc = coords.copy()
 
    dist = np.zeros(coords.shape[0])
    np.save(sample+'_'+region+'_'+ image1+'_index.npy', adata2.obs.index.astype('int'))


    coords[:,0] = mask.shape[0]-coords[:,0]


    m = [(o[1]-i[0])/(o[0]-i[1]) for i in coords]

    vz_coord1_rc = vz_coord1[[1,0]]
    vz_coord1_rc[0] = mask.shape[0]-vz_coord1_rc[0]

    vz_coord2_rc = vz_coord2[[1,0]]
    vz_coord2_rc[0] = mask.shape[0]-vz_coord2_rc[0]


    mz_coord1_rc = mz_coord1[[1,0]]
    mz_coord1_rc[0] = mask.shape[0]-mz_coord1_rc[0]

    mz_coord2_rc = mz_coord2[[1,0]]
    mz_coord2_rc[0] = mask.shape[0]-mz_coord2_rc[0]

    edges1 = edges.copy()
    for i in corners_rc:
      edges1[(i[0]-2):(i[0]+3),(i[1]-2):(i[1]+3)] = 0


    n_comps, comps = cv2.connectedComponents(edges1)
    if n_comps!=5:
      warnings.warn('1: Number of connected components is ' + str(n_comps))                         
      if (n_comps>5 and sum([(comps==i).sum()<5 for i in range(1,n_comps)])>0):
        for i in (np.where([(comps==i).sum()<5 for i in range(1,n_comps)])[0]+1):
          comps[comps==i] = 0                      
        if len(np.unique(comps))>5:
          warnings.warn('2: Number of connected components is ' + str(n_comps))
     
    comps_vz_coord1 = comps[(vz_coord1_rc[0]-10):(vz_coord1_rc[0]+10),(vz_coord1_rc[1]-10):(vz_coord1_rc[1]+10)]
    comps_vz_coord2 = comps[(vz_coord2_rc[0]-10):(vz_coord2_rc[0]+10),(vz_coord2_rc[1]-10):(vz_coord2_rc[1]+10)]
    comps_mz_coord1 = comps[(mz_coord1_rc[0]-10):(mz_coord1_rc[0]+10),(mz_coord1_rc[1]-10):(mz_coord1_rc[1]+10)]
    comps_mz_coord2 = comps[(mz_coord2_rc[0]-10):(mz_coord2_rc[0]+10),(mz_coord2_rc[1]-10):(mz_coord2_rc[1]+10)]

    vz_num = np.intersect1d(comps_vz_coord1[comps_vz_coord1>0], comps_vz_coord2[comps_vz_coord2>0])[0]
    mz_num = np.intersect1d(comps_mz_coord1[comps_mz_coord1>0],	comps_mz_coord2[comps_mz_coord2>0])[0]

    #if direction=='left':                        
    #  vz_num = np.argmin([np.min(np.where(comps==i)[1]) for i in range(1,5)]) + 1
    #  mz_num = np.argmax([np.max(np.where(comps==i)[1]) for i in range(1,5)]) + 1

    #elif direction=='right':                        
    #  vz_num = np.argmax([np.max(np.where(comps==i)[1]) for i in range(1,5)]) + 1
    #  mz_num = np.argmin([np.min(np.where(comps==i)[1]) for i in range(1,5)]) + 1

    #elif direction=='down':                        
    #  vz_num = np.argmax([np.max(np.where(comps==i)[0]) for i in range(1,5)]) + 1
    #  mz_num = np.argmin([np.min(np.where(comps==i)[0]) for i in range(1,5)]) + 1

    #elif direction=='up':                        
    #  vz_num = np.argmin([np.min(np.where(comps==i)[0]) for i in range(1,5)]) + 1
    #  mz_num = np.argmax([np.max(np.where(comps==i)[0]) for i in range(1,5)]) + 1

    vz = comps==vz_num
    mz = comps==mz_num

                          
    for i in (vz_coord1_rc, vz_coord2_rc):
      vz[(i[0]-2):(i[0]+3),(i[1]-2):(i[1]+3)] = edges[(i[0]-2):(i[0]+3),(i[1]-2):(i[1]+3)]


    for i in (mz_coord1_rc, mz_coord2_rc):  
      mz[(i[0]-2):(i[0]+3),(i[1]-2):(i[1]+3)] = edges[(i[0]-2):(i[0]+3),(i[1]-2):(i[1]+3)]

    vz = vz.astype('int')
    mz = mz.astype('int')


    if direction=='left':
      x1 = [mask.shape[1]]*len(m)
      y1 = [o[1]+i*(mask.shape[1]-o[0]) for i in m]

    elif direction=='right':
      x1 = [0]*len(m)
      y1 = [o[1]-i*o[0] for i in m]

    elif direction=='up':
      y1 = [0]*len(m)
      x1 = [-o[1]/i+o[0] for i in m]

    elif direction=='down':
      y1 = [mask.shape[0]]*len(m)
      x1 = [-(o[1]-mask.shape[0])/i+o[0] for i in m]

    x_cr = [(i,mask.shape[0]-j) for i,j in zip(x1,y1)]
    o_cr = (o[0],mask.shape[0]-o[1])
    num_coords = coords.shape[0]
    self.num_coords = num_coords
    self.x_cr = np.array(x_cr)
    self.o_cr = o_cr
    self.vz = csr_matrix(vz.astype(np.uint8))
    self.mz = csr_matrix(mz.astype(np.uint8))
    #return num_coords,x_cr,o_cr,vz,mz

  def calc_dist(self,i):
    img2 = np.zeros(self.vz.shape)
    cv2.line(img2, np.round(self.x_cr[i]).astype('int'),self.o_cr, color = (1), thickness=2)
    A1 = np.where((img2==self.vz)&(img2==1))
    A3 = np.where((img2==self.mz)&(img2==1))
    if len(A1[0])==0:  
      warnings.warn(self.image + ': ' + str(i)+': A1 empty')
#      dist[i] = 0
      return float('nan')
    elif len(A3[0])==0:
      warnings.warn(self.image + ': ' + str(i)+': A3 empty')
#      dist[i] = 0
      return float('nan')
    else:
      A1 = np.array((A1[0][0], A1[1][0]))
      A3 = np.array((A3[0][0], A3[1][0]))
      A1A = np.sum((self.coords_rc[i]-A1)**2)
      A1A3 = np.sum((A3-A1)**2)
#      dist[i] = A1A/A1A3
      return A1A/A1A3


  def generate_mask(annotation_full, border, layer, direction):
    mask1 = annotation_full-border
    n_comps, comps = cv2.connectedComponents(mask1)
    for i in range(n_comps):
      if (comps==i).sum()<50:
        comps[comps==i] = 0

    dict1 = dict(zip(np.unique(comps), range(len(np.unique(comps)))))
    for i in list(dict1.keys())[1:]:
      comps[comps==i] = dict1[i]

    if direction=='down':
      a = np.argsort([np.min(np.where(comps==i)[0]) for i in range(1,7)])
      mz_cp = np.where(a==0)[0][0] + 1
      cp_sp = np.where(a==1)[0][0] + 1
      sp_iz = np.where(a==2)[0][0] + 1
      iz_osvz = np.where(a==3)[0][0] + 1
      osvz_isvz = np.where(a==4)[0][0] + 1
      isvz_vz = np.where(a==5)[0][0] + 1
      layer_dict = dict(zip(['cp', 'sp', 'iz', 'osvz', 'isvz'], [(mz_cp, cp_sp), (cp_sp, sp_iz), (sp_iz, iz_osvz), (iz_osvz, osvz_isvz), (osvz_isvz, isvz_vz)]))
      l0_coords = np.where(comps1==layer_dict[layer][0])
      l0_coords0_cr = (l0_coords[1][np.argmin(l0_coords[1])], l0_coords[0][np.argmin(l0_coords[1])])
      l0_coords1_cr = (l0_coords[1][np.argmax(l0_coords[1])], l0_coords[0][np.argmax(l0_coords[1])])
      l1_coords = np.where(comps1==layer_dict[layer][1])
      l1_coords0_cr = (l1_coords[1][np.argmin(l1_coords[1])], l1_coords[0][np.argmin(l1_coords[1])])
      l1_coords1_cr = (l1_coords[1][np.argmax(l1_coords[1])], l1_coords[0][np.argmax(l1_coords[1])])
      mask = (comps==layer_dict[layer][0]) | (comps==layer_dict[layer][1])
      mask = mask.astype(np.uint8)
      cv2.line(mask, l0_coords0_cr, l1_coords0_cr, color = (1), thickness=1)
      cv2.line(mask, l0_coords1_cr, l1_coords1_cr, color = (1), thickness=1)
      mask = binary_fill_holes(edges)
      mask = mask.astype('bool')

    elif direction=='up':
      a = np.argsort([np.min(np.where(comps==i)[0]) for i in range(1,7)])
      mz_cp = np.where(a==5)[0][0] + 1
      cp_sp = np.where(a==4)[0][0] + 1
      sp_iz = np.where(a==3)[0][0] + 1
      iz_osvz = np.where(a==2)[0][0] + 1
      osvz_isvz = np.where(a==1)[0][0] + 1
      isvz_vz = np.where(a==0)[0][0] + 1
      layer_dict = dict(zip(['cp', 'sp', 'iz', 'osvz', 'isvz'], [(mz_cp, cp_sp), (cp_sp, sp_iz), (sp_iz, iz_osvz), (iz_osvz, osvz_isvz), (osvz_isvz, isvz_vz)]))
      l0_coords = np.where(comps1==layer_dict[layer][0])
      l0_coords0_cr = (l0_coords[1][np.argmin(l0_coords[1])], l0_coords[0][np.argmin(l0_coords[1])])
      l0_coords1_cr = (l0_coords[1][np.argmax(l0_coords[1])], l0_coords[0][np.argmax(l0_coords[1])])
      l1_coords = np.where(comps1==layer_dict[layer][1])
      l1_coords0_cr = (l1_coords[1][np.argmin(l1_coords[1])], l1_coords[0][np.argmin(l1_coords[1])])
      l1_coords1_cr = (l1_coords[1][np.argmax(l1_coords[1])], l1_coords[0][np.argmax(l1_coords[1])])
      mask = (comps==layer_dict[layer][0]) | (comps==layer_dict[layer][1])
      mask = mask.astype(np.uint8)
      cv2.line(mask, l0_coords0_cr, l1_coords0_cr, color = (1), thickness=1)
      cv2.line(mask, l0_coords1_cr, l1_coords1_cr, color = (1), thickness=1)
      mask = binary_fill_holes(edges)
      mask = mask.astype('bool')

    elif direction=='left':
      a = np.argsort([np.min(np.where(comps==i)[1]) for i in range(1,7)])
      mz_cp = np.where(a==5)[0][0] + 1
      cp_sp = np.where(a==4)[0][0] + 1
      sp_iz = np.where(a==3)[0][0] + 1
      iz_osvz = np.where(a==2)[0][0] + 1
      osvz_isvz = np.where(a==1)[0][0] + 1
      isvz_vz = np.where(a==0)[0][0] + 1
      layer_dict = dict(zip(['cp', 'sp', 'iz', 'osvz', 'isvz'], [(mz_cp, cp_sp), (cp_sp, sp_iz), (sp_iz, iz_osvz), (iz_osvz, osvz_isvz), (osvz_isvz, isvz_vz)]))
      l0_coords = np.where(comps1==layer_dict[layer][0])
      l0_coords0_cr = (l0_coords[1][np.argmin(l0_coords[0])], l0_coords[0][np.argmin(l0_coords[0])])
      l0_coords1_cr = (l0_coords[1][np.argmax(l0_coords[0])], l0_coords[0][np.argmax(l0_coords[0])])
      l1_coords = np.where(comps1==layer_dict[layer][1])
      l1_coords0_cr = (l1_coords[1][np.argmin(l1_coords[0])], l1_coords[0][np.argmin(l1_coords[0])])
      l1_coords1_cr = (l1_coords[1][np.argmax(l1_coords[0])], l1_coords[0][np.argmax(l1_coords[0])])
      mask = (comps==layer_dict[layer][0]) | (comps==layer_dict[layer][1])
      mask = mask.astype(np.uint8)
      cv2.line(mask, l0_coords0_cr, l1_coords0_cr, color = (1), thickness=1)
      cv2.line(mask, l0_coords1_cr, l1_coords1_cr, color = (1), thickness=1)
      mask = binary_fill_holes(edges)
      mask = mask.astype('bool')

    elif direction=='right':
      a = np.argsort([np.min(np.where(comps==i)[1]) for i in range(1,7)])
      mz_cp = np.where(a==0)[0][0] + 1
      cp_sp = np.where(a==1)[0][0] + 1
      sp_iz = np.where(a==2)[0][0] + 1
      iz_osvz = np.where(a==3)[0][0] + 1
      osvz_isvz = np.where(a==4)[0][0] + 1
      isvz_vz = np.where(a==5)[0][0] + 1
      layer_dict = dict(zip(['cp', 'sp', 'iz', 'osvz', 'isvz'], [(mz_cp, cp_sp), (cp_sp, sp_iz), (sp_iz, iz_osvz), (iz_osvz, osvz_isvz), (osvz_isvz, isvz_vz)]))
      l0_coords = np.where(comps1==layer_dict[layer][0])
      l0_coords0_cr = (l0_coords[1][np.argmin(l0_coords[0])], l0_coords[0][np.argmin(l0_coords[0])])
      l0_coords1_cr = (l0_coords[1][np.argmax(l0_coords[0])], l0_coords[0][np.argmax(l0_coords[0])])
      l1_coords = np.where(comps1==layer_dict[layer][1])
      l1_coords0_cr = (l1_coords[1][np.argmin(l1_coords[0])], l1_coords[0][np.argmin(l1_coords[0])])
      l1_coords1_cr = (l1_coords[1][np.argmax(l1_coords[0])], l1_coords[0][np.argmax(l1_coords[0])])
      mask = (comps==layer_dict[layer][0]) | (comps==layer_dict[layer][1])
      mask = mask.astype(np.uint8)
      cv2.line(mask, l0_coords0_cr, l1_coords0_cr, color = (1), thickness=1)
      cv2.line(mask, l0_coords1_cr, l1_coords1_cr, color = (1), thickness=1)
      mask = binary_fill_holes(edges) 
      mask = mask.astype('bool')

    return mask

  def cp_line(annotation_full, border, direction):
    mask1 = annotation_full-border
    n_comps, comps = cv2.connectedComponents(mask1)
    for i in range(n_comps):
      if (comps==i).sum()<50:
        comps[comps==i] = 0

    dict1 = dict(zip(np.unique(comps), range(len(np.unique(comps)))))
    for i in list(dict1.keys())[1:]:
      comps[comps==i] = dict1[i]

    if direction=='down':
      a = np.argsort([np.min(np.where(comps==i)[0]) for i in range(1,7)])
      cp_sp = np.where(a==1)[0][0] + 1

    elif direction=='up':
      a = np.argsort([np.min(np.where(comps==i)[0]) for i in range(1,7)])
      cp_sp = np.where(a==4)[0][0] + 1

    elif direction=='left':
      a = np.argsort([np.min(np.where(comps==i)[1]) for i in range(1,7)])
      cp_sp = np.where(a==4)[0][0] + 1

    elif direction=='right':
      a = np.argsort([np.min(np.where(comps==i)[1]) for i in range(1,7)])
      cp_sp = np.where(a==1)[0][0] + 1
      
    mask = comps==cp_sp
    self.cp = mask

  def calc_cp_dist(self,i):
    img2 = np.zeros(self.cp.shape)
    cv2.line(img2, np.round(self.x_cr[i]).astype('int'),self.o_cr, color = (1), thickness=2)
    A2 = np.where((img2==self.cp)&(img2==1))
    A3 = np.where((img2==self.mz)&(img2==1))
    if len(A2[0])==0:
      warnings.warn(self.image + ': ' + str(i)+': A1 empty')
#      dist[i] = 0
      return float('nan')
    elif len(A3[0])==0:
      warnings.warn(self.image + ': ' + str(i)+': A3 empty')
#      dist[i] = 0
      return float('nan')
    else:
      A2 = np.array((A2[0][0], A2[1][0]))
      A3 = np.array((A3[0][0], A3[1][0]))
      A2A = np.sum((self.coords_rc[i]-A2)**2)
      A2A3 = np.sum((A3-A2)**2)
#      dist[i] = A1A/A1A3
      return A2A/A2A3



  #def calc_dist_parallel(self):
  #  pool = multiprocessing.Pool(12)
  #  #dist = pool.starmap(calc_dist,zip(range(self.coords.shape[0]), repeat(self)))
  #  dist = pool.map(partial(calc_dist, dist_calculator=self),range(self.coords.shape[0]))
  #  return dist
    



