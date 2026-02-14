
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import os
import os.path
import glob
from torchvision import transforms
import random
import numpy as np
import torch.nn.functional as F
import math
from PIL import Image
from torch.utils.data import DataLoader
import cv2


class ToTensor(object):
    def __call__(self, data):

        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data
    
def loader(path):
    path_data = np.float32(np.load(path))

    return path_data

class Dataset(data.Dataset):
    def __init__(self, viirs_dir=None, oisst_dir= None,  transform=[], mask_type = 'random_mask', mask_file = None):
        self.viirs_files = self.load_file_list(viirs_dir)
        self.oisst_files = self.load_file_list(oisst_dir)

        if len(self.viirs_files) == 0:
            raise(RuntimeError("Found 0 images in the input files " + "\n"))
        
        self.transform = transform    
        self.mask_type = mask_type
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.viirs_files[index])
            item = self.load_item(0)
        return item

    def __len__(self):
        return len(self.viirs_files)  


    def load_file_list(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist =  list(glob.glob(flist + '/*.npy'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []       



    def load_item(self, index):
        data = {}
                     
        if self.mask_type == 'train':
            viirs_path = self.viirs_files[index]
            oisst_path = self.oisst_files[index]
            
            viirs_name = viirs_path[-1].split("/")[-1]
            oisst_name = oisst_path[-1].split("/")[-1]

            if viirs_name[0:14] != oisst_name[0:14]:
                print('date is different!')
            
            viirs_sst = loader(viirs_path).squeeze()
            land_mask = np.where(viirs_sst==-999, 1,0)

            viirs_sst = np.where(viirs_sst<=0, 0,viirs_sst)

            h, w = viirs_sst.shape
            oisst_sst = loader(oisst_path).squeeze()
            oisst_sst = np.where(oisst_sst<=0, 0,oisst_sst)
            
            mask_image = np.zeros((300,300))
            
            nan_oisst = np.where(oisst_sst<=0, np.nan, oisst_sst)
            oisst_min = np.nanmin(nan_oisst) 
            oisst_max = np.nanmax(nan_oisst) 
         
            viirs_cloud_mask  = np.where(viirs_sst<=0, 1, 0)
            oisst_cloud_mask  = np.where(oisst_sst<=0, 1, 0)

            viirs_sst = ((viirs_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            oisst_sst = ((oisst_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            
                      
            oisst_sst = np.where(oisst_cloud_mask==1, np.nan, oisst_sst)
            oisst_sst = cv2.resize(oisst_sst, (h, w), interpolation = cv2.INTER_LINEAR)
            
            oisst_sst = np.nan_to_num(oisst_sst)
            oisst_cloud_mask = np.where(oisst_sst==0,1,0)
            cloud_mask = np.where(viirs_cloud_mask==1,1,oisst_cloud_mask)
            
            land_mask = np.where(oisst_cloud_mask==1,1,land_mask)
            cloud_mask = np.where(land_mask==1,0,cloud_mask)

            viirs_sst = np.where(viirs_cloud_mask==1, 0, viirs_sst)
            viirs_sst = np.where(oisst_cloud_mask==1, 0, viirs_sst)

            oisst_sst = np.where(land_mask==1, 0, oisst_sst)
            
            viirs_sst = np.where(cloud_mask==1, 0, viirs_sst)
            oisst_sst = np.where(cloud_mask==1, 0, oisst_sst)

            if viirs_sst.ndim == 2:
                viirs_sst = viirs_sst[:, :, np.newaxis]
            if oisst_sst.ndim == 2:
                oisst_sst = oisst_sst[:, :, np.newaxis]
            if mask_image.ndim == 2:
                mask_image = mask_image[:, :, np.newaxis]
        

            data = {'gt_image': viirs_sst,  'structure_image': oisst_sst}
            data['inpaint_map'] = np.where(cloud_mask[:,:,None] == 1, 1, mask_image)
            data['gt_cloud_mask'] = cloud_mask[:,:,None]
            data['land_mask'] = land_mask[:,:,None]

            if self.transform:
                data = self.transform(data)
            
            data = self.to_tensor(data)

            data['oisst_min'] = torch.from_numpy(np.asarray(oisst_min))
            data['oisst_max'] = torch.from_numpy(np.asarray(oisst_max))

        elif self.mask_type == 'mask_reconstruction':
            
            viirs_path = self.viirs_files[index]
            oisst_path = self.oisst_files[index]
            
            viirs_name = viirs_path[-1].split("/")[-1]
            oisst_name = oisst_path[-1].split("/")[-1]

            if viirs_name[0:14] != oisst_name[0:14]:
                print('date is different!')
            
            viirs_sst = loader(viirs_path).squeeze()
            land_mask = np.where(viirs_sst==-999, 1,0)

            viirs_sst = np.where(viirs_sst<=0, 0,viirs_sst)

            h, w = viirs_sst.shape
            oisst_sst = loader(oisst_path).squeeze()
            oisst_sst = np.where(oisst_sst<=0, 0,oisst_sst)
            
            mask_image = np.zeros((300,300))
            mask_image[75:-75,75:-75] = 1
            
            nan_oisst = np.where(oisst_sst<=0, np.nan, oisst_sst)
            oisst_min = np.nanmin(nan_oisst) 
            oisst_max = np.nanmax(nan_oisst) 
         
            viirs_cloud_mask  = np.where(viirs_sst<=0, 1, 0)
            oisst_cloud_mask  = np.where(oisst_sst<=0, 1, 0)

            viirs_sst = ((viirs_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            oisst_sst = ((oisst_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            
                      
            oisst_sst = np.where(oisst_cloud_mask==1, np.nan, oisst_sst)
            oisst_sst = cv2.resize(oisst_sst, (h, w), interpolation = cv2.INTER_LINEAR)
            
            oisst_sst = np.nan_to_num(oisst_sst)
            oisst_cloud_mask = np.where(oisst_sst==0,1,0)
            cloud_mask = np.where(viirs_cloud_mask==1,1,oisst_cloud_mask)
            
            land_mask = np.where(oisst_cloud_mask==1,1,land_mask)
            cloud_mask = np.where(land_mask==1,0,cloud_mask)

            viirs_sst = np.where(viirs_cloud_mask==1, 0, viirs_sst)
            viirs_sst = np.where(oisst_cloud_mask==1, 0, viirs_sst)

            oisst_sst = np.where(land_mask==1, 0, oisst_sst)


            if viirs_sst.ndim == 2:
                viirs_sst = viirs_sst[:, :, np.newaxis]
            if oisst_sst.ndim == 2:
                oisst_sst = oisst_sst[:, :, np.newaxis]
            if mask_image.ndim == 2:
                mask_image = mask_image[:, :, np.newaxis]
        

            data = {'gt_image': viirs_sst,  'structure_image': oisst_sst}
            data['inpaint_map'] = np.where(cloud_mask[:,:,None] == 1, 1, mask_image)
            data['gt_cloud_mask'] = cloud_mask[:,:,None]
            data['land_mask'] = land_mask[:,:,None]

            if self.transform:
                data = self.transform(data)
            
            data = self.to_tensor(data)

            data['oisst_min'] = torch.from_numpy(np.asarray(oisst_min))
            data['oisst_max'] = torch.from_numpy(np.asarray(oisst_max))
            
        elif self.mask_type == 'cloud_reconstruction':
            
            viirs_path = self.viirs_files[index]
            oisst_path = self.oisst_files[index]
            
            viirs_name = viirs_path[-1].split("/")[-1]
            oisst_name = oisst_path[-1].split("/")[-1]

            if viirs_name[0:14] != oisst_name[0:14]:
                print('date is different!')
            
            viirs_sst = loader(viirs_path).squeeze()
            land_mask = np.where(viirs_sst==-999, 1,0)

            viirs_sst = np.where(viirs_sst<=0, 0,viirs_sst)

            h, w = viirs_sst.shape
            oisst_sst = loader(oisst_path).squeeze()
            oisst_sst = np.where(oisst_sst<=0, 0,oisst_sst)
            
            mask_image = np.zeros((300,300))
            
            nan_oisst = np.where(oisst_sst<=0, np.nan, oisst_sst)
            oisst_min = np.nanmin(nan_oisst) 
            oisst_max = np.nanmax(nan_oisst) 
         
            viirs_cloud_mask  = np.where(viirs_sst<=0, 1, 0)
            oisst_cloud_mask  = np.where(oisst_sst<=0, 1, 0)

            viirs_sst = ((viirs_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            oisst_sst = ((oisst_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            
                      
            oisst_sst = np.where(oisst_cloud_mask==1, np.nan, oisst_sst)

            oisst_sst = np.nan_to_num(oisst_sst)
            cloud_mask = np.where(viirs_cloud_mask==1,1,oisst_cloud_mask)
            
            land_mask = np.where(oisst_cloud_mask==1,1,land_mask)
            cloud_mask = np.where(land_mask==1,0,cloud_mask)

            viirs_sst = np.where(viirs_cloud_mask==1, 0, viirs_sst)
            viirs_sst = np.where(oisst_cloud_mask==1, 0, viirs_sst)

            oisst_sst = np.where(land_mask==1, 0, oisst_sst)


            if viirs_sst.ndim == 2:
                viirs_sst = viirs_sst[:, :, np.newaxis]
            if oisst_sst.ndim == 2:
                oisst_sst = oisst_sst[:, :, np.newaxis]
            if mask_image.ndim == 2:
                mask_image = mask_image[:, :, np.newaxis]
        

            data = {'gt_image': viirs_sst,  'structure_image': oisst_sst}
            data['inpaint_map'] = np.where(cloud_mask[:,:,None] == 1, 1, mask_image)
            data['gt_cloud_mask'] = cloud_mask[:,:,None]
            data['land_mask'] = land_mask[:,:,None]

            if self.transform:
                data = self.transform(data)
            
            data = self.to_tensor(data)

            data['oisst_min'] = torch.from_numpy(np.asarray(oisst_min))
            data['oisst_max'] = torch.from_numpy(np.asarray(oisst_max))


        elif self.mask_type == 'test_time_series':
            
            viirs_path = self.viirs_files[index]
            oisst_path = self.oisst_files[index]

            viirs_name = viirs_path[-1].split("/")[-1]
            oisst_name = oisst_path[-1].split("/")[-1]

            if viirs_name[0:14] != oisst_name[0:14]:
                print('date is different!')
            

            viirs_sst_files = loader(viirs_path).squeeze()
            viirs_sst = viirs_sst_files[:,:,0]
            viirs_sst_past = viirs_sst_files[:,:,1]
            
            land_mask = np.where(viirs_sst==-999, 1,0)

            viirs_sst = np.where(viirs_sst<=0, 0,viirs_sst)

            viirs_sst_past = np.where(viirs_sst_past<=0, 0,viirs_sst_past)

            h, w = viirs_sst.shape
            oisst_sst = loader(oisst_path).squeeze()
            oisst_sst = np.where(oisst_sst<=0, 0,oisst_sst)
            
            crop_size = 300

            
            mask_image = np.zeros((300,300))
            mask_image[75:-75,75:-75] = 1      
               
            mask_image2 = np.zeros((300,300))
            mask_image2[75:-75,75:-75] = 1      
                                    
            nan_oisst = np.where(oisst_sst<=0, np.nan, oisst_sst)
            oisst_min = np.nanmin(nan_oisst) 
            oisst_max = np.nanmax(nan_oisst) 
         
            viirs_cloud_mask  = np.where(viirs_sst<=0, 1, 0)
            viirs_cloud_mask_past  = np.where(viirs_sst_past<=0, 1, 0)
            oisst_cloud_mask  = np.where(oisst_sst<=0, 1, 0)

            
            viirs_sst = ((viirs_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            viirs_sst_past = ((viirs_sst_past-oisst_min)/(oisst_max - oisst_min))*2 -1
            oisst_sst = ((oisst_sst-oisst_min)/(oisst_max - oisst_min))*2 -1
            
                       
            oisst_sst = np.where(oisst_cloud_mask==1, np.nan, oisst_sst)
            oisst_sst = cv2.resize(oisst_sst, (h, w), interpolation = cv2.INTER_LINEAR)
            
            oisst_sst = np.nan_to_num(oisst_sst)
            oisst_cloud_mask = np.where(oisst_sst==0,1,0)
            cloud_mask = np.where(viirs_cloud_mask==1,1,oisst_cloud_mask)
            

            land_mask = np.where(oisst_cloud_mask==1,1,land_mask)
            cloud_mask = np.where(land_mask==1,0,cloud_mask)

            viirs_sst = np.where(viirs_cloud_mask==1, 0, viirs_sst)
            viirs_sst = np.where(oisst_cloud_mask==1, 0, viirs_sst)
            
            viirs_cloud_mask_past = (1-viirs_cloud_mask_past)*mask_image2
            viirs_cloud_mask_past = np.where(oisst_cloud_mask==1,0,viirs_cloud_mask_past)

            viirs_sst_past = np.where(viirs_cloud_mask_past==0, 0, viirs_sst_past)

            viirs_sst_past = np.where(oisst_cloud_mask==1, 0, viirs_sst_past)

            oisst_sst = np.where(land_mask==1, 0, oisst_sst)


            if viirs_sst.ndim == 2:
                viirs_sst = viirs_sst[:, :, np.newaxis]
            if oisst_sst.ndim == 2:
                oisst_sst = oisst_sst[:, :, np.newaxis]
            if viirs_sst_past.ndim == 2:
                viirs_sst_past = viirs_sst_past[:, :, np.newaxis]
            if mask_image.ndim == 2:
                mask_image = mask_image[:, :, np.newaxis]
        

            data = {'gt_image': viirs_sst, 'structure_image': oisst_sst, 
                    'context': np.concatenate([viirs_sst_past, viirs_cloud_mask_past[:,:,None]], axis=2)}
            data['inpaint_map'] = np.where(cloud_mask[:,:,None] == 1, 1, mask_image)
            data['gt_cloud_mask'] = cloud_mask[:,:,None]
            data['land_mask'] = land_mask[:,:,None]

            if self.transform:
                data = self.transform(data)
        
            data = self.to_tensor(data)

            
            data['oisst_min'] = torch.from_numpy(np.asarray(oisst_min))
            data['oisst_max'] = torch.from_numpy(np.asarray(oisst_max))       

        return data


