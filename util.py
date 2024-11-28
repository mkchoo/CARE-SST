import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


def draw_gradient_map(ax, image, a, b, bbox, vmin, vmax, title):
    m = Basemap(ax=ax, llcrnrlon=bbox[0], llcrnrlat=bbox[1], urcrnrlon=bbox[2], urcrnrlat=bbox[3], area_thresh=0.001, resolution='l')
    x, y = m(a, b)
    label_font_size = 10

    parallels = np.arange(int(round(bbox[1])), int(round(bbox[3])) + 0.1, 3.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.01, color='0.5', fontsize=label_font_size)
    meridians = np.arange(int(round(bbox[0])), int(round(bbox[2])) + 0.1, 3.)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.01, color='0.5', fontsize=label_font_size)

    m.fillcontinents(color='grey', lake_color='grey')
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.drawlsmask(lsmask=1, resolution='h', land_color='grey')
    ss = m.pcolormesh(x, y, image, vmin=vmin, vmax=vmax)
    cbar = m.colorbar(ss, extendfrac='auto', spacing='uniform', location='bottom', pad="10%")
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=12)
            
## 네트워크 저장하기
def save(ckpt_dir, netG, optG, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 
                'optG': optG.state_dict(),},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, netG, optG):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, optG, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    optG.load_state_dict(dict_model['optG'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, optG, epoch

# Directory setup
def setup_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            
def denorm(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val

# Function to save images and numpy arrays
def save_images_and_arrays(id, imgs, oisst_min, oisst_max, prefix=''):
    for key, img in imgs.items():
        plt.imshow(img.squeeze(), vmin=oisst_min, vmax=oisst_max)
        plt.savefig(os.path.join(result_dir, 'png', f'{id:04d}_{prefix}{key}.png'))
        plt.clf()
        np.save(os.path.join(result_dir, 'numpy', f'{id:04d}_{prefix}{key}.npy'), img)