import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import scipy
import cmocean

# Directory setup
def setup_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

# Denormalization function
def denorm(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val

# Function to save images and numpy arrays
def save_images_and_arrays(id, imgs, oisst_min, oisst_max, result_dir, prefix=''):
    for key, img in imgs.items():
        plt.imshow(img.squeeze(), vmin=oisst_min, vmax=oisst_max, cmap=cmocean.cm.balance)
        plt.savefig(os.path.join(result_dir, 'png', f'{id:04d}_{prefix}{key}.png'))
        plt.clf()
        np.save(os.path.join(result_dir, 'numpy', f'{id:04d}_{prefix}{key}.npy'), img)


# Data postprocessing
def post_process(img_input, min_val, max_val, land_mask=None, cloud_mask=None):
    if torch.is_tensor(img_input):
        img = img_input.cpu().numpy()
    else:
        img = img_input

    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze()

    img = denorm(img, min_val, max_val)

    if land_mask is not None:
        img = np.where(land_mask == 1, np.nan, img)
    if cloud_mask is not None:
        img = np.where(cloud_mask == 1, np.nan, img)
    return img

# Save visualizations
def save_visualization(sampled_steps, timesteps, meta_data, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    min_val, max_val = meta_data['min'], meta_data['max']
    land_mask = meta_data['land_mask']

    for idx, (img_step, step_name) in enumerate(zip(sampled_steps, timesteps)):
        processed_img = post_process(img_step, min_val, max_val, land_mask)

        ax = axes[idx]
        im = ax.imshow(processed_img, vmin=min_val, vmax=max_val, cmap = cmocean.cm.balance)
        ax.set_title(f'Timestep {step_name}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)

# Plot visualizations
def draw_map(ax, image, a, b, bbox, vmin, vmax, title):
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
    ss = m.pcolormesh(x, y, image, vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance)
    cbar = m.colorbar(ss, extendfrac='auto', spacing='uniform', location='bottom', pad="10%")
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=12)
            
def save(ckpt_dir, netG, optG, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 
                'optG': optG.state_dict(),},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

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

# Image blending functions
def _spline_window(window_size, power=2):
    """
    Create a spline window.
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.windows.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.windows.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind[:int(window_size / 2)]/2

def blend_two_images(img1, img2, overlap_width, axis=1):
    """
    axis=1 for horizontal blending, axis=0 for vertical blending
    """
    if axis == 1:  # Horizontal blending
        overlap1 = img1[:, -overlap_width:]  
        overlap2 = img2[:, :overlap_width] 

        alpha = _spline_window(overlap_width*2)

        blended_overlap = overlap1 * (1 - alpha) + overlap2 * alpha
        stitched_image = np.hstack((img1[:, :-overlap_width], blended_overlap, img2[:, overlap_width:]))
    else:  # Vertical blending
        overlap1 = img1[-overlap_width:, :]  
        overlap2 = img2[:overlap_width, :]  

        alpha = _spline_window(overlap_width*2)
        
        blended_overlap = overlap1 * (1 - alpha)[:, None] + overlap2 * alpha[:, None]
        stitched_image = np.vstack((img1[:-overlap_width, :], blended_overlap, img2[overlap_width:, :]))
    return stitched_image

def load_and_blend_images(image_files, grid_size, overlap):

    images = [np.load(file).squeeze() for file in image_files]
    
    rows, cols = grid_size
    assert len(images) == rows * cols
    

    if images[0].ndim == 3:
        color = True
    else:
        color = False
    
    blended_rows = []
    for i in range(rows):
        row_start_idx = i * cols
        row_images = images[row_start_idx:row_start_idx + cols]
        blended_row = row_images[0]
        for img in row_images[1:]:
            blended_row = blend_two_images(blended_row, img, overlap_width=overlap, axis=1)
        blended_rows.append(blended_row)

    final_image = blended_rows[0]
    for row_img in blended_rows[1:]:
        final_image = blend_two_images(final_image, row_img, overlap_width=overlap, axis=0)
    
    return final_image