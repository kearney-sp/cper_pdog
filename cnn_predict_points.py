from shapely.geometry import Polygon, Point
import itertools
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
from rasterio import mask
from rasterio.plot import show
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import torch.nn as nn
import numpy as np
import shapely.wkt as wkt
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from tqdm import tqdm
from skimage.measure import label, regionprops
import os
import dask
import time
import gc
import re
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays
from scipy.interpolate import NearestNDInterpolator
import distributed
import ctypes

import xarray as xr
import rioxarray as riox
from xrspatial import convolution, focal, hillshade
from skimage.transform import resize
from dask.distributed import LocalCluster, Client

from collections import namedtuple
from operator import mul

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

# create function to normalize all data in range 0-1
def normalize_fn(image, image_suffix, stats_dict):
    if image_suffix in stats_dict.keys():
        min_tmp = stats_dict[image_suffix]['min']
        max_tmp = stats_dict[image_suffix]['max']
    else:
        # normalize to individual image if min/max stats not specified in dictionary
        min_tmp = np.min(image)
        max_tmp = np.max(image)
    return (image - min_tmp) / (max_tmp - min_tmp)

def calc_tpi(dtm, inner_r, outer_r, interpolate=True, values=True, bounds=(-2.0, 2.0)):
    cellsize_x, cellsize_y = convolution.calc_cellsize(dtm)
    kernel = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_r, inner_r)
    tpi = dtm - focal.apply(dtm, kernel)
    tpi = tpi.rio.write_nodata(-9999.)
    tpi = tpi.where((tpi > bounds[0]) & (tpi < bounds[1]))
    if interpolate:
        if tpi.isnull().any().values:
            tpi = tpi.rio.interpolate_na(method='nearest')
    if values:
        return tpi.values
    else:
        return tpi

def calc_ndvi(ms, interpolate=True, values=True):
    ndvi = (ms.sel(band=4).astype('float32') - ms.sel(band=3).astype('float32'))\
            / (ms.sel(band=4).astype('float32') + ms.sel(band=3).astype('float32'))
    ndvi = ndvi.rio.write_nodata(-9999.)
    ndvi = ndvi.where(ndvi != -9999.)
    if interpolate:
        if ndvi.isnull().any().values:
            ndvi = ndvi.rio.interpolate_na(method='nearest')
    if values:
        return ndvi.values
    else:
        return ndvi


if __name__ == '__main__':
    dask.config.set({"distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0})
    dask.config.set(scheduler='processes')

    os.environ["MALLOC_TRIM_THRESHOLD_"] = str(dask.config.get("distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_"))

    cluster = LocalCluster(n_workers=8, threads_per_worker=2, processes=True)
    client = Client(cluster)
    client.amm.start()


    outDIR = './cnn_pred_results/'
    if not os.path.exists(outDIR):
        os.mkdir(outDIR)

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['burrow']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda' #'cuda'# 'cpu'# 
    model_fnl = 'deeplabplus'
    res_fnl = 5
    inputs_fnl = ['rgb', 'tpi', 'ndvi'] 
    preprocess = True
    prob_thresh = 0.5

    size_dict = {
        2: {'tile_size': 256,
            'buff_size': 64},
        5: {'tile_size': 192,
            'buff_size': 48},
        10: {'tile_size': 128,
             'buff_size': 32},
        15: {'tile_size': 96,
             'buff_size': 16},
        30: {'tile_size': 64,
             'buff_size': 16}
    }

    #past_subset = None
    past_subset = ['22E', 'CN']

    img_f_dict = {
        '5W': {
            'rgb': ['/mnt/d/202109/outputs/202109_5W_RGB/CPER_202109_5W_RGB_ortho.tif'],
            'ms': ['/mnt/d/202109/outputs/202109_5W_MS/CPER_202109_5W_MS_ortho.tif'],
            'dsm': ['/mnt/d/202109/outputs/202109_5W_RGB/CPER_202109_5W_RGB_dsm.tif']
        },
        '29-30': {
            'rgb': ['/mnt/d/202109/outputs/202109_29_30_RGB/CPER_202109_29_30_RGB_ortho.tif',
                    '/mnt/d/202109/outputs/202109_29_30_RGB/CPER_202109_29_30_RGB_ortho.tif'],
            'ms': ['/mnt/d/202109/outputs/202109_29_30_MS/CPER_202109_29_30_North_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_29_30_MS/CPER_202109_29_30_South_MS_ortho.tif'],
            'dsm': ['/mnt/d/202109/outputs/202109_29_30_RGB/CPER_202109_29_30_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_29_30_RGB/CPER_202109_29_30_RGB_DSM.tif']
        },
        '22W': {
            'rgb': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_RGB_ortho.tif'],
            'ms': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_MS_ortho.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_MS_ortho.tif'],
            'dsm': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_RGB_DSM.tif',
                    '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_RGB_DSM.tif']
        },
        '22E': {
            'rgb': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight3_RGB_ortho.tif'],
            'ms': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_MS_ortho.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_MS_ortho.tif'],
            'dsm': ['/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight1_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight2_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_22EW/CPER_202109_22EW_Flight3_RGB_DSM.tif']
        },
        'CN': {
            'rgb': ['/mnt/d/202109/outputs/202109_CN_RGB/Orthos/CPER_CN_Flight2_202109_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/Orthos/CPER_CN_Flight3_202109_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/Orthos/CPER_CN_Flight4_202109_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/Orthos/CPER_CN_Flight5_202109_RGB_ortho.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/Orthos/CPER_CN_Flight5_202109_RGB_ortho.tif'],
            'ms': ['/mnt/d/202109/outputs/202109_CN_MS/CPER_202109_CN_Flight2_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_CN_MS/CPER_202109_CN_Flight2_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_CN_MS/CPER_202109_CN_Flight3_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_CN_MS/CPER_202109_CN_Flight3_MS_ortho.tif',
                  '/mnt/d/202109/outputs/202109_CN_MS/CPER_202109_CN_Flight4_MS_ortho.tif',],
            'dsm': ['/mnt/d/202109/outputs/202109_CN_RGB/DSMs/CPER_CN_Flight2_202109_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/DSMs/CPER_CN_Flight3_202109_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/DSMs/CPER_CN_Flight4_202109_RGB_DSM.tif',
                   '/mnt/d/202109/outputs/202109_CN_RGB/DSMs/CPER_CN_Flight5_202109_RGB_DSM.tif',
                    '/mnt/d/202109/outputs/202109_CN_RGB/DSMs/CPER_CN_Flight5_202109_RGB_DSM.tif']
        }
    }

    if past_subset is not None:
        img_f_dict_tmp = img_f_dict.copy()
        img_f_dict = {}
        for k in img_f_dict_tmp:
             if k in past_subset:
                    img_f_dict[k] = img_f_dict_tmp[k]

    cper_f = '/mnt/c/Users/TBGPEA-Sean/Desktop/Pdogs_UAS/cper_pdog_pastures_2017_clip.shp'

    full_buff_size = 10
    full_tile_size = 150
    tile_size = size_dict[res_fnl]['tile_size']
    buff_size = size_dict[res_fnl]['buff_size']
    chunk_size = 250
    buff_size_m = np.ceil(buff_size * res_fnl * 0.01)

    # load best saved checkpoint
    if res_fnl == 2:
        best_model = torch.load('./cnn_results_' + model_fnl + '_' + str(res_fnl) + 'cm/best_model_' + '_'.join(inputs_fnl) + '.pth')
    else:
        best_model = torch.load('./cnn_results_' + model_fnl + '_' + str(res_fnl) + 'cm/best_model_' + '_'.join(inputs_fnl) + '_' + str(res_fnl) + 'cm.pth')

    if DEVICE == 'cpu':
        best_model = best_model.cpu()
    best_model.eval()

    # load the image stats from the training data
    df_image_stats = pd.read_csv('./_utils/image_stats_2cm.csv').set_index('stat')

    # convert image stats dictionary to dataframe
    image_stats = {i: {'min': df_image_stats.loc['min', i],
                       'max': df_image_stats.loc['max', i]} for i in df_image_stats.columns}

    cper_gdf = gpd.read_file(cper_f)

    for pasture in tqdm(img_f_dict):
        print('\n\n----------\nPasture: ' + pasture)

        # get the bounding box of the pasture
        past_bbox = cper_gdf[cper_gdf['Past_Name_'] == pasture].buffer(
            full_buff_size).bounds.apply(lambda x: int(x))

        total_bounds = {'xmin': past_bbox['minx'],
                        'xmax': past_bbox['maxx'],
                        'ymin': past_bbox['miny'],
                        'ymax': past_bbox['maxy']}

        n_row_tiles = int(np.ceil((total_bounds['ymax'] - total_bounds['ymin'])/full_tile_size))
        n_col_tiles = int(np.ceil((total_bounds['xmax'] - total_bounds['xmin'])/full_tile_size))

        outSHP = os.path.join(outDIR, 'burrow_pts_pred_' + '_'.join([pasture] + inputs_fnl + [str(res_fnl)]) + 'cm.shp')
        if os.path.exists(outSHP):
            gdf_out = gpd.read_file(outSHP)
            r_ct_pred = len(gdf_out)
            rc_completed = gdf_out.apply(lambda x: '_'.join([str(x.tile_row), str(x.tile_col)]), axis=1).unique()
        elif os.path.exists(re.sub('.shp', '.csv', outSHP)):
            gdf_out = pd.read_csv(re.sub('.shp', '.csv', outSHP))
            r_ct_pred = len(gdf_out)
            rc_completed = gdf_out.apply(lambda x: '_'.join([str(x.tile_row), str(x.tile_col)]), axis=1).unique()
        else:
            r_ct_pred = 0
            gdf_out = gpd.GeoDataFrame()
            rc_completed = []
        tile_ct = 0
        for full_r in range(n_row_tiles):
            print('running row: ' + str(full_r + 1) + ' of ' + str(n_row_tiles))
            for full_c in tqdm(range(n_col_tiles)):
                if len(client.cluster.workers) < 8:
                    client.shutdown()
                    client.close()
                    cluster = LocalCluster(n_workers=8, threads_per_worker=2, processes=True)
                    client = Client(cluster)
                    client.amm.start()
                if '_'.join([str(full_r), str(full_c)]) in rc_completed:
                    #print('skipping row/column combination, already in shapefile!')
                    continue
                else:
                    try:
                        t0=time.time()

                        ll_tile = [full_c * full_tile_size + total_bounds['xmin'],
                                   full_r * full_tile_size + total_bounds['ymin']]
                        ul_tile = [ll_tile[0], ll_tile[1] + full_tile_size]
                        ur_tile = [x + full_tile_size for x in ll_tile]
                        lr_tile = [ll_tile[0] + full_tile_size, ll_tile[1]]
                        tile_poly = Polygon([ll_tile, ul_tile, ur_tile, lr_tile])

                        ll = [full_c * full_tile_size + total_bounds['xmin'] - buff_size_m,
                              full_r * full_tile_size + total_bounds['ymin'] - buff_size_m]
                        ul = [ll[0], ll[1] + full_tile_size + (buff_size_m * 2.0)]
                        ur = [x + full_tile_size + (buff_size_m * 2.0) for x in ll]
                        lr = [ll[0] + full_tile_size + (buff_size_m * 2.0), ll[1]]

                        image_dict = {}
                        newsize_r = int(round((ul[1] - ll[1]) / (res_fnl * 0.01), 0))
                        newsize_c = int(round((lr[0] - ll[0]) / (res_fnl * 0.01), 0))
                        if 'rgb' in inputs_fnl:
                            print('getting RGB')
                            t1=time.time()

                            rgb_xr_list = []
                            for rgb_f in img_f_dict[pasture]['rgb']:
                                with riox.open_rasterio(rgb_f, masked=True) as rgb_src:
                                    rgb_xr_list.append(rgb_src.sel(band=slice(0, 3),
                                                                   x=slice(ll[0], lr[0]),
                                                                   y=slice(ul[1], ll[1]), 
                                                                   drop=True))
                            rgb_xr_list = [x.where(x != 255) for x in rgb_xr_list if not any([s == 0 for s in x.shape])]
                            rgb_xr_list = [x.rio.write_nodata(-9999.).chunk({'band': -1,
                                                                           'x': chunk_size,
                                                                           'y': chunk_size}) for x in rgb_xr_list]

                            rgb_xr = merge_arrays(rgb_xr_list,
                                                  bounds=(ll[0], ll[1], lr[0], ur[1]), 
                                                  res=res_fnl*0.01, 
                                                  crs=rio.CRS.from_epsg(32613),
                                                  method='max',
                                                  nodata=-9999.)
                            rgb_xr = rgb_xr.where(rgb_xr != -9999., drop=True).chunk({'band': -1,
                                                                           'x': chunk_size,
                                                                           'y': chunk_size})
                            if rgb_xr.isnull().any().values:
                                rgb_xr = rgb_xr.rio.interpolate_na(method='nearest')

                            image_dict['rgb'] = rgb_xr.values
                            #rgb_xr.close()
                            t2=time.time()
                            print('... completed in', round(t2 - t1, 1), 'secs')
                            del rgb_xr, rgb_xr_list
                        if 'dsm' in inputs_fnl or 'tpi' in inputs_fnl:
                            t1 = time.time()
                            print('getting DSM')
                            dsm_xr_list = []
                            for dsm_f in img_f_dict[pasture]['dsm']:
                                with riox.open_rasterio(dsm_f, masked=True) as dsm_src:
                                    dsm_xr_list.append(dsm_src.sel(x=slice(ll[0], lr[0]),
                                                                   y=slice(ul[1], ll[1]), 
                                                                   drop=True))

                            dsm_xr_list = [x.where(x > 0, drop=True) for x in dsm_xr_list]
                            dsm_xr_list = [x.squeeze() for x in dsm_xr_list if not any([s == 0 for s in x.shape])]
                            dsm_xr_list = [x.rio.write_nodata(-9999.).chunk({'x': chunk_size,
                                                                             'y': chunk_size}) for x in dsm_xr_list]

                            dsm_xr = merge_arrays(dsm_xr_list,
                                                  bounds=(ll[0], ll[1], lr[0], ur[1]), 
                                                  res=res_fnl*0.01, 
                                                  crs=rio.CRS.from_epsg(32613),
                                                  method='max',
                                                  nodata=-9999.)
                            dsm_xr = dsm_xr.where(dsm_xr > 0)
                            dsm_xr = dsm_xr.where(dsm_xr != -9999.)
                            
                            if dsm_xr.isnull().any().values:
                                dsm_xr = dsm_xr.rio.interpolate_na(method='nearest')
                            
                            if 'dsm' in inputs_fnl:
                                image_dict['dsm'] = dsm_xr.values
                            #dsm_xr.close()
                            t2=time.time()
                            print('... completed in', round(t2 - t1, 1), 'secs')
                        if 'tpi' in inputs_fnl: 
                            t1 = time.time()
                            print('computing TPI')
                            # prepare an annulus kernel with a ring at a distance from 5-10 cells away from focal point
                            outer_radius = "0.75m"
                            inner_radius = "0.25m"
                            image_dict['tpi'] = calc_tpi(dsm_xr.chunk({'x': chunk_size,
                                                                       'y': chunk_size}), 
                                                         inner_r=inner_radius, 
                                                         outer_r=outer_radius, 
                                                         interpolate=True,
                                                         values=True)
                            #dsm_xr.close()
                            t2=time.time()
                            print('... completed in', round(t2 - t1, 1), 'secs')
                            del dsm_xr, dsm_xr_list
                        if 'ndvi' in inputs_fnl:
                            t1 = time.time()
                            print('computing NDVI')
                            ms_xr_list = []
                            for ms_f in img_f_dict[pasture]['ms']:
                                with riox.open_rasterio(ms_f, masked=True) as ms_src:
                                    ms_xr_list.append(ms_src.sel(band=[4, 3],
                                                                   x=slice(ll[0], lr[0]),
                                                                   y=slice(ul[1], ll[1]), 
                                                                   drop=True))

                            #ms_xr_list = [x.where(x != 65535, drop=True) for x in ms_xr_list]
                            ms_xr_list = [x.where(x != 65535, drop=True) for x in ms_xr_list if not any([s == 0 for s in x.shape])]
                            ms_xr_list = [x.rio.write_nodata(-9999.).chunk({'x': chunk_size,
                                                                            'y': chunk_size}) for x in ms_xr_list]

                            ms_xr = merge_arrays(ms_xr_list,
                                                  bounds=(ll[0], ll[1], lr[0], ur[1]), 
                                                  res=res_fnl*0.01, 
                                                  crs=rio.CRS.from_epsg(32613),
                                                  method='max',
                                                  nodata=-9999.)
                            ms_xr = ms_xr.where(ms_xr != -9999.)

                            image_dict['ndvi'] = calc_ndvi(ms_xr, values=True)
                            #ms_xr.close()
                            t2=time.time()
                            print('... completed in', round(t2 - t1, 1), 'secs')
                            del ms_xr, ms_xr_list

                        if 'rgb' in image_dict:
                            tshape = image_dict['rgb'].shape[1:]
                        else:
                            tshape = image_dict[inputs_fnl[0]].shape

                        n_row_chunks = int(np.ceil(tshape[0]/tile_size))
                        n_col_chunks = int(np.ceil(tshape[1]/tile_size))

                        pr_mask = np.empty(tshape)
                        t1 = time.time()
                        print('predicting binary burrow image')
                        for r in range(n_row_chunks):
                            if (r + 1) * tile_size > tshape[0]:
                                r_min = tshape[0] - tile_size
                                r_max = tshape[0]
                                r_max_comp = tshape[0]
                            elif (r + 1) * tile_size + buff_size > tshape[0]:
                                r_min = r * tile_size
                                r_max = (r + 1) * tile_size
                                r_max_comp = r_max
                            else:
                                r_min = r * tile_size
                                r_max = (r + 1) * tile_size
                                r_max_comp = r_max + buff_size
                            for c in range(n_col_chunks):
                                image_sub_dict = {}
                                if (c + 1) * tile_size > tshape[1]:
                                    c_min = tshape[1] - tile_size
                                    c_max = tshape[1]
                                    c_max_comp = tshape[1]
                                elif (c + 1) * tile_size + buff_size > tshape[1]:
                                    c_min = c * tile_size
                                    c_max = (c + 1) * tile_size
                                    c_max_comp = c_max
                                else:
                                    c_min = c * tile_size
                                    c_max = (c + 1) * tile_size
                                    c_max_comp = c_max + buff_size
                                for k in image_dict:
                                    if k == 'rgb':
                                        image_sub_dict[k] = image_dict[k][:,
                                                                          slice(max(0, r_min-buff_size), r_max_comp),
                                                                          slice(max(0, c_min-buff_size), c_max_comp)].astype('float32')
                                    else:
                                        image_sub_dict[k] = image_dict[k][slice(max(0, r_min-buff_size), r_max_comp),
                                                                          slice(max(0, c_min-buff_size), c_max_comp)].astype('float32')
                                    if len(image_sub_dict[k].shape) == 2:
                                        image_sub_dict[k] = np.expand_dims(image_sub_dict[k], 0)
                                    if np.all(np.isnan(image_sub_dict[k])):
                                        continue
                                    elif np.any(np.isnan(image_sub_dict[k])):
                                        for i in range(image_sub_dict[k].shape[0]):
                                            if np.any(np.isnan(image_sub_dict[k][i, :, :])):
                                                data = image_sub_dict[k][i, :, :].copy()
                                                mask = np.where(~np.isnan(data))
                                                interp = NearestNDInterpolator(np.transpose(mask), data[mask])
                                                image_sub_dict[k][i, :, :] = interp(*np.indices(data.shape))
                                                del data, mask, interp
        
                                if np.any([np.all(np.isnan(image_sub_dict[k])) for k in image_sub_dict]):
                                    pr_mask[r_min:r_max, c_min:c_max] = np.nan
                                    continue
                                else:
                                    if preprocess:
                                        for i in image_sub_dict:
                                            image_sub_dict[i] = normalize_fn(image_sub_dict[i], i, image_stats)
                                    image_list = [image_sub_dict[i] for i in inputs_fnl]
                                    image_out = np.concatenate(image_list, axis=0)
                                    x_tensor = torch.from_numpy(image_out).to(DEVICE).unsqueeze(0)
                                    if type(best_model) == nn.DataParallel:
                                        pred_tmp = best_model.module.predict(x_tensor).cpu().detach().numpy().squeeze() >= prob_thresh
                                        buff_r_min = buff_size * int(r_min-buff_size > 0)
                                        buff_r_max = buff_size * int(r_max+buff_size <= tshape[0])
                                        buff_c_min = buff_size * int(c_min-buff_size > 0)
                                        buff_c_max = buff_size * int(c_max+buff_size <= tshape[1])
                                        if pred_tmp.shape[1] > tile_size:
                                            pr_mask[r_min:r_max, c_min:c_max] = pred_tmp[buff_r_min:pred_tmp.shape[0]-buff_r_max,
                                                                                         buff_c_min:pred_tmp.shape[1]-buff_c_max]
                                        else:
                                            pr_mask[r_min:r_max, c_min:c_max] = pred_tmp
                                    else:
                                        pred_tmp = best_model.predict(x_tensor).cpu().detach().numpy().squeeze() >= prob_thresh
                                        buff_r_min = buff_size * int(r_min-buff_size > 0)
                                        buff_r_max = buff_size * int(r_max+buff_size <= tshape[0])
                                        buff_c_min = buff_size * int(c_min-buff_size > 0)
                                        buff_c_max = buff_size * int(c_max+buff_size <= tshape[1])
                                        if pred_tmp.shape[1] > tile_size:
                                            pr_mask[r_min:r_max, c_min:c_max] = pred_tmp[buff_r_min:pred_tmp.shape[0]-buff_r_max,
                                                                                         buff_c_min:pred_tmp.shape[1]-buff_c_max]
                                        else:
                                            pr_mask[r_min:r_max, c_min:c_max] = pred_tmp
                        t2=time.time()
                        print('... completed in', round(t2 - t1, 1), 'secs')
                        t1 = time.time()
                        print('getting burrow locations')
                        if np.all(pr_mask == 0):
                            gdf_tmp = gpd.GeoDataFrame(data=pd.DataFrame({'area': ''}, index=[r_ct_pred]))
                            gdf_tmp['tile_row'] = full_r
                            gdf_tmp['tile_col'] =  full_c
                            gdf_tmp['tile_size'] = full_tile_size
                            gdf_out = pd.concat([gdf_out, gdf_tmp])
                            del gdf_tmp
                            r_ct_pred += 1
                        else:
                            pr_labels = label(pr_mask)
                            pr_regions = regionprops(pr_labels)
                            pr_regions = [r for r in pr_regions if (r.area*(res_fnl/100)**2 > 0.05) & (r.area*(res_fnl/100)**2 < 5.0)]
                            if len(pr_regions) == 0:
                                #print('no burrow locations found!')
                                gdf_tmp = gpd.GeoDataFrame(data=pd.DataFrame({'area': ''}, index=[r_ct_pred]))
                                gdf_tmp['tile_row'] = full_r
                                gdf_tmp['tile_col'] =  full_c
                                gdf_tmp['tile_size'] = full_tile_size
                                gdf_out = pd.concat([gdf_out, gdf_tmp])
                                del gdf_tmp
                                r_ct_pred += 1
                            else:
                                r_ct_tile = 0
                                for r in pr_regions:
                                    gdf_tmp = gpd.GeoDataFrame(data=pd.DataFrame({'area': r.area}, 
                                                                                 index=[r_ct_pred]), 
                                                               geometry=[Point([ll[0] + r.centroid[1]*(res_fnl*0.01),
                                                                                ul[1] - r.centroid[0]*(res_fnl*0.01)])], 
                                                               crs='EPSG:32613')
                                    if gdf_tmp.geometry.within(tile_poly).values[0]:
                                        gdf_tmp['tile_row'] = full_r
                                        gdf_tmp['tile_col'] =  full_c
                                        gdf_tmp['tile_size'] = full_tile_size
                                        gdf_out = pd.concat([gdf_out, gdf_tmp])
                                        r_ct_tile += 1
                                    if type(gdf_out) is pd.core.frame.DataFrame:
                                        gdf_out = gpd.GeoDataFrame(gdf_out, geometry = gdf_out['geometry'])
                                    del gdf_tmp
                                    r_ct_pred += 1
                                if r_ct_tile == 0:
                                    gdf_tmp = gpd.GeoDataFrame(data=pd.DataFrame({'area': ''}, index=[r_ct_pred]))
                                    gdf_tmp['tile_row'] = full_r
                                    gdf_tmp['tile_col'] =  full_c
                                    gdf_tmp['tile_size'] = full_tile_size
                                    gdf_out = pd.concat([gdf_out, gdf_tmp])
                                    del gdf_tmp
                                    r_ct_pred += 1
                            t2=time.time()
                            print('... completed in', round(t2 - t1, 1), 'secs')
                        if type(gdf_out) is pd.core.frame.DataFrame:
                            gdf_out.to_csv(re.sub('.shp', '.csv', outSHP), index=False)
                        else:
                            gdf_out.to_file(outSHP)
                        try:
                            del pr_mask, pred_tmp, pr_labels, pr_regions, image_dict, image_sub_dict, image_list, image_out
                        except NameError:
                            pass
                        gc.collect()
                        client.run(gc.collect)
                        client.run(trim_memory)
                        if (tile_ct > 0) & (tile_ct % 15 == 0):
                            try:
                                client.restart(timeout=9)
                                time.sleep(10)
                            except TimeoutError:
                                client.shutdown()
                                client.close()
                                cluster = LocalCluster(n_workers=8, threads_per_worker=2, processes=True)
                                client = Client(cluster)
                                client.amm.start()

                        #client.restart()
                    except NoDataInBounds:
                        gdf_tmp = gpd.GeoDataFrame(data=pd.DataFrame({'area': ''}, index=[r_ct_pred]))
                        gdf_tmp['tile_row'] = full_r
                        gdf_tmp['tile_col'] =  full_c
                        gdf_tmp['tile_size'] = full_tile_size
                        gdf_out = pd.concat([gdf_out, gdf_tmp])
                        del gdf_tmp
                        r_ct_pred += 1
                        if type(gdf_out) is pd.core.frame.DataFrame:
                            gdf_out.to_csv(re.sub('.shp', '.csv', outSHP), index=False)
                        else:
                            gdf_out.to_file(outSHP)
                        continue
                        #print('No data in bounds. Skipping row/column.')
                tile_ct += 1
            #if not '_'.join([str(full_r), str(full_c)]) in rc_completed:
        print('Pasture-group finished!')
        try:
            client.restart(timeout=9)
            time.sleep(10)
        except TimeoutError:
            client.shutdown()
            client.close()
            cluster = LocalCluster(n_workers=8, threads_per_worker=2, processes=True)
            client = Client(cluster)
            client.amm.start()
