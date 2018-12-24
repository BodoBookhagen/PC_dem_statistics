#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:22:03 2017

@author: Bodo Bookhagen,    V0.1 Oct-Nov, 2017
                            V0.2 Sep 2018


example call:
python -W ignore /raid-cachi/bodo/Dropbox/soft/github/PC_dem_statistics/PC_DEM_statistics.py \
--inlas /raid-cachi/bodo/Dropbox/California/SCI/Pozo/cat1/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1.las \
--raster_m_range "1 10 1" \
--shapefile_clip /raid-cachi/bodo/Dropbox/California/SCI/Pozo/shapefiles/Pozo_DTM_noveg_UTM11_NAD83_cat1.shp \
--epsg_code 26911 --nr_of_cores 12 --create_geotiff 1 --create_gmt 1 --create_las 1 \
--create_shapefiles 0 2>&1 | tee Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_pc_dem_statistics_1_10_1.log

or
python -W ignore /raid-cachi/bodo/Dropbox/soft/github/PC_dem_statistics/PC_DEM_statistics.py \
--inlas /raid-cachi/bodo/Dropbox/California/SCI/Pozo/Pozo_USGS_UTM11_NAD83_all_color_cl2.las \
--raster_m_range "1 10 1" \
--shapefile_clip /raid-cachi/bodo/Dropbox/California/SCI/SCI_Pozo_100m_buffer_catchment_UTM11N_NAD83.shp \
--epsg_code 26911 --nr_of_cores 12 --create_geotiff 1 --create_gmt 1 --create_las 1 \
--create_shapefiles 0 2>&1 | tee Pozo_USGS_UTM11_NAD83_all_color_cl2_pc_dem_statistics_1_10_1.log
 
"""

from laspy.file import File
import copy, glob, time, sys
import numpy as np, os, argparse, pickle, h5py, subprocess, gdal, osr, datetime
from numpy.linalg import svd
from pykdtree.kdtree import KDTree
from scipy import interpolate
#from scipy import spatial
from scipy import linalg
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.lines as mlines
#import matplotlib.dates as mdates
#import matplotlib.patches as mpatches
#from matplotlib.colors import LogNorm
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
from multiprocessing import Pool
from skimage import exposure

### Function definitions
def cmdLineParser():
    # Command Line Parsing
    parser = argparse.ArgumentParser(description='PointCloud (PC) processing for DEM statistics. Deriving gridded ground data (elevation and slope) using centroid coordinates. B. Bookhagen (bodo.bookhagen@uni-potsdam.de), V0.1 Oct 2018.')
    # Important and required:
    parser.add_argument('--inlas', type=str, default='',  help='LAS/LAZ file with point-cloud data. Ideally, this file contains only ground points (class == 2)')
    parser.add_argument('--raster_m', type=float, default=1,  help='Raster spacing for subsampling seed points on LAS/LAZ PC. Usually 0.5 to 10 m, default = 1.')
    parser.add_argument('--raster_m_range', type=str, default="",  help='Raster spacing for subsampling seed points on LAS/LAZ PC. Uses a list of ranges with spacing, e.g., --raster_m_range "1 10 1" will create raster files with spatial resolutions of 1 to 10 m in 1 m steps.')
    parser.add_argument('--dem_fname', type=str, default='',  help='Filename of DEM to extract point spacing. Used to identify seed-point coordinates')
    parser.add_argument('--shapefile_clip', type=str, default='',  help='Name of shapefile to be used to clip interpolated surfaces. This is likely the shapefile you have previously generated to subset/clip the point-cloud data.')
    parser.add_argument('--epsg_code', type=int, default=26911,  help='EPSG code (integer) to define projection information. This should be the same EPSG code as the input data (no re-projection included yet) and can be taken from LAS/LAZ input file. Add this to ensure that output shapefile and GeoTIFFs are properly geocoded.')
    parser.add_argument('--create_geotiff', type=int, default=0,  help='Create interpolated geotif files from PC data (default no: --create_geotiff 0, set to --create_geotiff 1 to generate geotiff files). Note that creating geotiff files may increase processing time.')
    parser.add_argument('--create_shapefiles', type=int, default=0,  help='Create point shapefiles in UTM (see --epsg_code) and Geographic-DD projection. These contain all attributes calculated during the processing (default no: --create_shapefiles 0, set to --create_shapefiles 1 to generate shapefiles).')
    parser.add_argument('--create_gmt', type=int, default=0,  help='Create gmt point or vector files for plotting with GMT shapefiles in UTM (see --epsg_code) and Geographic-DD projection. These contain all attributes calculated during the processing (default no: --create_gmt 0, set to --create_gmt 1 to generate GMT files).')
    parser.add_argument('--create_las', type=int, default=0,  help='Create LAS point file from seed points  (currently no writing of LAZ files supported). The color shows mean elevation of the seed points. These contain all attributes calculated during the processing (default no: --create_las 0, set to --create_las 1 to generate LAS files).')
    parser.add_argument('--mean_z_only', type=int, default=0,  help='Calculate mean elevation for grid cell size and no other parameters.')
    parser.add_argument('--nr_of_cores', type=int, default=0,  help='Max. number of cores to use for multi-core processing. Default is to use all cores (0), set to --nr_of_cores 6 to use 6 cores. For some memory-intensive applications, it may be useful to reduce the number of cores.')
    parser.add_argument('--max_nr_of_neighbors_kdtree', type=int, default=1000,  help='Setting the max. number of neighbors for KDTree search. This can remain at 100 points for airborne lidar data. You may want to consider increasing this when using terrestrial lidar data or SfM data.')
    parser.add_argument('--pt_lower_threshold', type=int, default=5,  help='Lower point threshold for performing plane fitting and slope normalization. If there are less than pt_lower_threshold in the seed-point neighborhood, a point fitting is not performed and values are set to NaN.')
    return parser.parse_args()

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    source = https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    try:
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    except AssertionError:
        return np.nan, np.nan, np.nan
    
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    plane_normal = svd(M)[0][:,-1]
    d = -ctr.dot(plane_normal)
    z = (-plane_normal[0] * points[0,:] - plane_normal[1] * points[1,:] - d) * 1. /plane_normal[2]
    errors = z - points[2,:]
    residual = np.linalg.norm(errors)

    return ctr, plane_normal, residual

def curvFit_lstsq_polygon(points, order=2):
    """
    Fitting a second order polynom to a point cloud and deriving the curvature in a simplified form.
    More details: https://gis.stackexchange.com/questions/37066/how-to-calculate-terrain-curvature
    """
    
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    try:
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    except AssertionError:
        return np.nan, np.nan, np.nan
    points = points.T
    
    X,Y = np.meshgrid(np.arange(np.nanmin(points[:,0]),np.nanmax(points[:,0]), current_rstep_size/10), np.arange(np.nanmin(points[:,1]),np.nanmax(points[:,1]), current_rstep_size/10))
    XX = X.flatten()
    YY = Y.flatten()
    if order == 1:
        # best-fit linear plane
        A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
        C,_,_,_ = linalg.lstsq(A, points[:,2])    # coefficients
        
        # evaluate it on grid
        #Z = C[0]*X + C[1]*Y + C[2]
        
        # or expressed using matrix/vector product
        #Z_order1 = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
        slope = np.mean(C[0:2])
        curvature = np.nan
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(points.shape[0]), points[:,:2], np.prod(points[:,:2], axis=1), points[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, points[:,2])
        
        # evaluate it on a grid
        Z_order2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        #Z = Dx² + Ey² + Fxy + Gx + Hy + I
        #Curvature = -2(D + E)
        #Slope = sqrt(G^2 + H ^2)
        curvature = -2 * (C[4] + C[5])
        slope = np.sqrt( C[1]**2 + C[2]**2 )

        Z_pts = np.dot(np.c_[np.ones(points.shape[0]), points[:,0], points[:,1], points[:,0]*points[:,1], points[:,0]**2, points[:,1]**2], C)
        errors = points[:,2] - Z_pts
        dZ_residuals = np.linalg.norm(errors)
    del A, C, Z_order2, Z_pts, errors
    return slope, curvature, dZ_residuals


def gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in, zfield, input_vrt, output_grid,radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=''):
    #Use gdal_grid to interpolate from point data to grid
    interpolation_method_string='nearest:radius1='+str(radius1)+':radius2='+str(radius2)+':angle=0.0:nodata=-9999'
    epsg_string='epsg:'+str(inps.epsg_code)
    if clipsrclayer == '':
        cmd = ['gdal_grid', '-of', 'GTiff', '-co', 'PREDICTOR=3', '-co', 'COMPRESS=DEFLATE', '-co', 'ZLEVEL=7', '-ot', grid_datatype, \
               '-txe', str(np.min(x_coords)), str(np.max(x_coords)), '-tye', str(np.max(y_coords)), str(np.min(y_coords)),  \
               '-zfield', zfield, '-a', interpolation_method_string, \
               '-outsize',str(ncols), str(nrows), '-a_srs', epsg_string, '-clipsrclayer', inps.shapefile_clip, '-l', \
               layer_in, input_vrt, output_grid]
    else:
        cmd = ['gdal_grid', '-of', 'GTiff', '-co', 'PREDICTOR=3', '-co', 'COMPRESS=DEFLATE', '-co', 'ZLEVEL=7', '-ot', grid_datatype, \
               '-txe', str(np.min(x_coords)), str(np.max(x_coords)), '-tye', str(np.max(y_coords)), str(np.min(y_coords)),  \
               '-zfield', zfield, '-a', interpolation_method_string, \
               '-outsize',str(ncols), str(nrows), '-a_srs', epsg_string, '-l', \
               layer_in, input_vrt, output_grid]
    #print(' '.join(cmd))
    logfile_fname = os.path.join(inps.basedir, 'log') + '/gdal_grid_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(inps.basedir, 'log') + '/gdal_grid_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
    if os.path.exists(output_grid):
        ds = gdal.Open(output_grid)
        data_tif = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
        data_tif[np.where(data_tif == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
        ds = None
    else:
        print('gdal_grid could not create %s.'%output_grid)
        data_tif=np.NaN
    return data_tif

def griddata_clip_geotif(tif_fname, points, data2i, xxyy, ncols, nrows, geotransform):
    #interpolate point to a gridded dataset using interpolate.griddata and nearest neighbor interpolation. Next, data will be clipped by shapefile to remove potential interpolation artifacts
    #sample call:
    #griddata_clip_geotif(nrlidari_tif_fn, points, pts_seed_stats[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
    datai = interpolate.griddata(points, data2i, xxyy, method='nearest')
    output_raster = gdal.GetDriverByName('GTiff').Create(tif_fname,ncols, nrows, 1 ,gdal.GDT_Float32,['TFW=YES', 'COMPRESS=DEFLATE', 'ZLEVEL=9'])  # Open the file, see here for information about compression: http://gis.stackexchange.com/questions/1104/should-gdal-be-set-to-produce-geotiff-files-with-compression-which-algorithm-sh
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(inps.epsg_code)
    output_raster.SetProjection( srs.ExportToWkt() )
    output_raster.GetRasterBand(1).WriteArray(datai) 
    output_raster.FlushCache()
    output_raster=None
    if os.path.exists(inps.shapefile_clip) == False:
        print('Shapefile does not exist: %s'%inps.shapefile_clip)
        exit()
    tif_fname2 = os.path.join(os.path.dirname(tif_fname),'.'.join(os.path.basename(tif_fname).split('.')[0:-1]) + '2.tif')
    cmd = ['gdalwarp', '--config', 'GDALWARP_IGNORE_BAD_CUTLINE', 'YES', '-dstnodata', '-9999', '-co', 'COMPRESS=DEFLATE', '-co', 'ZLEVEL=9', '-crop_to_cutline', '-cutline', inps.shapefile_clip, tif_fname, tif_fname2]
    logfile_fname = os.path.join(inps.basedir, 'log') + '/gdalwarp_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(inps.basedir, 'log') + '/ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
    os.remove(tif_fname)
    os.rename(tif_fname2, tif_fname)
    cmd = ['gdalinfo', '-hist', '-stats', '-mm', tif_fname]
    logfile_fname = os.path.join(inps.basedir, 'log') + '/gdalinfo_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt'
    logfile_error_fname = os.path.join(inps.basedir, 'log') + '/gdalinfo_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt'
    with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
        subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
        subprocess_p.wait()
    ds = gdal.Open(tif_fname)
    datai = np.array(ds.GetRasterBand(1).ReadAsArray())
    datai[np.where(datai == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
    ds = None
    return datai


def calc_stats_for_seed_points_wrapper(i):
    #print('starting {}/{}'.format(i+1, len(pos_array)))
    from_pos = pos_array[i] #Get start/end from position array
    to_pos = pos_array[i+1]
    #Setup array for seed point results:
    subarr = np.arange(from_pos,to_pos) #Slice the data into the selected part...
    pts_seed_stats_result = np.empty((subarr.shape[0], nr_of_datasets))
    
    #Setup array for PC results (X, Y, Z, Dz)
    dxyzn_subarr_result = np.empty((subarr.shape[0], dxyzn_max_nre, 4))
 
    for ii in range(subarr.shape[0]):
        pts_seed_stats_result[ii,:], dxyzn_subarr_result[ii,:,:] = calc_stats_for_seed_points(subarr[ii]) #Run point cloud processing for this inddex

    pickle_fn = os.path.join(pickle_dir, 'PC_seed_points_{}.pickle'.format(str(i).zfill(4)))
    pickle.dump((pts_seed_stats_result, dxyzn_subarr_result), open(pickle_fn,'wb'))
    if np.mod(i,10) == 0:
        print('{}, '.format(str(i).zfill(2)), end='', flush=True)
    pts_seed_stats_result = None
    dxyzn_subarr_result = None
        
def calc_stats_for_seed_points(i):
    ids2use = np.where(pc_xyz_distance[i] != np.inf)[0]
    pts_xyz = pc_xyz[pc_xyz_distance_id[i][ids2use]]
        
    nr_pts_xyz = pts_xyz.shape[0]
    if inps.mean_z_only == 1:
        #only calculate mean elevation for grid-cell size
        if pts_xyz.shape[0] < inps.pt_lower_threshold:
            pts_xyz_meanpt = np.nan
            pts_xyz_normal = np.nan
            pts_seed_stats = np.array([pc_xyz_rstep_seed[i,0], pc_xyz_rstep_seed[i,1], pc_xyz_rstep_seed[i,2], 
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                       np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan, np.nan])
            dxyzn = np.empty((dxyzn_max_nre, 4))
            dxyzn.fill(np.nan)
        else:
            pts_seed_stats = np.array([pc_xyz_rstep_seed[i,0], pc_xyz_rstep_seed[i,1], pc_xyz_rstep_seed[i,2], 
                           np.mean(pts_xyz[:,0]), np.mean(pts_xyz[:,1]), np.mean(pts_xyz[:,2]), 
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                           np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan, np.nan])
            dxyzn = np.empty((dxyzn_max_nre, 4))
            dxyzn.fill(np.nan)
        return pts_seed_stats, dxyzn
        
    if pts_xyz.shape[0] < inps.pt_lower_threshold:
        #print('Less than 5 points, plane fitting not meaningful for i = %s'%"{:,}".format(i))
        pts_xyz_meanpt = np.nan
        pts_xyz_normal = np.nan
        pts_seed_stats = np.array([pc_xyz_rstep_seed[i,0], pc_xyz_rstep_seed[i,1], pc_xyz_rstep_seed[i,2], 
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                   np.nan, np.nan, np.nan, np.nan, nr_pts_xyz, np.nan, np.nan, np.nan, np.nan])
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
    else:
        pts_xyz_meanpt, pts_xyz_normal, plane_residual = planeFit(pts_xyz.T)
        #residual calculated from = np.linalg.norm(errors)

        #calculate curvature
        slope_lstsq, curvature_lstsq, curv_residuals = curvFit_lstsq_polygon(pts_xyz.T, order=2)

        #normalize /detrend points with plane
        d = -pts_xyz_meanpt.dot(pts_xyz_normal)
        z = (-pts_xyz_normal[0] * pts_xyz[:,0] - pts_xyz_normal[1] * pts_xyz[:,1] - d) * 1. /pts_xyz_normal[2]
        plane_slope = pts_xyz_normal[2]
        #calculate offset for each point from plane
        dz = pts_xyz[:,2] - z
    
        #stack points into X, Y, Z, delta-Z for each point
        dxyzn = np.empty((dxyzn_max_nre, 4))
        dxyzn.fill(np.nan)
        dxyzn[range(pts_xyz.shape[0]),:] = np.vstack([np.vstack((pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2], dz)).T])
    
        #for each seed point, store relevant point statistics. Columns are:
        #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
        #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: variance dz, 15: slope of fitted plane, 16: plane residuals, \
        #17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals
        pts_seed_stats = np.array([pc_xyz_rstep_seed[i,0], pc_xyz_rstep_seed[i,1], pc_xyz_rstep_seed[i,2], 
                       pts_xyz_meanpt[0], pts_xyz_meanpt[1], pts_xyz_meanpt[2], 
                       np.min(pts_xyz, axis=0)[2], np.max(pts_xyz, axis=0)[2], dz.max(), dz.min(), np.std(dz), dz.max()-dz.min(), \
                       np.percentile(dz, 90)-np.percentile(dz,10), np.percentile(dz, 75)-np.percentile(dz,25), np.var(dz), plane_slope, plane_residual, \
                       nr_pts_xyz, slope_lstsq, curvature_lstsq, curv_residuals, np.nanstd(pts_xyz[:,2])])
    return pts_seed_stats, dxyzn

#Start of the main program
if __name__ == '__main__': 

    inps = cmdLineParser()

#    inps = argparse.ArgumentParser(description='PointCloud (PC) processing for DEM statistics. Deriving gridded ground data (elevation and slope) using centroid coordinates. B. Bookhagen (bodo.bookhagen@uni-potsdam.de), V0.1 Oct 2018.')
#    inps.nr_of_cores = 0
#    inps.inlas = '/raid-cachi/bodo/Dropbox/California/SCI/Pozo/cat1/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1.las'
#    inps.raster_m_range='1 10 1'
#    inps.shapefile_clip = '/raid-cachi/bodo/Dropbox/California/SCI/Pozo/shapefiles/Pozo_DTM_noveg_UTM11_NAD83_cat1.shp'
#    inps.epsg_code=26911
#    inps.max_nr_of_neighbors_kdtree = 1000
#    inps.pt_lower_threshold = 5
#    inps.nr_of_cores=12
#    inps.create_geotiff = 1
#    inps.create_las = 1
#    inps.create_gmt = 1
#    inps.create_shapefiles = 0
#    
    if not inps.raster_m_range == '':
        #not empty, extracting start, end, and step sizes
        range_m_start = float(inps.raster_m_range.split(' ')[0])
        range_m_stop = float(inps.raster_m_range.split(' ')[1])
        range_m_step = float(inps.raster_m_range.split(' ')[2])
        
    inps.basedir = os.path.dirname(inps.inlas)
    if inps.basedir == '':
        inps.basedir = os.getcwd()
        
    pickle_dir = os.path.join(inps.basedir, 'pickle')
    if os.path.exists(pickle_dir) == False:
        os.mkdir(pickle_dir)

    figure_dir = os.path.join(inps.basedir, 'figures')
    if os.path.exists(figure_dir) == False:
        os.mkdir(figure_dir)
    
    geotif_dir = os.path.join(inps.basedir, 'geotiff')
    if os.path.exists(geotif_dir) == False:
        os.mkdir(geotif_dir)
    
    vrt_dir = os.path.join(inps.basedir, 'vrt')
    if os.path.exists(vrt_dir) == False:
        os.mkdir(vrt_dir)

    hdf_dir = os.path.join(inps.basedir, 'hdf')
    if os.path.exists(hdf_dir) == False:
        os.mkdir(hdf_dir)

    las_dir = os.path.join(inps.basedir, 'LAS')
    if os.path.exists(las_dir) == False:
        os.mkdir(las_dir)

    log_dir = os.path.join(inps.basedir, 'log')
    if os.path.exists(log_dir) == False:
        os.mkdir(log_dir)

    ### Loading data and filtering
    if os.path.exists(inps.inlas) == False:
        print('\n%s does not exist. Exiting.'%inps.inlas)
        sys.exit
    else:    
        print('\nLoading input file: %s'%inps.inlas)
    inFile = File(inps.inlas, mode='r')
    pc_xyz = np.vstack((inFile.get_x()*inFile.header.scale[0]+inFile.header.offset[0], inFile.get_y()*inFile.header.scale[1]+inFile.header.offset[1], inFile.get_z()*inFile.header.scale[2]+inFile.header.offset[2])).transpose()
    pc_xy = np.vstack((inFile.get_x()*inFile.header.scale[0]+inFile.header.offset[0], inFile.get_y()*inFile.header.scale[1]+inFile.header.offset[1])).transpose()
    #pc_xyz is now a point cloud with x, y, z
    print('\tLoaded %s points'%"{:,}".format(pc_xyz.shape[0]))
    
    ### pyKDTree setup and calculation
    #Generate KDTree for fast searching
    #cKDTree is faster than KDTree, but pyKDTree is fast then cKDTree
    print('\tGenerating XY-pyKDTree (2D) ... ',end='', flush=True)
    pc_xy_pykdtree = KDTree(pc_xy)
    print('done.')
    
    print('\tGenerating XYZ-pyKDTree (3D) ... ',end='', flush=True)
    pc_xyz_pykdtree = KDTree(pc_xyz)
    print('done.')

    #Setup coordinates
    [x_min, x_max] = np.min(pc_xyz[:,0]), np.max(pc_xyz[:,0])
    [y_min, y_max] = np.min(pc_xyz[:,1]), np.max(pc_xyz[:,1])
    [z_min, z_max] = np.min(pc_xyz[:,2]), np.max(pc_xyz[:,2])
    
    ### Search KDTree with points on a regularly-spaced raster
    #generating equally-spaced raster overlay from input coordinates with stepsize rstep_size
    #This will be used to query the point cloud. Step_size should be small enough and likely 1/2 of the output file resolution. 
    #Note that this uses a 2D raster overlay to slice a 3D point cloud.
    range_m_list = np.arange(range_m_start, range_m_stop+range_m_step, range_m_step)
    ts0 = time.time()
    for i in range(len(range_m_list)):
        current_rstep_size = range_m_list[i]
        print('At iteration %d of %d with grid size %0.2f m\n'%(i+1, len(range_m_list), current_rstep_size), end='', flush=True)
        current_search_radius = current_rstep_size/2
        rstep_size = current_rstep_size
        x_elements = len(np.arange(x_min.round(), x_max.round(), rstep_size))
        y_elements = len(np.arange(y_min.round(), y_max.round(), rstep_size))
        
        #get coordinate range and shift coordinates by half of the step size to make sure rater overlay is centered. 
        #This is not really necessary and only matters for very small point clouds with edge effects or for very large steps sizes:
        x_coords = np.arange(x_min.round(), x_max.round(), rstep_size) + rstep_size / 2
        y_coords = np.arange(y_min.round(), y_max.round(), rstep_size) + rstep_size / 2
        
        #create combination of all coordinates (this is using lists and could be optimized)
        xy_coordinates = np.array([(x,y) for x in x_coords for y in y_coords])
            
        #using the 2D KDTree to find the points that are closest to the defined 2D raster overlay
        [pc_xy_pykdtree_distance, pc_xy_pykdtree_id] = pc_xy_pykdtree.query(xy_coordinates, k=1)
        
        #take only points that are within the search radius (may omit some points at the border regions
        pc_distances_lt_rstep_size = np.where(pc_xy_pykdtree_distance <= rstep_size)[0]
        
        #the following list contains all IDs to the actual lidar points that are closest to the center of raster overlay. 
        #We will use these points as seed points for determining slopes, ground elevation and perform plane-fitting
        pc_xy_pykdtree_id = pc_xy_pykdtree_id[pc_distances_lt_rstep_size]
        
        
        #now select these points from the 3D pointcloud with X, Y, Z coordinates.
        #We refer to these as seed points from the rstep part of this script
        pc_xyz_rstep_seed = pc_xyz[pc_xy_pykdtree_id]

        #remove 2D pyKDTree from memory
        pc_xy_pykdtree_distance = None
        pc_xy_pykdtree_id = None
        
        ### Query points - use KDTree
        #find points from 3D seed / query points  / raster overlay with radius = inps.sphere_radius_m
        print('\tQuerying pyKDTree with radius %0.2f m... '%(current_search_radius), end='', flush=True)
        pc_xyz_distance, pc_xyz_distance_id  = pc_xyz_pykdtree.query(pc_xyz_rstep_seed, k=inps.max_nr_of_neighbors_kdtree, distance_upper_bound=current_search_radius)
        print('done.')
        
        ### Calculate statistics for each seed point (and each associated sphere): normalization, elevation range, std. dev., mean, median
        #for each seed point, store relevant point statistics. Columns are:
        #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
        #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: variance dz, 15: slope of fitted plane, 16: plane residuals, \
        #17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals
    
        ## Setup variables
        pc_xyz_distance_nr = len(pc_xyz_distance)
        nr_of_seed_points = len(pc_xyz_rstep_seed)
        nr_of_datasets = 22 #nr of columns to save
        nr_of_processes = 100 #splitting the for loop into 100 processes and dividing array into 100 steps in pos_array
        dxyzn_max_nre = np.max([len(x) for x in pc_xyz_distance]) #extract list of neighboring points
        dxyzn_nre = np.sum([len(x) for x in pc_xyz_distance])
        print('dxyzn_nre: %s'%"{:,}".format(dxyzn_nre))
        dxyzn_nre_pos_array = np.array(np.linspace(0, dxyzn_nre, nr_of_processes), dtype=int)
        pos_array = np.array(np.linspace(0, nr_of_seed_points, nr_of_processes), dtype=int) #This creates a position array so you can select from:to in each loop
        if inps.nr_of_cores == 0:
            inps.nr_of_cores = multiprocessing.cpu_count()
    
        
        print('\tCalculating PC statistics for radius %0.2f m using %d/%d cores for %s seed points [%%]:'%(current_search_radius, np.round(inps.nr_of_cores).astype(int), multiprocessing.cpu_count(), "{:,}".format(nr_of_seed_points)), end='', flush=True )
        #for each seed point, store relevant point statistics. Columns are:
        #0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  10: Dz-std.dev, 11: Dz-range, \
        #12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: variance dz, 15: slope of fitted plane, 16: plane residuals, \
        #17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals
    
        ts = time.time()
        p = Pool(processes=np.round(inps.nr_of_cores).astype(int))
        for _ in p.imap_unordered(calc_stats_for_seed_points_wrapper, np.arange(0,len(pos_array)-1)):
            pass    
        print('\n',end='', flush=True)
        
        #combine pickle files
        pkls = glob.glob(os.path.join(pickle_dir, 'PC_seed_points_*')) #Now get all the pickle files we made
        pkls.sort() #make sure they're sorted
        dxyzn = np.empty((dxyzn_nre, 4)) #output for every lidar point (dz value)
        pts_seed_stats = np.empty((pc_xyz_distance_nr,nr_of_datasets)) #output for seed points
        count = 0
        dxyzn_counter = 0
        for fid in pkls:
            seed_res, dxyzn_res = pickle.load(open(fid,'rb')) #Loop through and put each pickle into the right place in the output array
            if seed_res.shape[0] != pos_array[count+1] - pos_array[count]:
                print('File %s, length of records do not match. file: %d vs pos_array: %d'%(fid, seed_res.shape[0], pos_array[count+1] - pos_array[count]))
                if seed_res.shape[0] < pos_array[count+1] - pos_array[count]:
                    pts_seed_stats[range(pos_array[count],pos_array[count+1]-1),:] = seed_res
                elif seed_res.shape[0] > pos_array[count+1] - pos_array[count]:
                    pts_seed_stats[range(pos_array[count],pos_array[count+1]),:] = seed_res[:-1]
            else:
                pts_seed_stats[range(pos_array[count],pos_array[count+1]),:] = seed_res
            #re-arrange dxyzn and remove nans
            dxyzn_reshape = dxyzn_res.reshape((dxyzn_res.shape[0]*dxyzn_res.shape[1], dxyzn_res.shape[2]))
            idx_nonan = np.where(np.isnan(dxyzn_reshape[:,0]) == False)[0]
            dxyzn_reshape = dxyzn_reshape[idx_nonan,:]
            dxyzn[range(dxyzn_counter,dxyzn_counter+dxyzn_reshape.shape[0]),:] = dxyzn_reshape
            dxyzn_counter = dxyzn_counter + dxyzn_reshape.shape[0]
            count += 1
            del seed_res, dxyzn_res, dxyzn_reshape
        #remove pickle files
        for ii in pkls:
            os.remove(ii)
        pkls=None
    
        #write as compressed HDF file
        print('\tWriting to CSV, GMT, and shapefiles... ', end='', flush=True)
        #write csv
        header_str='1SeedX, 2SeedY, 3SeedZ, 4MeanX, 5MeanY, 6MeanZ, 7Z_min, 8Z_max, 9Dz_max, 10Dz_min, 11Dz_std, 12Dz_range, 13Dz_9010p, 14Dz_7525p, 15Dz_var, 16Pl_slp, 17Pl_res, 18Nr_lidar, 19SlopeLSQ, 20CurvLSQ, 21CurvRes, 22StdZ'
        seed_pts_stats_csv = '_seed_pts_stats_raster_%0.2fm.csv'%(current_rstep_size)
        pc_seed_pts_stats_csv_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_csv)
        seed_pts_stats_vrt = '_seed_pts_stats_raster_%0.2fm.vrt'%(current_rstep_size)
        pc_seed_pts_stats_vrt_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_vrt)
        seed_pts_stats_shp = '_seed_pts_stats_raster_%0.2fm.shp'%(current_rstep_size)
        pc_seed_pts_stats_shp_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_shp)
        seed_pts_stats_dd_shp = '_seed_pts_stats_raster_%0.2fm_dd.shp'%(current_rstep_size)
        pc_seed_pts_stats_dd_shp_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_dd_shp)
        seed_pts_stats_gmt = '_seed_pts_stats_raster_%0.2fm.gmt'%(current_rstep_size)
        pc_seed_pts_stats_gmt_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_gmt)
        seed_pts_stats_dd_gmt = '_seed_pts_stats_raster_%0.2fm_dd.gmt'%(current_rstep_size)
        pc_seed_pts_stats_dd_gmt_fn = os.path.join(vrt_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + seed_pts_stats_dd_gmt)
        idxnan = np.where(np.isnan(pts_seed_stats))
        if os.path.exists(pc_seed_pts_stats_csv_fn) == False:
            #before writing to CSV file, remove all lines with np.nan in pts_seed_stats
            pts_seed_stats_nonan = np.copy(pts_seed_stats)
            idx_nodata = np.where(np.isnan(pts_seed_stats_nonan))
            rows2remove = np.unique(idx_nodata[0])
            pts_seed_stats_nonan = np.delete(pts_seed_stats_nonan, (rows2remove), axis=0)
            np.savetxt(pc_seed_pts_stats_csv_fn, pts_seed_stats_nonan, fmt='%.4f', delimiter=',', header=header_str, comments='')
        pts_seed_stats_nonan = None
        idxnan = None
    
        # write VRT for shapefile generation
        vrt_f = open(pc_seed_pts_stats_vrt_fn,'w')
        vrt_f.write('<OGRVRTDataSource>\n')
        vrt_f.write('\t<OGRVRTLayer name="%s">\n'%os.path.basename(pc_seed_pts_stats_vrt_fn))
        vrt_f.write('\t\t<SrcDataSource>%s</SrcDataSource>\n'%os.path.join(vrt_dir, os.path.basename(pc_seed_pts_stats_csv_fn)))
        vrt_f.write('\t\t<SrcLayer>%s</SrcLayer>\n'%'.'.join(os.path.basename(pc_seed_pts_stats_csv_fn).split('.')[0:-1]))
        vrt_f.write('\t\t<LayerSRS>EPSG:%d</LayerSRS>\n'%inps.epsg_code)
        vrt_f.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
        vrt_f.write('\t\t<GeometryField encoding="PointFromColumns" x="4MeanX" y="5MeanY"/>\n')
        vrt_f.write('\t\t\t<Field name="1SeedX" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="2SeedY" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="3SeedZ" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="4MeanX" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="5MeanY" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="6MeanZ" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="7Z_min" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="8Z_max" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="9Dz_max" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="10Dz_min" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="11Dz_std" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="12Dz_range" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="13Dz_9010p" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="14Dz_7525p" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="15Dz_var" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="16Pl_slp" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="17Pl_res" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="18Nr_lidar" type="Real" width="8"/>\n')
        vrt_f.write('\t\t\t<Field name="19SlopeLSQ" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="20CurvLSQ" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="21CurvRes" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t\t\t<Field name="22StdZ" type="Real" width="8" precision="7"/>\n')
        vrt_f.write('\t</OGRVRTLayer>\n')
        vrt_f.write('</OGRVRTDataSource>\n')
        vrt_f.close()
    
        # Generate shapefile from VRT
        if inps.create_shapefiles == 1 and os.path.exists(pc_seed_pts_stats_shp_fn) == False:
            cwd=os.getcwd()
            os.chdir(vrt_dir)
            cmd = ['ogr2ogr', pc_seed_pts_stats_shp_fn, pc_seed_pts_stats_vrt_fn]
            logfile_fname = os.path.join(log_dir,  'ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'ogr2ogr_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            os.chdir(cwd)
        
        if inps.create_shapefiles == 1 and os.path.exists(pc_seed_pts_stats_dd_shp_fn) == False:
            cwd=os.getcwd()
            os.chdir(vrt_dir)
            cmd = ['ogr2ogr', '-t_srs', 'epsg:4326', pc_seed_pts_stats_dd_shp_fn, pc_seed_pts_stats_shp_fn]
            logfile_fname = os.path.join(log_dir, 'ogr2ogr_reproject_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'ogr2ogr_reproject_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            os.chdir(cwd)
    
        # Convert VRT to GMT dataset for plotting
        if inps.create_gmt == 1 and os.path.exists(pc_seed_pts_stats_gmt_fn) == False:
            cwd=os.getcwd()
            os.chdir(vrt_dir)
            cmd = ['ogr2ogr', '-f', 'GMT', pc_seed_pts_stats_gmt_fn, pc_seed_pts_stats_vrt_fn]
            logfile_fname = os.path.join(log_dir, 'ogr2ogr_gmt_utm_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir,  'ogr2ogr_gmt_utm_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            os.chdir(cwd)

        if inps.create_gmt == 1 and os.path.exists(pc_seed_pts_stats_dd_gmt_fn) == False:
            cwd=os.getcwd()
            os.chdir(vrt_dir)
            cmd = ['ogr2ogr', '-f', 'GMT', '-t_srs', 'epsg:4326', pc_seed_pts_stats_dd_gmt_fn, pc_seed_pts_stats_gmt_fn]
            logfile_fname = os.path.join(log_dir,  'ogr2ogr_gmt_dd_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '.txt')
            logfile_error_fname = os.path.join(log_dir, 'ogr2ogr_gmt_dd_' + datetime.datetime.now().strftime('%Y%b%d_%H%M%S') + '_err.txt')
            with open(logfile_fname, 'w') as out, open(logfile_error_fname, 'w') as err:
                subprocess_p = subprocess.Popen(cmd, stdout=out, stderr=err)
                subprocess_p.wait()
            os.chdir(cwd)
    
        print('\tdone.')
    
        ### Interpolate to equally-spaced grid and generate GeoTIFF output
        if inps.create_geotiff == 1:
            print('\tInterpolating seed points (mean-X, mean-Y) and writing geotiff raster... ',end='', flush=True)
            idx_nonan = np.where(np.isnan(pts_seed_stats[:,3])==False)
        
            xres=current_rstep_size
            yres=current_rstep_size
            ncols=len(x_coords)
            nrows=len(y_coords)
            geotransform = (x_coords.min() - (current_rstep_size / 2), xres, 0 , y_coords.min() - (current_rstep_size / 2),0, yres) 

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(inps.epsg_code)
            
            nrlidar_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_nrlidarpts.tif'%(current_rstep_size))
            z_std_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_z_stddev.tif'%(current_rstep_size))
            dz_std_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_dz_stddev.tif'%(current_rstep_size))
            dz_range9010_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_dz_range9010.tif'%(current_rstep_size))
            dz_iqr_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_dz_iqr.tif'%(current_rstep_size))
            plane_slope_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_planeslope.tif'%(current_rstep_size))
            plane_curv_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_curv.tif'%(current_rstep_size))
            z_mean_tif_fn = os.path.join(geotif_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_%0.2fm_z_mean.tif'%(current_rstep_size))

            #interpolate nr_lidar_measurements
#We have experimented with the following options:
## First, use scipy.interpolate (see function griddata_clip_geotif). While this works, it's incredible slow
#nr_lidar = griddata_clip_geotif(nrlidar_tif_fn, points, pts_seed_stats[idx_nonan,17][0], xxyy=(xx,yy), ncols=ncols, nrows=nrows, geotransform=geotransform)
#
## Second, the API of gdal.Grid, but for some reason this wasn't as efficient (but should work well)
#gridoptions = gdal.GridOptions(options=[], format="GTiff", outputType=gdal.GDT_Float32, width=nrows, height=ncols, creationOptions=None, \
#                               outputBounds=[ np.min(x_coords), np.min(y_coords), np.max(x_coords) , np.max(y_coords)], \
#                               outputSRS=srs, noData=-9999, algorithm='nearest:radius1=3.0:radius2=3.0:angle=0.0:nodata=-9999', \
#                               layers='Pozo_USGS_UTM11_NAD83_all_color_cl_cat1_seed_pts_stats_raster_1.00m', SQLStatement=None, where=None, \
#                               spatFilter=None, zfield='18Nr_lidar', z_increase=None, z_multiply=None, callback=None, callback_data=None)
#ds = gdal.Grid(nrlidar_tif_fn, pc_seed_pts_stats_vrt_fn, options=gridoptions)
#ds.SetGeoTransform(geotransform)
#ds.SetProjection( srs.ExportToWkt() )
#ds.GetRasterBand(1).WriteArray(datai) 
#ds.FlushCache()
#ds=None
#
## Third, the command line version of gdal_grid. This appears to be the fastest option
#                    nr_lidar=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
#                                                   zfield='18Nr_Lidar', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=nrlidar_tif_fn,\
#                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)

            if os.path.exists(nrlidar_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('nr_lidar_measurements, ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    nr_lidar=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='18Nr_Lidar', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=nrlidar_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Int16', clipsrclayer=inps.shapefile_clip)
                else:
                    nr_lidar=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='18Nr_Lidar', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=nrlidar_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Int16', clipsrclayer='')
            else:
                ds = gdal.Open(nrlidar_tif_fn)
                nr_lidar = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
                nr_lidar[np.where(nr_lidar == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
            
            if os.path.exists(z_std_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('z std. dev., ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    z_std=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='22StdZ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=z_std_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    z_std=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='22StdZ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=z_std_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(z_std_tif_fn)
                z_std = np.array(ds.GetRasterBand(1).ReadAsArray())
                z_std[np.where(z_std == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None

            if os.path.exists(dz_std_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('Dz std. dev., ', end='', flush=True)                
                if os.path.exists(inps.shapefile_clip):
                    dz_std=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='11Dz_std', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_std_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    dz_std=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='11Dz_std', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_std_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(dz_std_tif_fn)
                dz_std = np.array(ds.GetRasterBand(1).ReadAsArray())
                dz_std[np.where(dz_std == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None

            if os.path.exists(z_mean_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('z mean, ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    z_mean=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='6MeanZ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=z_mean_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    z_mean=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='6MeanZ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=z_mean_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(z_mean_tif_fn)
                z_mean = np.array(ds.GetRasterBand(1).ReadAsArray())
                z_mean[np.where(z_mean == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
            
            #interpolate Dz_range 90-10 percentile
            if os.path.exists(dz_range9010_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('Dz range (90-10th perc.), ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    dz_range9010=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='13Dz_9010p', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_range9010_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    dz_range9010=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='13Dz_9010p', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_range9010_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(dz_range9010_tif_fn)
                dz_range9010 = np.array(ds.GetRasterBand(1).ReadAsArray())
                dz_range9010[np.where(dz_range9010 == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
            
            #interpolate Dz_range 75-25 percentile
            if os.path.exists(dz_iqr_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('Dz range (75-25th perc.), ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    dz_iqr=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='14Dz_7525p', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_iqr_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    dz_iqr=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='14Dz_7525p', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=dz_iqr_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(dz_iqr_tif_fn)
                dz_iqr = np.array(ds.GetRasterBand(1).ReadAsArray())
                dz_iqr[np.where(dz_iqr == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
                        
            #interpolate Plane_slope
            if os.path.exists(plane_slope_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('plane slope, ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    plane_slope=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='16Pl_slp', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=plane_slope_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    plane_slope=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='16Pl_slp', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=plane_slope_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(plane_slope_tif_fn)
                plane_slope = np.array(ds.GetRasterBand(1).ReadAsArray())
                plane_slope[np.where(plane_slope == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
            
            #interpolate LST-Square curvate
            if os.path.exists(plane_curv_tif_fn) == False and os.path.exists(inps.shapefile_clip):
                print('LSTSQ curvature, ', end='', flush=True)
                if os.path.exists(inps.shapefile_clip):
                    plane_curv=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='20CurvLSQ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=plane_curv_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer=inps.shapefile_clip)
                else:
                    plane_curv=gdal_grid_interpolate(x_coords, y_coords, ncols, nrows, layer_in=os.path.basename(pc_seed_pts_stats_vrt_fn), \
                                                   zfield='20CurvLSQ', input_vrt=pc_seed_pts_stats_vrt_fn, output_grid=plane_curv_tif_fn,\
                                                   radius1=5.0, radius2=5.0, grid_datatype='Float32', clipsrclayer='')
            else:
                ds = gdal.Open(plane_curv_tif_fn)
                plane_curv = np.array(ds.GetRasterBand(1).ReadAsArray())
                plane_curv[np.where(plane_curv == ds.GetRasterBand(1).GetNoDataValue())] = np.nan
                ds = None
            print('   done.')

        print('\tWriting to HDF file ... ', end='', flush=True)
        hdf5_fn = str(os.path.basename(inps.inlas).split('.')[:-1][0]) + '_seed_pts_stats_raster_%0.2fm.h5'%current_rstep_size
        pc_results_fn = os.path.join(hdf_dir, hdf5_fn)
        hdf_out = h5py.File(pc_results_fn,'w')
        hdf_out.attrs['help'] = 'Results from slope normalization: radius %0.2fm'%( current_rstep_size) 
        pts_seed_stats_fc = hdf_out.create_dataset('pts_seed_stats',data=pts_seed_stats, chunks=True, compression="gzip", compression_opts=7)
        pts_seed_stats_fc.attrs['help'] = '''Nr. of seed pts: %d,  pts_seed_stats shape: %d x %d, with col: name 
        0: Seed-X, 1: Seed-Y, 2: Seed-Z, 3: Mean-X, 4: Mean-Y, 5: Mean-Z, 6: Z-min, 7: Z-max, 8: Dz-max, 9: Dz-min,  
        10: Dz-std.dev, 11: Dz-range, 12: Dz-90-10th percentile range, 13: Dz-75-25th percentile range, 14: variance dz, 
        15: slope of fitted plane, 16: plane residuals, 17: nr. of lidar points, 18: slope_lstsq, 19: curvature_lstsq, 20: curvature residuals, 21:Std. Z'''\
        %(nr_of_seed_points, pts_seed_stats.shape[0], pts_seed_stats.shape[1])

        if inps.create_geotiff == 1:
            nr_lidar_fc = hdf_out.create_dataset('nr_lidar',data=nr_lidar, chunks=True, compression="gzip", compression_opts=7)
            nr_lidar_fc.attrs['help'] = 'Nr. of lidar measurements per grid cell for seed point'
            dz_std_fc = hdf_out.create_dataset('dz_std',data=dz_std, chunks=True, compression="gzip", compression_opts=7)
            dz_std_fc.attrs['help'] = 'Dz standard deviation'
            dz_range9010_fc = hdf_out.create_dataset('dz_range9010',data=dz_range9010, chunks=True, compression="gzip", compression_opts=7)
            dz_range9010_fc.attrs['help'] = 'dz_range9010'
            dz_iqr_fc = hdf_out.create_dataset('dz_iqr',data=dz_iqr, chunks=True, compression="gzip", compression_opts=7)
            dz_iqr_fc.attrs['help'] = 'dz_iqr'
            plane_slope_fc = hdf_out.create_dataset('plane_slope',data=plane_slope, chunks=True, compression="gzip", compression_opts=7)
            plane_slope_fc.attrs['help'] = 'plane_slope'
            plane_curv_fc = hdf_out.create_dataset('plane_curv',data=plane_curv, chunks=True, compression="gzip", compression_opts=7)
            plane_curv_fc.attrs['help'] = 'plane_curv'
            z_std_fc = hdf_out.create_dataset('z_std',data=z_std, chunks=True, compression="gzip", compression_opts=7)
            z_std_fc.attrs['help'] = 'z_std'
            z_mean_fc = hdf_out.create_dataset('z_mean',data=z_mean, chunks=True, compression="gzip", compression_opts=7)
            z_mean_fc.attrs['help'] = 'z_mean'
            geotransform_fc = hdf_out.create_dataset('geotransform',data=geotransform)
            geotransform_fc.attrs['help'] = 'geotransform'
            epsg_code_fc = hdf_out.create_dataset('epsg_code',data=inps.epsg_code)
            epsg_code_fc.attrs['help'] = 'epsg_code'
            nr_lidar = None
            dz_std = None
            dz_range9010 = None
            dz_iqr = None
            plane_slope = None
            plane_curv = None
            z_std = None
            z_mean = None
        hdf_out.close()
        print('done.')
    
        ### Write to LAS/LAZ file
        outpus_las_fn = '_seed_pts_%0.2fm_radius.las'%(current_rstep_size)
        outpus_las_fn = os.path.join(las_dir, str(os.path.basename(inps.inlas).split('.')[:-1][0]) + outpus_las_fn)
        if inps.create_las == 1 and os.path.exists(outpus_las_fn) == False:
            print('\tWriting seed points to LAS file: %s... '%outpus_las_fn, end='', flush=True)    
            pts2write = pts_seed_stats[:,0:3]
            #normalize input and generate colors using colormap
            v = pts2write[:,2]
            #stretch to 10-90th percentile
            v_1090p = np.percentile(v, [10, 90])
            v_rescale = exposure.rescale_intensity(v, in_range=(v_1090p[0], v_1090p[1]))
            colormap_PuOr = mpl.cm.PuOr
            rgb = colormap_PuOr(v_rescale)
            #remove last column - alpha value
            rgb = (rgb[:, :3] * (np.power(2,16)-1)).astype('uint16')
        
            outFile = File(outpus_las_fn, mode='w', header=inFile.header)
            new_header = copy.copy(outFile.header)
            #setting some variables
            new_header.created_year = datetime.datetime.now().year
            new_header.created_day = datetime.datetime.now().timetuple().tm_yday
            new_header.x_max = pts2write[:,0].max()
            new_header.x_min = pts2write[:,0].min()
            new_header.y_max = pts2write[:,1].max()
            new_header.y_min = pts2write[:,1].min()
            new_header.z_max = pts2write[:,2].max()
            new_header.z_min = pts2write[:,2].min()
            new_header.point_records_count = pts2write.shape[0]
            new_header.point_return_count = 0
            outFile.header.count = v.shape[0]
        #    outFile.Classification = np.ones((dz.shape[0])).astype('uint8') * 2
            outFile.X = pts2write[:,0]
            outFile.Y = pts2write[:,1]
            outFile.Z = pts2write[:,2]
            outFile.Red = rgb[:,0]
            outFile.Green = rgb[:,1]
            outFile.Blue = rgb[:,2]    
            outFile.close()    
            print('done.')
    
        #remove data from memory
        pc_xyz_distance = None
        pc_xyz_distance_id = None
        pc_xyz_rstep_seed = None        
        points = None
        pts2write = None
        v = None
        rgb = None
        idx_nonan = None
        xx = None
        yy = None
        pts_seed_stats = None
        
        print('\ttime: %0.2fs or %0.2fm'%(time.time() - ts, (time.time() - ts)/60))        
        
    print('total time: %0.2fs or %0.2fm'%(time.time() - ts0, (time.time() - ts0)/60))        

