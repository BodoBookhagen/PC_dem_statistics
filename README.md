# Deriving optimal DEM resolution from point clouds
This code accompanies the manuscript Smith et al. [LINK & CITATION]. This specific code uses lidar point clouds as input data to derive the optimal DEM size. The code loads an LAS file, creates a equi-distance grid and calculated elevation, slope, and point-cloud scatter for each grid cell. The grid cell size varies and allows to estimate optimal conditions for slope and aspect calculations. 

## Installation
This is a Python 3.x code that will run on any OS, which supports the packages. It runs and has been tested on Linux (Ubuntu/Debian), Windows 10, and Mac OS X. You will need several packages for python to run this code. These are standard packages and are included in many distributionss. If you use [conda](https://conda.io/docs/index.html), you can install the required packages (currently we are using Python 3.6 or 3.7):
```
conda create -y -n py3 python scipy pandas numpy matplotlib scikit-image gdal ipython spyder h5py
```

You can active this environment with `source activate py3`.

You don't need ipython or spyder to run this code and you can remove these repositories in the command line above, but they usually come in handy. Also, if you plan to store and process large PC datasets, it may come in handy storing the processed data in compressed HDF5 or H5 files. However, for some installations, the above installed h5py does not contain the gzip compression (or any compression). And you may have to update the installation with 
```
source activate py3
conda install -y -c conda-forge h5py
```


Install a fast and simple LAS/LAZ reader/writer. You can do similar steps through lastools, but this interface is fairly simple to use:
```
source activate py3
pip install laspy
```
If you have issues with pip, see: https://stackoverflow.com/questions/47955397/pip3-error-namespacepath-object-has-no-attribute-sor


NOT WORKING YET (lastools will kill gdal): In order to read and write zipped LAS files (LAZ) files, install lastools. These will come in handy. Note that if you have installed pdal, you usually don't need this:
```
source activate py3
conda install -y -c conda-forge lastools
```

If you plan to use some commands/pipelines from pdal, install the following. We usually use this for point-cloud pre-processing, filtering, and classification:
```
source activate py3
conda install -c mathieu pdal 
```


This code uses [pykdtree](https://github.com/storpipfugl/pykdtree). There are other KDTree implementations, for example [scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.spatial.cKDTree.html). But pykdtree is faster (but doesn't allow you to save the results such as cKDTree). Because we aim at very large point clouds, the pyKDTree algorithm is significantly faster for generating and querying the KDtree and will increase processing speed (we have run tests with 1e6 to 1e9 points). To install pyKDTree:
```
source activate py3
conda install -y -c conda-forge pykdtree

```

Last, install the repository into your favorite github directory, for example to ~/github:
```
cd ~
git clone https://github.com/UP-RS-ESP/PC_DEM_stats

```
You are now ready to run the code from the command line (see below).

## First steps and running the code
The python code `pc_dem_statistics.py` loads a LAS pointcloud and overlays an equi-distance grid (e.g., 1 to 10 m). For each grid cell, the most central point (x, y, z coordinates) is chosen and is used to store field attributes. The code calculates the mean elevation and performs a least-square plane fitting for slope estimation. The slope is used to detrend the elevation data and store the natural scatter (dz) for that grid cell. We argue that the scattering of points around the mean is an indicator for how precise the elevation has been determined.

You should pre-process your point clouds. For example, you want your point cloud to only contain points that are used for estimating bare-earth ground. This usually requires classifying the pointcloud into ground points and other classes (e.g., buildings, vegetation). There are different ways to do this (e.g., lastools, pdal, and others). Again, the python code `pc_dem_statistics.py` requires bare-earth points only.

The code requires the following input parameters:
- --inlas <filename.las>
- --raster_m_range "<gridcell_m_start gridcell_m_stop gridcell_m_stepsize>"     (for example, --raster_m_range "1 10 1" will use 1m grid-cell sizes up to 10m in 1m steps)
- --shapefile_clip <filename.shp>       It is useful to have a polygon shapefile outlining the area of interest to remove potential border effects that may appear through interpolation (Note: <filename.shp> will need to be in the same projection as the las file)
- --epsg_code <EPSG-CODE>       This is used for writing Geotiff files. In order to generate geotifs with proper projection information, provide the EPSG code (e.g., --epsg_code 26911)

An example run for the example dataset provided in the [example](example) directory is:
```
python -W ignore ~/github/PC_DEM_stats/pc_dem_statistics.py \
    --inlas Pozo_USGS_UTM11_NAD83_all_color_cl2.las \
    --raster_m_range "1 10 1" \
    --nr_of_cores 20 \
    --shapefile_clip /raid-cachi/bodo/Dropbox/California/SCI/Pozo/shapefiles/Pozo_DTM_noveg_UTM11_NAD83_cat1_b50m.shp \
    --epsg_code 26911 2>&1 | tee Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_pc_dem_groud_slope_statistics_1_10_1.log
```

python -W ignore /raid-cachi/bodo/Dropbox/soft/github/PC_dem_statistics/PC_DEM_statistics.py --inlas Pozo_USGS_UTM11_NAD83_all_color_cl2.las     --raster_m_range "1 10 1"     --shapefile_clip /raid-cachi/bodo/Dropbox/California/SCI/SCI_Pozo_100m_buffer_catchment_UTM11N_NAD83.shp     --epsg_code 26911 --nr_of_cores 20 2>&1 | tee Pozo_USGS_UTM11_NAD83_all_color_cl2_pc_dem_groud_slope_statistics_1_10_1.log
