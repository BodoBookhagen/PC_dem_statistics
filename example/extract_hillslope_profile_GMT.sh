#!/bin/bash
gmt gmtset MAP_FRAME_PEN    1
gmt gmtset MAP_FRAME_WIDTH    0.1
gmt gmtset MAP_FRAME_TYPE     plain
gmt gmtset FONT_TITLE    Helvetica-Bold
gmt gmtset FONT_LABEL    Helvetica-Bold 14p
gmt gmtset PS_PAGE_ORIENTATION    landscape
gmt gmtset PS_MEDIA    A4
gmt gmtset FORMAT_GEO_MAP    D
gmt gmtset MAP_DEGREE_SYMBOL degree
gmt gmtset PROJ_LENGTH_UNIT cm

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/usr/local/gmt/lib
export GMTSAR=/usr/local/gmt
export PATH=$GMTSAR/bin:$PATH

#DEFINE GRIDS to be sampled and plotted
ELEVATION_MEAN_1m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_1.00m_z_mean.nc
ELEVATION_MEAN_5m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_5.00m_z_mean.nc
ELEVATION_MEAN_10m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_10.00m_z_mean.nc
ELEVATION_STD_1m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_1.00m_dz_stddev.nc
ELEVATION_STD_5m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_5.00m_dz_stddev.nc
ELEVATION_STD_10m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_10.00m_dz_stddev.nc
DZ_STD_1m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_1.00m_dz_stddev.nc
DZ_STD_5m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_5.00m_dz_stddev.nc
DZ_STD_10m=geotiff/Pozo_USGS_UTM11_NAD83_all_color_cl2_cat1_10.00m_dz_stddev.nc

HILLSLOPE_PROFILE=hillslope_profile.gmt
HILLSLOPE_PROFILE=hillslope_profile.gmt
echo "235120 3764500" >$HILLSLOPE_PROFILE
echo "235550 3764300" >>$HILLSLOPE_PROFILE

#sample profile every 0.25 km with profile length of 1km (0.5km on each side of profile) and do this every 2km along shapefile: -C1k/0.25/2
#-Sa+c1+selevation_profile_stacked.txt
 
gmt grdtrack -E235120/3764500/235550/3764300 -G$ELEVATION_MEAN_1m -G$ELEVATION_STD_1m -G$DZ_STD_1m  >hillslope_profile_1m.txt
gmt grdtrack -E235120/3764500/235550/3764300 -G$ELEVATION_MEAN_5m -G$ELEVATION_STD_5m -G$DZ_STD_5m  >hillslope_profile_5m.txt
gmt grdtrack -E235120/3764500/235550/3764300 -G$ELEVATION_MEAN_10m -G$ELEVATION_STD_10m -G$DZ_STD_10m  >hillslope_profile_10m.txt

