#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:33:55 2018

@author: bodo
"""
import pandas as pd
from io import StringIO
import ogr, osr, gdal
import numpy as np
import matplotlib.pyplot as plt

input_profile='elevation_velocity_profile.txt'
old = pd.read_csv("profile_no_nlm.csv")

utm_fname = 'offset_20150819_20170909_ws32_sws48_zws8_skip2_os4_snr09_Y_m_yr.tif'
inputEPSG = 4326
outputEPSG = int(gdal.Info(utm_fname, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
# create coordinate transformation fo rinput
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(inputEPSG)
# create coordinate transformation for output
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(outputEPSG)
coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

f = open(input_profile, 'r')
profile_data = f.read()
profile_chunks = profile_data.split('>')

profile_pd = pd.DataFrame(columns=['Longitude', 'Latitude', 'UTM_X', 'UTM_Y', 'Distance_Profile_m', 'Elevation_mean', 'Elevation_stddev', 'Vel1_mean', 'Vel1_stddev', 'Vel2_mean', 'Vel2_stddev', 'Vel3_mean', 'Vel3_stddev'])
for chunk in profile_chunks:
    chunk_pd = pd.DataFrame()
    if len(chunk) > 0:
        chunk_lines = chunk.split('\n')
        for i in chunk_lines:
            if 'Cross profile' in i:
                profile_nr = i.split(' ')[4].split('-')[2]
                profile_long = i.split(' ')[-2].split('/')[0]
                profile_lat = i.split(' ')[-2].split('/')[1]
                profile_az = i.split(' ')[-1].split('=')[1]
            else:
                chunk_pd = chunk_pd.append(pd.read_csv(StringIO(i), sep='\t', names=['Longitude', 'Latitude', 'Distance_km', 'Azimuth', 'Elevation', 'Vel1', 'Vel2','Vel3']))
        #now calculate statistcs for this chunk:
        elevation_mean = chunk_pd['Elevation'].mean()
        elevation_stddev = chunk_pd['Elevation'].std()
        vel1_mean = chunk_pd['Vel1'].mean()
        vel1_stddev = chunk_pd['Vel1'].std()
        vel2_mean = chunk_pd['Vel2'].mean()
        vel2_stddev = chunk_pd['Vel2'].std()
        vel3_mean = chunk_pd['Vel3'].mean()
        vel3_stddev = chunk_pd['Vel3'].std()
        chunk_pd = chunk_pd.reset_index(drop=True)
        center_latitutde = chunk_pd['Latitude'][chunk_pd['Distance_km'] == 0].values[0]
        center_longitude = chunk_pd['Longitude'][chunk_pd['Distance_km'] == 0].values[0]
        #convert Lat/Long to UTM
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(center_longitude, center_latitutde)
        # transform point
        point.Transform(coordTransform)
        # print point in EPSG 4326
        #print(point.GetX(), point.GetY())
        UTM_X = point.GetX()
        UTM_Y = point.GetY()        
        #add statistics to profile_pd Dataframe (only containing one point per profile)
        profile_pd = profile_pd.append({'Longitude': center_longitude, 'Latitude':center_latitutde,
                                        'UTM_X':UTM_X, 'UTM_Y':UTM_Y, 'Elevation_mean': elevation_mean,
                                        'Elevation_stddev': elevation_stddev, 'Vel1_mean': vel1_mean, 
                                        'Vel1_stddev': vel1_stddev, 'Vel2_mean': vel2_mean, 'Vel2_stddev': vel2_stddev,
                                        'Vel3_mean': vel3_mean, 'Vel3_stddev': vel3_stddev}, ignore_index=True)
        # create a geometry from coordinates
    else:
        continue
    profile_pd['Distance_Profile_m'] = np.sqrt(((profile_pd['UTM_X']-profile_pd['UTM_X'].shift(1))**2) + ((profile_pd['UTM_Y']-profile_pd['UTM_Y'].shift(1))**2))
    profile_pd['Distance_Profile_m'][0] = 0
    profile_pd['Distance_Profile_m'] = profile_pd['Distance_Profile_m'].cumsum()
    profile_pd['Distance_Profile_r_m'] = profile_pd['Distance_Profile_m'].iloc[::-1]

    
#plot profiles
window_size = 5
#plt.clf()
fig = plt.figure(figsize= (6,8))
ax1 = plt.subplot(311)
#ax1.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Elevation_mean'], 'k-')
ax1.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Elevation_mean'].rolling(window_size, center=True).mean(), 'k-')
ax1.set_xlabel('Distance along Profile (km)')
ax1.set_ylabel('Elevation (m)')
ax1.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Elevation_mean'].rolling(window_size, center=True).mean()-profile_pd['Elevation_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Elevation_mean'].rolling(window_size, center=True).mean()+profile_pd['Elevation_stddev'].rolling(window_size, center=True).mean(), color='gray')
ax1.grid()
ax1.set_xlim([60, 0])
ax1.set_ylim([2500, 6500])
plt.title('Elevation and flow velocities along the main flowline - Landsat')
ax2 = plt.subplot(312, sharex=ax1)

#plot old 
ax2.plot(old['Distance_Profile_r_m']/1000., old['Vel1_mean'].rolling(window_size, center=True).mean(),  color = 'seagreen', label='2013/14 unfiltered mean', alpha = 0.3)
ax2.fill_between(old['Distance_Profile_r_m']/1000., 
                 old['Vel1_mean'].rolling(window_size, center=True).mean()-old['Vel1_stddev'].rolling(window_size, center=True).mean(), 
                 old['Vel1_mean'].rolling(window_size, center=True).mean()+old['Vel1_stddev'].rolling(window_size, center=True).mean(), color='lightgreen', alpha = 0.3, label='2013/14 unfiltered std.dev.')
ax2.plot(old['Distance_Profile_r_m']/1000., old['Vel2_mean'].rolling(window_size, center=True).mean(), color = '#d9544d', alpha = 0.3, label='2014/15 unfiltered mean')
ax2.fill_between(old['Distance_Profile_r_m']/1000., 
                 old['Vel2_mean'].rolling(window_size, center=True).mean()-old['Vel2_stddev'].rolling(window_size, center=True).mean(), 
                 old['Vel2_mean'].rolling(window_size, center=True).mean()+old['Vel2_stddev'].rolling(window_size, center=True).mean(), color='#d58a94', label='2014/15 unfiltered std.dev.', alpha = 0.3)
ax2.plot(old['Distance_Profile_r_m']/1000., old['Vel3_mean'].rolling(window_size, center=True).mean(), color = '#607c8e', alpha = 0.3, label='2015/17 unfiltered mean')
ax2.fill_between(old['Distance_Profile_r_m']/1000., 
                 old['Vel3_mean'].rolling(window_size, center=True).mean()-old['Vel3_stddev'].rolling(window_size, center=True).mean(), 
                 old['Vel3_mean'].rolling(window_size, center=True).mean()+old['Vel3_stddev'].rolling(window_size, center=True).mean(), color='lightgray', label='2015/17 unfiltered std.dev.', alpha = 0.3)



#plot new
ax2.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel1_mean'].rolling(window_size, center=True).mean(),  'g-',label='2013/14 mean')
ax2.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel1_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel1_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel1_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel1_stddev'].rolling(window_size, center=True).mean(), color='palegreen', label='2013/14  std.dev.')
ax2.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel2_mean'].rolling(window_size, center=True).mean(), 'r-', label='2014/15 mean')
ax2.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel2_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel2_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel2_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel2_stddev'].rolling(window_size, center=True).mean(), color='salmon', label='2014/15 std.dev.')
ax2.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel3_mean'].rolling(window_size, center=True).mean(), 'b-', label='2015/17 mean')
ax2.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel3_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel3_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel3_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel3_stddev'].rolling(window_size, center=True).mean(), color='lightblue', label='2015/17 std.dev.')
ax2.set_xlabel('Distance along Profile (km)')
ax2.legend(prop={'size': 5}, loc = 1, ncol=2)
ax2.set_ylabel('Velocity (m/yr)')
ax2.grid()
ax2.set_xlim([60, 0])
ax2.set_ylim([0, 700])

ax3 = plt.subplot(313)

ax3.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel1_mean'].rolling(window_size, center=True).mean(),  'g-',label='2013/14 mean')
ax3.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel1_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel1_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel1_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel1_stddev'].rolling(window_size, center=True).mean(), color='palegreen', label='2013/14  std.dev.')
ax3.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel2_mean'].rolling(window_size, center=True).mean(), 'r-', label='2014/15 mean')
ax3.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel2_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel2_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel2_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel2_stddev'].rolling(window_size, center=True).mean(), color='salmon', label='2014/15 std.dev.')
ax3.plot(profile_pd['Distance_Profile_r_m']/1000., profile_pd['Vel3_mean'].rolling(window_size, center=True).mean(), 'b-', label='2015/17 mean')
ax3.fill_between(profile_pd['Distance_Profile_r_m']/1000., 
                 profile_pd['Vel3_mean'].rolling(window_size, center=True).mean()-profile_pd['Vel3_stddev'].rolling(window_size, center=True).mean(), 
                 profile_pd['Vel3_mean'].rolling(window_size, center=True).mean()+profile_pd['Vel3_stddev'].rolling(window_size, center=True).mean(), color='lightblue', label='2015/17 std.dev.')
ax3.set_xlabel('Distance along Profile (km)')
ax3.legend(prop={'size': 5}, loc = 1)
ax3.set_ylabel('Velocity (m/yr)')
ax3.grid()
ax3.set_xlim([40, 0])
ax3.set_ylim([0, 260])


plt.tight_layout()
plt.savefig("profile.png", dpi = 400)
plt.show()
