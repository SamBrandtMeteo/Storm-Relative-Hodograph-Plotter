#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:14:04 2022

@author: sambrandt
"""
###############################################################################
# Sam Brandt's Storm-Relative Hodograph Plotter v2.0
###############################################################################

# Modules #

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from siphon.simplewebservice.igra2 import IGRAUpperAir
from siphon.ncss import NCSS
#from siphon.catalog import TDSCatalog

###############################################################################

# Inputs #

# Choose data source
#datasource='IGRA' # Observed soundings from IGRA database   
#datasource='NWP' # Model reanalysis from RAP/RUC
datasource='Manual'


if datasource=='IGRA':
    # Choose time
    year=2010
    month=10
    day=6
    hour=12 # in UTC
    # Choose station
    station='USM00072376'
        # Comprehensive list of station numbers can be found at this link: 
        # https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt
if datasource=='NWP':
    
    #nwp_model='RAP/RUC'
    nwp_model='RAP/RUC'
    
    # Choose date
    year  = '2018'
    month = '10'
    day   = '02'
    hour  = '2100' # in UTC
    fcst  = '000' # forecast hour (keep it '000' if you want initialization)
    # Choose center latitude/longitude
    center_lat = 41.17
    center_lon = -80
    # Area to average the profile over
    north_lat = center_lat + 1.
    south_lat = center_lat - 1.
    west_lon = center_lon - 1.
    east_lon = center_lon + 1.
    
if datasource=='Manual':
    u=np.array([-1,5,9.5,20,27.5,32.5,36,36.5,37.5,42,50])/1.944
    v=np.array([9,16,20.1,22,18.5,12,11.5,9,6,5.5,1])/1.944
    z=np.array([0,100,250,500,1000,2000,3000,4000,5000,6000,7000])
    titletext='Manual Input Based on KEVX and KEOX VWPs'


# Resolution
# Resolution to which the sounding will be interpolated (in meters)
# Must be <500, *and* evenly divisible into 500. 100 is HIGHLY recommended
resolution=100

# Inflow layer depth
# The depth over which streamwise percentages will be plotted 
# 3000 meters is default
# MUST be an integer
inflow=1000

# Storm motion
move='right' # Right-moving supercell motion via Bunkers method
#move='left' # Left-moving supercell motion via Bunkers method
#move='mean' # 0-6 km mean wind as storm motion
#move='observed' # Custom storm motion
if move=='observed':
    # Direction measured counterclockwise from East, in degrees
    direction=68
    # Speed in meters per second
    speed=12

# Deviant tornado motion toggle. If set to true, will display DTM vector
deviant_tornado=True

# Color mode
mode='light'
#mode='dark'

hodotype='streamwise'

###############################################################################

# Functions #

# Interpolation function
# Takes in an array of what you want to interpolate, an array of height, and
# the desired layer depth
def interpolate(var,hgts,step):
    levels=np.arange(0,np.max(hgts),step)
    varinterp=np.zeros(len(levels))
    for i in range(0,len(levels)):
        lower=np.where(hgts-levels[i]<=0,hgts-levels[i],-np.inf).argmax()
        varinterp[i]=(((var[lower+1]-var[lower])/(hgts[lower+1]-hgts[lower]))*(levels[i]-hgts[lower])+var[lower])
    return varinterp 
# Bunkers right mover function
# Takes in arrays of u,v, and height, and calculates the motion of a
# right-moving supercell following the Bunkers method
def bunkers(u,v,z,mover):
    mwu=np.mean(u[z<=6000])
    mwv=np.mean(v[z<=6000])
    ulow=(u[z==0]+u[z==500])/2
    uupp=(u[z==5500]+u[z==6000])/2
    vlow=(v[z==0]+v[z==500])/2
    vupp=(v[z==5500]+v[z==6000])/2
    if uupp>ulow:
        theta=np.arctan((vupp-vlow)/(uupp-ulow))-np.pi/2
    elif uupp<ulow:
        theta=np.arctan((vupp-vlow)/(uupp-ulow))+np.pi/2
    rmu=7.5*np.cos(theta)+mwu
    rmv=7.5*np.sin(theta)+mwv
    lmu=-7.5*np.cos(theta)+mwu
    lmv=-7.5*np.sin(theta)+mwv
    if mover=='right':
        return rmu,rmv
    elif mover=='left':
        return lmu,lmv
    elif mover=='mean':
        return [mwu],[mwv]
# Note: this function is one of the reasons the resolution has to evenly
# divide 500

###############################################################################

# RAP/RUC Download
# This section's code adapted from Cameron Nixon's hodograph plotter

if datasource=='NWP':
    if nwp_model=='RAP/RUC':
        modeltextcode=['rap130anl','rap_130','ruc130anl','ruc2anl_130']
    elif nwp_model=='GFS_fcst':
        modeltextcode=['gfs-003-files','gfs_3','gfs-003-files','gfs_3']
    for attempt in range(5):
        try:
            try:
                # Downloads RAP data
                try:
                    data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-'+modeltextcode[0]+'/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/'+modeltextcode[1]+'_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                except:
                    data = NCSS('https://www.ncdc.noaa.gov/thredds/ncss/model-rap130anl-old/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/rap_130_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                # Initiates query
                query = data.query()
                # Subsets by data
                query.variables('Geopotential_height_isobaric','u-component_of_wind_isobaric', 'v-component_of_wind_isobaric','Geopotential_height_surface','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Pressure_surface','Relative_humidity_isobaric','Temperature_isobaric').add_lonlat()
                # Subsets by area
                query.lonlat_box(west_lon,east_lon,south_lat,north_lat)
                # Gets data
                data_RAP = data.get_data(query)
                # Defines data
                z=np.mean(data_RAP.variables['Geopotential_height_isobaric'][0,:,:,:],axis=(1,2))
                u=np.mean(data_RAP.variables['u-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                v=np.mean(data_RAP.variables['v-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                RH=np.mean(data_RAP.variables['Relative_humidity_isobaric'][0,:,:,:],axis=(1,2))/100
                T=np.mean(data_RAP.variables['Temperature_isobaric'][0,:,:,:],axis=(1,2))
                zsfc=np.mean(data_RAP.variables['Geopotential_height_surface'][0,:,:])
                usfc=np.mean(data_RAP.variables['u-component_of_wind_height_above_ground'][0,0,:,:])
                vsfc=np.mean(data_RAP.variables['v-component_of_wind_height_above_ground'][0,0,:,:])
                psfc=np.mean(data_RAP.variables['Pressure_surface'][0,:,:])
                z=np.flip(z)
                u=np.flip(u)
                v=np.flip(v)
                RH=np.flip(RH)
                T=np.flip(T)
                u=u[z>=zsfc]
                v=v[z>=zsfc]
                RH=RH[z>=zsfc]
                T=T[z>=zsfc]
                z=z[z>=zsfc]
                u=np.insert(u,0,usfc)
                v=np.insert(v,0,vsfc)
                RH=np.insert(RH,0,RH[0])
                T=np.insert(T,0,T[0])
                z=np.insert(z,0,zsfc)
                z=z-z[0]
                # Interpolation
                u=interpolate(u,z,resolution)
                v=interpolate(v,z,resolution)
                RH=interpolate(RH,z,resolution)
                T=interpolate(T,z,resolution)
                z=interpolate(z,z,resolution)
                T[0]=2*T[1]-T[2]
                # Close query
                data_RAP.close()
                # Model definition
                model = 'RAP'
                # Above framework is repeated multiple times below:
            except:
                try:   
                    try:
                        data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-'+modeltextcode[2]+'/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/'+modeltextcode[3]+'_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                    except:
                        data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-ruc130anl-old/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/ruc2anl_130_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                    query = data.query()
                    query.variables('Geopotential_height_isobaric','u-component_of_wind_isobaric', 'v-component_of_wind_isobaric','Geopotential_height_surface','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Pressure_surface','Relative_humidity_isobaric','Temperature_isobaric').add_lonlat()
                    query.lonlat_box(west_lon,east_lon,south_lat,north_lat)
                    data_RUC = data.get_data(query)
                    z=np.mean(data_RUC.variables['Geopotential_height_isobaric'][0,:,:,:],axis=(1,2))
                    u=np.mean(data_RUC.variables['u-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                    v=np.mean(data_RUC.variables['v-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                    RH=np.mean(data_RUC.variables['Relative_humidity_isobaric'][0,:,:,:],axis=(1,2))/100
                    T=np.mean(data_RUC.variables['Temperature_isobaric'][0,:,:,:],axis=(1,2))
                    zsfc=np.mean(data_RUC.variables['Geopotential_height_surface'][0,:,:])
                    usfc=np.mean(data_RUC.variables['u-component_of_wind_height_above_ground'][0,0,:,:])
                    vsfc=np.mean(data_RUC.variables['v-component_of_wind_height_above_ground'][0,0,:,:])
                    psfc=np.mean(data_RUC.variables['Pressure_surface'][0,:,:])
                    z=np.flip(z)
                    u=np.flip(u)
                    v=np.flip(v)
                    RH=np.flip(RH)
                    T=np.flip(T)
                    u=u[z>=zsfc]
                    v=v[z>=zsfc]
                    RH=RH[z>=zsfc]
                    T=T[z>=zsfc]
                    z=z[z>=zsfc]
                    u=np.insert(u,0,usfc)
                    v=np.insert(v,0,vsfc)
                    RH=np.insert(RH,0,RH[0])
                    T=np.insert(T,0,T[0])
                    z=np.insert(z,0,zsfc)
                    z=z-z[0]
                    # Interpolation
                    u=interpolate(u,z,resolution)
                    v=interpolate(v,z,resolution)
                    RH=interpolate(RH,z,resolution)
                    T=interpolate(T,z,resolution)
                    z=interpolate(z,z,resolution)
                    T[0]=2*T[1]-T[2]
                    data_RUC.close()
                    model = 'RUC'
                except:
                    try:
                        try:
                            data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-'+modeltextcode[0]+'/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/'+modeltextcode[1]+'_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                        except:
                            data = NCSS('https://www.ncdc.noaa.gov/thredds/ncss/model-rap130anl-old/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/rap_130_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                        query = data.query()
                        query.variables('Geopotential_height_isobaric','u-component_of_wind_isobaric', 'v-component_of_wind_isobaric','Geopotential_height_surface','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Pressure_surface','Relative_humidity_isobaric','Temperature_isobaric').add_lonlat()
                        query.lonlat_box(west_lon,east_lon,south_lat,north_lat)
                        data_RAP = data.get_data(query)
                        z=np.mean(data_RAP.variables['Geopotential_height_isobaric'][0,:,:,:],axis=(1,2))
                        u=np.mean(data_RAP.variables['u-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                        v=np.mean(data_RAP.variables['v-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                        RH=np.mean(data_RAP.variables['Relative_humidity_isobaric'][0,:,:,:],axis=(1,2))/100
                        T=np.mean(data_RAP.variables['Temperature_isobaric'][0,:,:,:],axis=(1,2))
                        zsfc=np.mean(data_RAP.variables['Geopotential_height_surface'][0,:,:])
                        usfc=np.mean(data_RAP.variables['u-component_of_wind_height_above_ground'][0,0,:,:])
                        vsfc=np.mean(data_RAP.variables['v-component_of_wind_height_above_ground'][0,0,:,:])
                        psfc=np.mean(data_RAP.variables['Pressure_surface'][0,:,:])
                        z=np.flip(z)
                        u=np.flip(u)
                        v=np.flip(v)
                        RH=np.flip(RH)
                        T=np.flip(T)
                        u=u[z>=zsfc]
                        v=v[z>=zsfc]
                        RH=RH[z>=zsfc]
                        T=T[z>=zsfc]
                        z=z[z>=zsfc]
                        u=np.insert(u,0,usfc)
                        v=np.insert(v,0,vsfc)
                        RH=np.insert(RH,0,RH[0])
                        T=np.insert(T,0,T[0])
                        z=np.insert(z,0,zsfc)
                        z=z-z[0]
                        # Interpolation
                        u=interpolate(u,z,resolution)
                        v=interpolate(v,z,resolution)
                        RH=interpolate(RH,z,resolution)
                        T=interpolate(T,z,resolution)
                        z=interpolate(z,z,resolution)
                        T[0]=2*T[1]-T[2]
                        data_RAP.close()
                        model = 'RAP'
                    except:
                        try:
                            data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-'+modeltextcode[2]+'/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/'+modeltextcode[3]+'_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                        except:
                            data = NCSS('https://www.ncei.noaa.gov/thredds/ncss/model-ruc130anl-old/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/ruc2anl_130_'+str(year)+str(month)+str(day)+'_'+str(hour)+'_'+str(fcst)+'.grb2')
                        query = data.query()
                        query.variables('Geopotential_height_isobaric','u-component_of_wind_isobaric', 'v-component_of_wind_isobaric','Geopotential_height_surface','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Pressure_surface','Relative_humidity_isobaric','Temperature_isobaric').add_lonlat()
                        query.lonlat_box(west_lon,east_lon,south_lat,north_lat)
                        data_RUC = data.get_data(query)
                        z=np.mean(data_RUC.variables['Geopotential_height_isobaric'][0,:,:,:],axis=(1,2))
                        u=np.mean(data_RUC.variables['u-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                        v=np.mean(data_RUC.variables['v-component_of_wind_isobaric'][0,:,:,:],axis=(1,2))
                        RH=np.mean(data_RUC.variables['Relative_humidity_isobaric'][0,:,:,:],axis=(1,2))/100
                        T=np.mean(data_RUC.variables['Temperature_isobaric'][0,:,:,:],axis=(1,2))
                        zsfc=np.mean(data_RUC.variables['Geopotential_height_surface'][0,:,:])
                        usfc=np.mean(data_RUC.variables['u-component_of_wind_height_above_ground'][0,0,:,:])
                        vsfc=np.mean(data_RUC.variables['v-component_of_wind_height_above_ground'][0,0,:,:])
                        psfc=np.mean(data_RUC.variables['Pressure_surface'][0,:,:])
                        z=np.flip(z)
                        u=np.flip(u)
                        v=np.flip(v)
                        RH=np.flip(RH)
                        T=np.flip(T)
                        u=u[z>=zsfc]
                        v=v[z>=zsfc]
                        RH=RH[z>=zsfc]
                        T=T[z>=zsfc]
                        z=z[z>=zsfc]
                        u=np.insert(u,0,usfc)
                        v=np.insert(v,0,vsfc)
                        RH=np.insert(RH,0,RH[0])
                        T=np.insert(T,0,T[0])
                        z=np.insert(z,0,zsfc)
                        z=z-z[0]
                        # Interpolation
                        u=interpolate(u,z,resolution)
                        v=interpolate(v,z,resolution)
                        RH=interpolate(RH,z,resolution)
                        T=interpolate(T,z,resolution)
                        z=interpolate(z,z,resolution)
                        T[0]=2*T[1]-T[2]
                        data_RUC.close()
                        model = 'RUC'
        except:
            print('server failure (UCAR)')
        else:
            break
    else:
        print('server down (UCAR)')
    # Define plot title
    #titletext=model+' at '+str(center_lat)+'°N '+str(center_lon)+'°E: '+str(month)+'/'+str(day)+'/'+str(year)+' valid '+str(hour)+'z'
    #if nwp_model=='GFS_fcst':
    #    titletext='GFS at '+str(center_lat)+'°N '+str(center_lon)+'°E: '+str(month)+'/'+str(day)+'/'+str(year)+' Init '+str(hour)+'z FH '+str(fcst)
    titletext=model+' at '+str(center_lat)+'°N '+str(center_lon)+'°E: '+str(month)+'/'+str(day)+'/'+str(year)+' Init '+str(hour)+'z FH '+str(fcst)
if datasource=='Manual':
    u=interpolate(u,z,resolution)
    v=interpolate(v,z,resolution)
    z=interpolate(z,z,resolution)
###############################################################################

# Storm Motion Logic #

# Storm motion calculations
if move!='observed':
    smu,smv=bunkers(u,v,z,move)
elif move=='observed':
    smu=[speed*np.cos(direction*(np.pi/180))]
    smv=[speed*np.sin(direction*(np.pi/180))]
# Calculate storm-relative winds
uSR=u-smu
vSR=v-smv
# Storm motion logic for plot text
if move=='right':
    movetext='RM'
elif move=='left':
    movetext='LM'
elif move=='mean':
    movetext='MW'
elif move=='observed':
    movetext=str(direction)+'/'+str(speed)
# Deviant tornado motion
uDTM=(np.mean(u[z<=500])+smu)/2
vDTM=(np.mean(v[z<=500])+smv)/2
    
###############################################################################

# Shear Calculations #

# Shear components
dudz=(u[2::]-u[0:-2])/(z[2::]-z[0:-2])
dvdz=(v[2::]-v[0:-2])/(z[2::]-z[0:-2])
dudz=np.insert(dudz,0,dudz[0])
dudz=np.insert(dudz,-1,dudz[-1])
dvdz=np.insert(dvdz,0,dvdz[0])
dvdz=np.insert(dvdz,-1,dvdz[-1])
# Shear magnitude
shear=np.sqrt(dudz**2+dvdz**2)+0.0000001
# Vorticity components
uvort=-dvdz
vvort=dudz
# Streamwise vorticity
strmw=abs((uSR*uvort+vSR*vvort)/(np.sqrt(uSR**2+vSR**2)))
# Streamwise fraction
strmwperc=strmw/shear

###############################################################################

# Color Schemes #

# Streamwise colormap (This shouldn't be changed, cool is very well suited)

# Light vs dark mode color schemes
if mode=='light':
    # [Background,spines/text/heightlabel,rings/axis/heightdot/smvector/shearlegend,freetrophodo,stormmotiontext]
    colors=['white','gray','lightgray','darkgray',[0.3,0.3,0.3]]
elif mode=='dark':
    colors=['black','darkgray','whitesmoke','lightgray','gray']

###############################################################################

# Plot #

# Define axis
fig,ax=plt.subplots(dpi=500,facecolor=colors[0])
# Set axis limits, ticks, and tick labels
ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
ax.set_aspect('equal')
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.set_xticks([])
ax.set_yticks([])
# Set plot border color
ax.spines['bottom'].set_color(colors[1])
ax.spines['top'].set_color(colors[1]) 
ax.spines['right'].set_color(colors[1])
ax.spines['left'].set_color(colors[1])
# Axis background color
ax.set_facecolor(colors[0])
# Rings
for i in range(0,8):
    if i==1:
        ax.plot((5+5*i)*np.cos(np.arange(0,2*np.pi+np.pi/64,np.pi/64)),(5+5*i)*np.sin(np.arange(0,2*np.pi+np.pi/64,np.pi/64)),color=colors[2],lw=0.5,linestyle='solid')
    elif i!=1:
        ax.plot((5+5*i)*np.cos(np.arange(0,2*np.pi+np.pi/64,np.pi/64)),(5+5*i)*np.sin(np.arange(0,2*np.pi+np.pi/64,np.pi/64)),color=colors[2],lw=0.5,linestyle='dashed')
# Ring labels
for i in range(0,5):
    ax.text(0.5,-5.5-5*i,str(5+5*i)+' m/s',ha='left',va='top',fontsize=4,color=colors[1])
# Cartesian Axes
ax.plot([0,0],[-40,40],lw=0.8,color=colors[2])
ax.plot([-40,40],[0,0],lw=0.8,color=colors[2])
# Inflow storm-relative wind threshold for supercells (Peters et al. 2020)
#ax.text(-7,-7,'Peters et al.\nSupercell',ha='center',va='center',fontsize=3,color=colors[1])


if hodotype=='streamwise':
    cmap=plt.cm.cool
    # Plot "Inflow" layer
    for i in range(0,int(inflow/resolution)):
        ax.plot(uSR[i:i+2],vSR[i:i+2],lw=180*shear[i],color=cmap(int(strmwperc[i]*255)))  
    # Plot "Free troposphere" layer

    if z[-1]<=10000:
        for i in range(int(inflow/resolution),int((round(z[-1],-3)-1000)/resolution)):
            ax.plot(uSR[i:i+2],vSR[i:i+2],lw=180*shear[i],color=colors[3])
        upper=int((round(z[-1],-3)/1000))
    elif z[-1]>10000:
        for i in range(int(inflow/resolution),int(10000/resolution)):
            ax.plot(uSR[i:i+2],vSR[i:i+2],lw=180*shear[i],color=colors[3])
        upper=11

    
# Hodograph height labels
for i in range(0,upper):
    ax.text(uSR[z==1000*i],vSR[z==1000*i]+0.5,str(i),ha='center',va='bottom',fontsize=4,color=colors[1])
    ax.scatter(uSR[z==1000*i],vSR[z==1000*i],color=colors[2],s=4,zorder=3)
ax.text(uSR[z==500],vSR[z==500]+0.5,'0.5',ha='center',va='bottom',fontsize=4,color=colors[1])
ax.scatter(uSR[z==500],vSR[z==500],color=colors[2],s=4,zorder=3)
# Ground-relative storm motion vector
ax.quiver(0,0,smu[0],smv[0],color=colors[2],units='xy',scale=1,scale_units='xy',headwidth=6,headlength=8,width=0.4,zorder=3)
ax.text(smu[0],smv[0],movetext,color=colors[4],fontsize=5,ha='center',va='center',zorder=4)
if deviant_tornado==True:
    ax.quiver(0,0,uDTM[0],vDTM[0],color=colors[2],units='xy',scale=1,scale_units='xy',headwidth=6,headlength=8,width=0.4,zorder=3)
    ax.text(uDTM[0],vDTM[0],'DTM',color=colors[4],fontsize=5,ha='center',va='center',zorder=4)

if hodotype=='streamwise':
    # Shear magnitude graphic
    for i in range(0,4):
        ax.plot([-26+5*i,-24+5*i],[-26,-26],color=colors[2],lw=180*(0.01+0.01*i))
    # Shear magnitude labels
    for i in range(0,4):
        ax.text(-25+5*i,-28,str(0.01+0.01*i),fontsize=3,color=colors[1],ha='center',va='top')  
    # Shear magnitude title
    ax.text(-17.5,-23.5,'Shear Magnitude (1/s)',fontsize=4,color=colors[1],ha='center')
    # Streamwise colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0,vmax=1))
    cbar=plt.colorbar(sm,orientation='horizontal',shrink=0.5,pad=0.03)
    # Streamwise colorbar labels
    cbar.ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'],fontsize=6,color=colors[1])
    cbar.set_label('Streamwise Percentage', fontsize=7,color=colors[1])
    
    cbar.outline.set_color(colors[1])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=colors[1])
    
# Title
ax.text(0,36.2,'Storm-Relative Hodograph',fontsize=7,ha='center',va='center',color=colors[1])
ax.text(29,-29,'Developed by Sam Brandt (v2.0)',fontsize=2,color=colors[1],va='center',ha='right')
ax.set_title(titletext,color=colors[1],fontsize=7)

###############################################################################




