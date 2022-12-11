#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:35:37 2022

@author: sambrandt
"""
###############################################################################
# Sam Brandt's Storm-Relative Hodograph Plotter - Operational Version
###############################################################################

# Modules #

# These modules are needed to make this code run. If this part is getting you
# stuck, this documentation may be able to help: 
# https://docs.python.org/3/installing/index.html
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
# numpy, matplotlib, pandas, datetime, and siphon needed

###############################################################################

# Accessibility #

# Color Scheme
colscheme='default' # Default cyan/magenta color scheme
#colscheme='colorblind' # Colorblind-friendly blue/reddish-orange color scheme

###############################################################################

# Inputs #

# Choose data source
datasource='vad' # Real-time radar-derived vertical wind profiles
#datasource='Manual' # Manually inputted data
#datasource='csv' # Data from .csv file on local Machine

if datasource=='vad':
    rid='kpoe' # String or list of strings of radar IDs
    weights=[0.2,0.8] # Weights used to average multiple radar sites
                        # WEIGHTS MUST ADD UP TO ONE
    scans=6 # Number of recent scans to average. Recommended to choose the
             # smallest number that yields a "smoother" looking hodograph
             
if datasource=='Manual':
    # {u,v,z} must be in standard SI units of {m/s,m/s,m} respectively
    u=np.array([-1.5,-1.5,2,7,15,22,44,70,85])/1.944
    v=np.array([10,32,37.5,39,38,35,31,26,28])/1.944
    z=np.array([0,500,750,1000,2000,3000,6000,9000,16000])
    titletext='Sample Manual Input' # Text to display as the plot title
    
if datasource=='csv':
    path='Your directory here/'
    file='Your filename here.csv'
    df=pd.read_csv(path+file)
    u=np.array(df['u'])/1.944
    v=np.array(df['v'])/1.944
    z=np.array(df['z'])
    # u,v,z are data column names in the csv file
    
# Custom Surface wind
# Adding a custom surface wind based on observations is highly recommended
custom_sfc_wind=False # Set to true to include a custom surface wind
directionsfc=135 # Direction measured counterclockwise from East, in degrees
speedsfc=2 # Speed in meters per second

# Storm motion
move='right' # Right-moving supercell motion (via Bunkers et al. 2000)
#move='left' # Left-moving supercell motion (via Bunkers et al. 2000)
#move='mean' # Mean wind over a specified depth as storm motion
#move='observed' # Custom storm motion

if move=='mean':
    maxsteer=6000 # Maximum height used to calculate mean wind (meters)
                  # 6000 is default following Bunkers et al. 2000
    
if move=='observed':
    direction=45 # Direction measured counterclockwise from East, in degrees
    speed=10 # Speed in meters per second

# Deviant tornado motion (following Nixon & Allen 2021) toggle. 
# If set to true, DTM vector will be displayed in addition to storm motion
deviant_tornado=False

# Layer
# The depth over which streamwise percentages will be plotted 
# 3000 meters is default
# MUST be an integer
layer=1000
# For tornado forecasting purposes, streamwise vorticity in the lowest few
# hundred meters is the most relevant. For other aspects of supercell dynamics,
# other layers will be more relevant

resolution=100 # Resolution to which the sounding will be interpolated (in m)
               # DO NOT CHANGE
               
###############################################################################  

# Output #          
path='Your directory here/' # Directory to run the script in. This is also the
                           # directory that the images will be saved to

###############################################################################

# Functions #

# Interpolation function
# Takes in an array of the variable you want to interpolate, 
# an array of height, and the interpolation step
# Outputs an array of the variable interpolated in height to the given step
def interpolate(var,hgts,step):
    levels=np.arange(0,np.max(hgts),step)
    varinterp=np.zeros(len(levels))
    for i in range(0,len(levels)):
        lower=np.where(hgts-levels[i]<=0,hgts-levels[i],-np.inf).argmax()
        varinterp[i]=(((var[lower+1]-var[lower])/(hgts[lower+1]-hgts[lower]))*(levels[i]-hgts[lower])+var[lower])
    return varinterp 

# Bunkers supercell motion function
# Takes in arrays of u, v, and height
# Calculates the motion of a right-moving supercell following the method
# outlined in Bunkers et al. 2000
def bunkers(u,v,z,mover):
    prop=7.5
    if mover=='mean':
        upper=maxsteer
        mwu=np.mean(u[z<=upper])
        mwv=np.mean(v[z<=upper])
        return [mwu],[mwv]
    else:
        upper=6000
        mwu=np.mean(u[z<=upper])
        mwv=np.mean(v[z<=upper])
        ulow=(u[z==0]+u[z==500])/2
        uupp=(u[z==upper-500]+u[z==upper])/2
        vlow=(v[z==0]+v[z==500])/2
        vupp=(v[z==upper-500]+v[z==upper])/2
        if uupp>ulow:
            theta=np.arctan((vupp-vlow)/(uupp-ulow))-np.pi/2
        elif uupp<ulow:
            theta=np.arctan((vupp-vlow)/(uupp-ulow))+np.pi/2
        try:
            rmu=prop*np.cos(theta)+mwu
            rmv=prop*np.sin(theta)+mwv
            lmu=-prop*np.cos(theta)+mwu
            lmv=-prop*np.sin(theta)+mwv
            if mover=='right':
                return rmu,rmv
            elif mover=='left':
                return lmu,lmv
        except UnboundLocalError:
            print('\n\n\nERROR: NOT ENOUGH DATA TO CALCULATE BUNKERS STORM MOTION\n\n\n')
            from sys import exit
            exit()

    
###############################################################################

# VAD Download #
# This section through line 466 is NOT my code (if you couldn't already tell 
# by its superior quality). HUGE shoutout to Tim Supinie's GitHub (tsupinie)

import struct

try:
    from urllib.request import urlopen, URLError
except ImportError:
    from urllib2 import urlopen, URLError

try:
    from io import BytesIO
except ImportError:
    from BytesIO import BytesIO

import re

_base_url = "ftp://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/DS.48vwp/"

class VADFile(object):
    fields = ['wind_dir', 'wind_spd', 'rms_error', 'divergence', 'slant_range', 'elev_angle']

    def __init__(self, file):
        self._rpg = file
        self._data = None

        self._read_headers()
        has_symbology_block, has_graphic_block, has_tabular_block = self._read_product_description_block()

        if has_symbology_block:
            self._read_product_symbology_block()

        if has_graphic_block:
            pass

        if has_tabular_block:
            self._read_tabular_block()

        self._data = self._get_data()
        return

    def _read_headers(self):
        wmo_header = self._read('s30')

        message_code = self._read('h')
        message_date = self._read('h')
        message_time = self._read('i')
        message_length = self._read('i')
        source_id = self._read('h')
        dest_id = self._read('h')
        num_blocks = self._read('h')

        return

    def _read_product_description_block(self):
        self._read('h') # Block separator
        self._radar_latitude  = self._read('i') / 1000.
        self._radar_longitude = self._read('i') / 1000.
        self._radar_elevation = self._read('h')

        product_code = self._read('h')
        if product_code != 48:
            raise IOError("This isn't a VWP file.")

        operational_mode    = self._read('h')
        self._vcp           = self._read('h')
        req_sequence_number = self._read('h')
        vol_sequence_number = self._read('h')

        scan_date    = self._read('h')
        scan_time    = self._read('i')
        product_date = self._read('h')
        product_time = self._read('i')

        self._read('h')   # Product-dependent variable 1 (unused)
        self._read('h')   # Product-dependent variable 2 (unused)
        self._read('h')   # Elevation (unused)
        self._read('h')   # Product-dependent variable 3 (unused)
        self._read('16h') # Product-dependent thresholds (how do I interpret these?)
        self._read('7h')  # Product-dependent variables 4-10 (mostly unused ... do I need the max?)

        version    = self._read('b')
        spot_blank = self._read('b')

        offset_symbology = self._read('i')
        offset_graphic   = self._read('i')
        offset_tabular   = self._read('i')

        self._time = datetime(1969, 12, 31, 0, 0, 0) + timedelta(days=scan_date, seconds=scan_time)

        return offset_symbology > 0, offset_graphic > 0, offset_tabular > 0

    def _read_product_symbology_block(self):
        self._read('h') # Block separator
        block_id = self._read('h')

        if block_id != 1:
            raise IOError("This isn't the product symbology block.")

        block_length    = self._read('i')
        num_layers      = self._read('h')
        layer_separator = self._read('h')
        layer_num_bytes = self._read('i')
        block_data      = self._read('%dh' % int(layer_num_bytes / struct.calcsize('h')))

        packet_code = -1
        packet_size = -1
        packet_counter = -1
        packet_value = -1
        packet = []
        for item in block_data:
            if packet_code == -1:
                packet_code = item
            elif packet_size == -1:
                packet_size = item
                packet_counter = 0
            elif packet_value == -1:
                packet_value = item
                packet_counter += struct.calcsize('h')
            else:
                packet.append(item)
                packet_counter += struct.calcsize('h')

                if packet_counter == packet_size:
                    if packet_code == 8:
                        str_data = struct.pack('>%dh' % int(packet_size / struct.calcsize('h') - 3), *packet[2:])
                    elif packet_code == 4:
                        pass

                    packet = []
                    packet_code = -1
                    packet_size = -1
                    packet_counter = -1
                    packet_value = -1
        return

    def _read_tabular_block(self):
        self._read('h')
        block_id = self._read('h')
        if block_id != 3:
            raise IOError("This isn't the tabular block.")

        block_size = self._read('i')

        self._read('h')
        self._read('h')
        self._read('i')
        self._read('i')
        self._read('h')
        self._read('h')
        self._read('h')

        self._read('h')
        self._read('i')
        self._read('i')
        self._read('h')
        product_code = self._read('h')

        operational_mode    = self._read('h')
        vcp                 = self._read('h')
        req_sequence_number = self._read('h')
        vol_sequence_number = self._read('h')

        scan_date    = self._read('h')
        scan_time    = self._read('i')
        product_date = self._read('h')
        product_time = self._read('i')

        self._read('h')   # Product-dependent variable 1 (unused)
        self._read('h')   # Product-dependent variable 2 (unused)
        self._read('h')   # Elevation (unused)
        self._read('h')   # Product-dependent variable 3 (unused)
        self._read('16h') # Product-dependent thresholds (how do I interpret these?)
        self._read('7h')  # Product-dependent variables 4-10 (mostly unused ... do I need the max?)

        version    = self._read('b')
        spot_blank = self._read('b')

        offset_symbology = self._read('i')
        offset_graphic   = self._read('i')
        offset_tabular   = self._read('i')

        self._read('h') # Block separator
        num_pages = self._read('h')
        self._text_message = []
        for idx in range(num_pages):
            num_chars = self._read('h')
            self._text_message.append([])
            while num_chars != -1:
                self._text_message[-1].append(self._read("s%d" % num_chars))
                num_chars = self._read('h')

        return

    def _read(self, type_string):
        if type_string[0] != 's':
            size = struct.calcsize(type_string)
            data = struct.unpack(">%s" % type_string, self._rpg.read(size))
        else:
            size = int(type_string[1:])
            data = tuple([ self._rpg.read(size).strip(b"\0").decode('utf-8') ])

        if len(data) == 1:
            return data[0]
        else:
            return list(data)

    def _get_data(self):
        vad_list = []
        for page in self._text_message:
            if (page[0].strip())[:20] == "VAD Algorithm Output":
                vad_list.extend(page[3:])

        data = dict((k, []) for k in VADFile.fields)

        for line in vad_list:
            values = line.strip().split()
            data['wind_dir'].append(float(values[4]))
            data['wind_spd'].append(float(values[5]))
            data['rms_error'].append(float(values[6]))
            data['divergence'].append(float(values[7]) if values[7] != 'NA' else np.nan)
            data['slant_range'].append(float(values[8]))
            data['elev_angle'].append(float(values[9]))

        for key, val in data.items():
            data[key] = np.array(val)


        data['slant_range'] *= 6067.1 / 3281.

        r_e = 4. / 3. * 6371
        data['altitude'] = np.sqrt(r_e ** 2 + data['slant_range'] ** 2 + 2 * r_e * data['slant_range'] * np.sin(np.radians(data['elev_angle']))) - r_e

        order = np.argsort(data['altitude'])
        for key, val in data.items():
            data[key] = val[order]
        return data

    def __getitem__(self, key):
        if key == 'time':
            val = self._time
        else:
            val = self._data[key]
        return val

    def add_surface_wind(self, sfc_wind):
        sfc_dir, sfc_spd = sfc_wind

        keys = ['wind_dir', 'wind_spd', 'rms_error', 'altitude']
        vals = [float(sfc_dir), float(sfc_spd), 0., 0.01]

        for key, val in zip(keys, vals):
            self._data[key] = np.append(val, self._data[key])

def find_file_times(rid):
    url = "%s/SI.%s/" % (_base_url, rid.lower())

    file_text = urlopen(url).read().decode('utf-8')
    file_list = re.findall("([\w]{3} [\d]{1,2} [\d]{2}:[\d]{2}) (sn.[\d]{4})", file_text)
    file_times, file_names = list(zip(*file_list))
    file_names = list(file_names)

    year = datetime.utcnow().year
    file_dts = []
    for ft in file_times:
        ft_dt = datetime.strptime("%d %s" % (year, ft), "%Y %b %d %H:%M")
        if ft_dt > datetime.utcnow():
            ft_dt = datetime.strptime("%d %s" % (year - 1, ft), "%Y %b %d %H:%M")

        file_dts.append(ft_dt)

    file_list = list(zip(file_names, file_dts))
    file_list.sort(key=lambda fl: fl[1])

    file_names, file_dts = list(zip(*file_list))
    file_names = list(file_names)

    # The files are only moved into place when the next one is generated, so shift the
    # file names by one index to account for that.
    file_names[:-1] = file_names[1:]
    file_names[-1] = 'sn.last'

    return list(zip(file_names, file_dts))[::-1]

def download_vad(rid, time=None, file_id=None, cache_path=None):
    if time is None:
        if file_id is None:
            url = "%s/SI.%s/sn.last" % (_base_url, rid.lower())
        else:
            url = "%s/SI.%s/sn.%04d" % (_base_url, rid.lower(), file_id)
    else:
        file_name = ""
        for fn, ft in find_file_times(rid):
            if ft <= time:
                file_name = fn
                break

        if file_name == "":
            raise ValueError("No VAD files before %s." % time.strftime("%d %B %Y %H%M UTC"))

        url = "%s/SI.%s/%s" % (_base_url, rid.lower(), file_name)

    try:
        frem = urlopen(url)
    except URLError:
        raise ValueError("Could not find radar site '%s'" % rid.upper())

    if cache_path is None:
        vad = VADFile(frem)
    else:
        bio = BytesIO(frem.read())
        vad = VADFile(bio)

    return vad

_base_url = "ftp://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/DS.48vwp/"

def test_download(rid,scans):
    fnames=find_file_times(rid)
    fname=[]
    drct=[]
    spd=[]
    alt=[]
    timesvalid=[]
    for i in range(1,scans+1):
        fname.append(fnames[i][0])
        url=_base_url+'/SI.'+rid+'/'+fname[i-1]
        frem=urlopen(url)
        vad=VADFile(frem)
        drct.append(270-vad['wind_dir'])
        spd.append(vad['wind_spd']/1.944)
        alt.append(vad['altitude']*1000)
        timesvalid.append(fnames[i][1])
    drct=np.array(drct)
    spd=np.array(spd)
    alt=np.array(alt)
    return drct,spd,alt,timesvalid

def test_download2(rid):
    url=_base_url+'/SI.'+rid+'/sn.last'
    frem=urlopen(url)
    vad=VADFile(frem)
    drct=270-vad['wind_dir']
    spd=vad['wind_spd']/1.944
    alt=vad['altitude']*1000
    timesvalid=find_file_times(rid)[0][1]
    return drct,spd,alt,timesvalid

if datasource=='vad' and isinstance(rid,str):
    rid=rid.lower()
    drct,spd,alt,timesvalid=test_download(rid,scans)
    u=[]
    v=[]
    z=[]
    lengths=[]
    for i in range(0,scans):
        u.append(interpolate(spd[i]*np.cos((np.pi/180)*drct[i]),alt[i],resolution))
        v.append(interpolate(spd[i]*np.sin((np.pi/180)*drct[i]),alt[i],resolution))
        z.append(interpolate(alt[i],alt[i],resolution))
        lengths.append(len(z[i]))
     
    length=np.min(lengths)
    for i in range(0,scans):
        u[i]=u[i][0:length]
        v[i]=v[i][0:length]
        z[i]=z[i][0:length]
        
    u=np.stack(u,axis=0)
    v=np.stack(v,axis=0)
    z=np.stack(z,axis=0)
    u=np.mean(u,axis=0)
    v=np.mean(v,axis=0)
    z=np.mean(z,axis=0)    
    
    titletext=rid.upper()+' VWP: '+str(timesvalid[0].year)+' '+str(timesvalid[scans-1].month)+'/'+str(timesvalid[scans-1].day)+' '+str(timesvalid[scans-1].hour).zfill(2)+':'+str(timesvalid[scans-1].minute).zfill(2)+'z to '+str(timesvalid[0].month)+'/'+str(timesvalid[0].day)+' '+str(timesvalid[0].hour).zfill(2)+':'+str(timesvalid[0].minute).zfill(2)+'z Average'

elif datasource=='vad' and isinstance(rid,list):
    u=[]
    v=[]
    z=[]
    lengths=[]
    times=[]
    rids=[]
    for i in range(0,len(rid)):
        rid[i]=rid[i].lower()
        drct,spd,alt,timesvalid=test_download2(rid[i])
        u.append(interpolate(spd*np.cos((np.pi/180)*drct),alt,resolution))
        v.append(interpolate(spd*np.sin((np.pi/180)*drct),alt,resolution))
        z.append(interpolate(alt,alt,resolution))
        times.append(timesvalid)
        lengths.append(len(z[i]))        
        
    length=np.min(lengths)
    for i in range(0,len(rid)):
        u[i]=u[i][0:length]
        v[i]=v[i][0:length]
        z[i]=z[i][0:length]
        
    u=np.stack(u,axis=0)
    v=np.stack(v,axis=0)
    z=np.stack(z,axis=0)
    
    utemp=[]
    vtemp=[]
    ztemp=[]
    
    for i in range(0,len(rid)):
        utemp.append(weights[i]*u[i,:])
        vtemp.append(weights[i]*v[i,:])
        ztemp.append(weights[i]*z[i,:])

    u=sum(utemp)
    v=sum(vtemp)
    z=sum(ztemp)
    
    titletext=''
    for i in range(0,len(rid)):
        titletext+=str(int(weights[i]*100))+'% '+rid[i].upper()+', '
        if len(titletext)>30:
            titletext+='\n'
    titletext+=str(times[0].year)+' '+str(times[0].month)+'/'+str(times[0].day)+' '+str(times[0].hour).zfill(2)+':'+str(times[0].minute).zfill(2)+'z'
    
###############################################################################

# .csv Download #

if datasource=='csv':
    u=interpolate(u,z,resolution)
    v=interpolate(v,z,resolution)
    z=interpolate(z,z,resolution)
    titletext=file[0:-4]

###############################################################################

# Manually Inputted Data #

if datasource=='Manual':
    u=interpolate(u,z,resolution)
    v=interpolate(v,z,resolution)
    z=interpolate(z,z,resolution)
    
###############################################################################

# Storm Motion Logic #

# Insert custom surface wind
if custom_sfc_wind==True:
    u[0]=speedsfc*np.cos((np.pi/180)*directionsfc)
    v[0]=speedsfc*np.sin((np.pi/180)*directionsfc)

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

# Color Scheme #

# [Background,spines/text/heightlabel,rings/axis/heightdot/smvector/shearlegend
# ,freetrophodo,stormmotiontext]
colors=['white','gray','lightgray','darkgray',[0.3,0.3,0.3]]
# Accessibility
if colscheme=='default':
    cmap=plt.cm.cool
elif colscheme=='colorblind':
    cmap=plt.cm.coolwarm

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

# Plot "Inflow" layer
for i in range(0,int(layer/resolution)):
    try:
        ax.plot(uSR[i:i+2],vSR[i:i+2],lw=180*shear[i],color=cmap(int(strmwperc[i]*255)))
    except IndexError:
        print('\n\n\nERROR: SPECIFIED "LAYER" INPUT IS TOO DEEP FOR THE DATA\n\n\n')
        from sys import exit
        exit()

# Plot "Free troposphere" layer
if z[-1]<=10000:
    for i in range(int(layer/resolution),int((round(z[-1],-3)-1000)/resolution)):
        ax.plot(uSR[i:i+2],vSR[i:i+2],lw=180*shear[i],color=colors[3])
    upper=int((round(z[-1],-3)/1000))
elif z[-1]>10000:
    for i in range(int(layer/resolution),int(10000/resolution)):
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
cbar.ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'],fontsize=6,color=colors[1])
cbar.set_label('Streamwise Percentage', fontsize=7,color=colors[1])
cbar.outline.set_color(colors[1])
cbar.outline.set_visible(False)
cbar.ax.tick_params(color=colors[1])

# Title
ax.text(29,-29,'Developed by Sam Brandt (GitHub: SamBrandtMeteo)',fontsize=2,color=colors[1],va='center',ha='right')
ax.set_title('Storm-Relative Hodograph\n'+titletext,color=colors[1],fontsize=7)

# Output
plt.show()
plt.savefig(path+'storm_relative_hodograph.png')

###############################################################################




