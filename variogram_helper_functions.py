
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import time
import h5py
import re
import hdf5plugin
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import random
import skgstat as skg
from pprint import pprint
import seaborn as sns
import matplotlib.ticker as ticker
import read_vars_WRF_RAMS
from libpysal.weights.distance import DistanceBand
import libpysal 
from esda.moran import Moran
from scipy.ndimage import gaussian_filter
from wrf import smooth2d
from shapely.geometry import Point, Polygon
import cartopy
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER



def print_bounding_box_RAMS_file(FILE):
    print('printing bounding box in (lon,lat) format with southwest corner, then southeast, then northeast, and finally northwest...')
    ds = xr.open_dataset(FILE,engine='h5netcdf',phony_dims='sort')

    lat_min = ds.GLAT.min().values
    lat_max = ds.GLAT.max().values
    lon_min = ds.GLON.min().values
    lon_max = ds.GLON.max().values

    print('('+str(lon_min)+','+str(lat_min)+'),','('+str(lon_max)+','+str(lat_min)+'),',\
          '('+str(lon_max)+','+str(lat_max)+'),','('+str(lon_min)+','+str(lat_max)+'),')

def find_WRF_file(SIMULATION,DOMAIN,WHICH_TIME):
    print('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/wrfout*')
    wrf_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/wrfout*'))# CSU machine
    print('        total # files = ',len(wrf_files))
    print('        first file is ',wrf_files[0])
    print('        last file is ',wrf_files[-1])
    if WHICH_TIME=='start':
        selected_fil    = wrf_files[0]
    if WHICH_TIME=='middle':
        selected_fil    = wrf_files[int(len(wrf_files)/2)]
    if WHICH_TIME=='end':
        selected_fil    = wrf_files[-1]
    print('        choosing the middle file: ',selected_fil)

    return selected_fil

def find_RAMS_file(SIMULATION, DOMAIN, WHICH_TIME):
    if DOMAIN=='1' or DOMAIN =='2':
        if SIMULATION=='ARG1.1-R':
            first_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G3_old'+'/out_30s/'
        else:
            first_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G3/out_30s/'
            
        print('searching in ',first_folder)
        rams_files=sorted(glob.glob(first_folder+'a-L-*g1.h5'))
        print('        total # files = ',len(rams_files))
        print('        first file is ',rams_files[0])
        print('        last file is ',rams_files[-1])

        if WHICH_TIME=='start':
            selected_fil    = rams_files[0]
            print('        choosing the starting file: ',selected_fil)
        if WHICH_TIME=='middle':
            selected_fil    = rams_files[int(len(rams_files)/2)]
            print('        choosing the middle file: ',selected_fil)
        if WHICH_TIME=='end':
            selected_fil    = rams_files[-1]
            print('        choosing the end file: ',selected_fil)
            
#         try:
#             first_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G3/out_30s/'
#             print('searching in ',first_folder)
#             rams_files=sorted(glob.glob(first_folder+'a-L-*g'+DOMAIN+'.h5'))
#             print('        total # files = ',len(rams_files))
#             print('        first file is ',rams_files[0])
#             print('        last file is ',rams_files[-1])

#             if WHICH_TIME=='start':
#                 selected_fil    = rams_files[0]
#             if WHICH_TIME=='middle':
#                 selected_fil    = rams_files[int(len(rams_files)/2)]
#             if WHICH_TIME=='end':
#                 selected_fil    = rams_files[-1]
#             print('        choosing the middle file: ',selected_fil)
#         except (IndexError, FileNotFoundError):
#             second_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/out/'
#             print('No files found or folder does not exist. Now searching in '+second_folder)
#             # Change directory to a different folder and try again
#             if os.path.isdir(second_folder):
#                 rams_files=sorted(glob.glob(second_folder+'a-A-*g'+DOMAIN+'.h5'))
#                 print('        total # files = ',len(rams_files))
#                 print('        first file is ',rams_files[0])
#                 print('        last file is ',rams_files[-1])

#                 if WHICH_TIME=='start':
#                     selected_fil    = rams_files[0]
#                 if WHICH_TIME=='middle':
#                     selected_fil    = rams_files[int(len(rams_files)/2)]
#                 if WHICH_TIME=='end':
#                     selected_fil    = rams_files[-1]
#                 print('        choosing the middle file: ',selected_fil)           
#             else:
#                 print("Alternate folder does not exist. Exiting function.")
        
    if DOMAIN=='3':
        try:
            first_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/out_30s/'
            print('searching in ',first_folder)
            rams_files=sorted(glob.glob(first_folder+'a-L-*g3.h5'))
            print('        total # files = ',len(rams_files))
            print('        first file is ',rams_files[0])
            print('        last file is ',rams_files[-1])

            if WHICH_TIME=='start':
                selected_fil    = rams_files[0]
            if WHICH_TIME=='middle':
                selected_fil    = rams_files[int(len(rams_files)/2)]
            if WHICH_TIME=='end':
                selected_fil    = rams_files[-1]
            print('        choosing the middle file: ',selected_fil)
        except (IndexError, FileNotFoundError):
            second_folder = '/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'_old/out_30s/'
            print('No files found or folder does not exist. Now searching in '+second_folder)
            # Change directory to a different folder and try again
            if os.path.isdir(second_folder):
                rams_files=sorted(glob.glob(second_folder+'a-L-*g3.h5'))#
                print('        total # files = ',len(rams_files))
                print('        first file is ',rams_files[0])
                print('        last file is ',rams_files[-1])

                if WHICH_TIME=='start':
                    selected_fil    = rams_files[0]
                if WHICH_TIME=='middle':
                    selected_fil    = rams_files[int(len(rams_files)/2)]
                if WHICH_TIME=='end':
                    selected_fil    = rams_files[-1]
                print('        choosing the middle file: ',selected_fil)
            else:
                print("Alternate folder does not exist. Exiting function.")

    return selected_fil
   
def read_head(headfile,h5file):
        # Function that reads header files from RAMS

        # Inputs:
        #   headfile: header file including full path in str format
        #   h5file: h5 datafile including full path in str format

        # Returns:
        #   zmn: height levels for momentum values (i.e., grid box upper and lower levels)
        #   ztn: height levels for thermodynaic values (i.e., grid box centers)
        #   nx:: the number of x points for the domain associated with the h5file
        #   ny: the number of y points for the domain associated with the h5file
        #   npa: the number of surface patches


        dom_num = h5file[h5file.index('.h5')-1] # Find index of .h5 to determine position showing which nest domain to use

        with open(headfile) as f:
            contents = f.readlines()

        idx_zmn = contents.index('__zmn0'+dom_num+'\n')
        nz_m = int(contents[idx_zmn+1])
        zmn = np.zeros(nz_m)
        for i in np.arange(0,nz_m):
            zmn[i] =  float(contents[idx_zmn+2+i])

        idx_ztn = contents.index('__ztn0'+dom_num+'\n')
        nz_t = int(contents[idx_ztn+1])
        ztn = np.zeros(nz_t)
        for i in np.arange(0,nz_t):
            ztn[i] =  float(contents[idx_ztn+2+i])

        ztop = np.max(ztn) # Model domain top (m)

        # Grad the size of the horizontal grid spacing
        idx_dxy = contents.index('__deltaxn\n')
        dxy = float(contents[idx_dxy+1+int(dom_num)].strip())

        idx_npatch = contents.index('__npatch\n')
        npa = int(contents[idx_npatch+2])

        idx_ny = contents.index('__nnyp\n')
        idx_nx = contents.index('__nnxp\n')
        ny = np.ones(int(contents[idx_ny+1]))
        nx = np.ones(int(contents[idx_ny+1]))
        for i in np.arange(0,len(ny)):
            nx[i] = int(contents[idx_nx+2+i])
            ny[i] = int(contents[idx_ny+2+i])

        ny_out = ny[int(dom_num)-1]
        nx_out = nx[int(dom_num)-1]

        return zmn, ztn, nx_out, ny_out, dxy, npa 
    
def remove_edges(TWOD_FIELD, EDGE_WIDTH_IN_KM, GRID_SPACING_IN_KM):
    print('removing '+str(int(EDGE_WIDTH_IN_KM))+' km along each edge of the domain')
    edge_width_pixels = int(EDGE_WIDTH_IN_KM/GRID_SPACING_IN_KM)
    return TWOD_FIELD[edge_width_pixels:-edge_width_pixels,edge_width_pixels:-edge_width_pixels]
    
def produce_random_coords(X_DIM,Y_DIM,SAMPLE_SIZE,COORDS_RETURN_TYPE='list',SAMPLING_STRATEGY='randomly_distributed'):
    print('getting a random sample of coordinates...')
    print('        shape of the arrays is ',Y_DIM,'x',X_DIM)
    x      = np.arange(0,X_DIM)
    y      = np.arange(0,Y_DIM)
    # # full coordinate arrays
    xx, yy = np.meshgrid(x, y)
    
    if SAMPLE_SIZE>=(X_DIM*Y_DIM):
        print('        sample = or > than the population; choosing all points')
        coords_tuples_2d = np.vstack(([yy.T], [xx.T])).T
        print('        shape of combined coords matrix: ',np.shape(coords_tuples_2d))
        coords_all = coords_tuples_2d.reshape(-1, 2).tolist()
        print('        shape of 1d list of coords: ',np.shape(coords_all))
        
        if COORDS_RETURN_TYPE=='tuple':
            coords_all =  [tuple(sublist) for sublist in coords_all]
            
        coords = coords_all 
        
    else:
        
        if SAMPLING_STRATEGY=='randomly_distributed':
            print('sampling points irregularly/randomly...')
            coords_tuples_2d = np.vstack(([yy.T], [xx.T])).T
            print('        shape of combined coords matrix: ',np.shape(coords_tuples_2d))
            coords_all = coords_tuples_2d.reshape(-1, 2).tolist()
            print('        shape of 1d list of coords: ',np.shape(coords_all))
        
            if COORDS_RETURN_TYPE=='tuple':
                coords_all =  [tuple(sublist) for sublist in coords_all]
            
            coords = random.sample(coords_all,SAMPLE_SIZE)
            
        elif SAMPLING_STRATEGY=='lattice':
            print('sampling points on a lattice...')
            total_points = X_DIM * Y_DIM
            step_size = int(np.sqrt(total_points/SAMPLE_SIZE))

            # Generate the grid indices
            x_indices = np.arange(0, X_DIM, step_size)
            y_indices = np.arange(0, Y_DIM, step_size)

            # Create the grid points using meshgrid and ravel to flatten
            x_grid, y_grid = np.meshgrid(x_indices, y_indices)
            coords_tuples_2d = np.vstack(([y_grid.T], [x_grid.T])).T
            print('        shape of combined coords matrix: ',np.shape(coords_tuples_2d))
            coords = coords_tuples_2d.reshape(-1, 2).tolist()
            print('        shape of 1d list of coords: ',np.shape(coords))
            
            if COORDS_RETURN_TYPE=='tuple':
                coords =  [tuple(sublist) for sublist in coords]
    
        else:
            print('provide an appropriate value for the SAMPLING_STRATEGY argument')
            
        
    return coords

def produce_random_coords_conditional(SAMPLE_SIZE,TWOD_CONDITIONAL_FIELD, CONDITION_STATEMENT=lambda x: x > -500,COORDS_RETURN_TYPE='list'):
    print('getting a random sample of coordinates where ',CONDITION_STATEMENT)
    print('        shape of the 2D condition field is ',np.shape(TWOD_CONDITIONAL_FIELD))
    
    def indices_where_condition_met(array, condition):
        indices = np.where(condition(array))
        return list(zip(indices[0], indices[1]))

    # Get indices where condition is met
    coords_all = indices_where_condition_met(TWOD_CONDITIONAL_FIELD, CONDITION_STATEMENT)
    print('length of all coordinates where condition is met is ',len(coords_all),' about ',int(len(coords_all)*100.0/TWOD_CONDITIONAL_FIELD.size), ' percent of the total grid points')

    if COORDS_RETURN_TYPE=='list':
        coords_all =  [list(sublist) for sublist in coords_all]
    if COORDS_RETURN_TYPE=='tuple':
        pass

    print('        shape of 1d list of coords: ',np.shape(coords_all))
    
    if SAMPLE_SIZE>=(np.shape(TWOD_CONDITIONAL_FIELD)[0]*np.shape(TWOD_CONDITIONAL_FIELD)[1]):
        print('        sample = or > than the population; choosing all points')
        coords = coords_all 
    if SAMPLE_SIZE>len(coords_all):
        coords = coords_all 
    else:
        coords = random.sample(coords_all,SAMPLE_SIZE)
    return coords
    
def get_values_at_random_coords(TWOD_FIELD, COORDS, COORDS_RETURN_TYPE='list'):
    print('getting values at the chosen coordinates...')
    print('        got the data... min = ',np.nanmin(TWOD_FIELD),' max = ',np.nanmax(TWOD_FIELD))
    print('        percentage of nans is ',np.count_nonzero(np.isnan(TWOD_FIELD))/len(TWOD_FIELD.flatten()))
    print('        choosing '+str(len(COORDS))+' random points...')
    print('        get field values from these points...')
    values = np.fromiter((TWOD_FIELD[c[0], c[1]] for c in COORDS), dtype=float)
    # Remove nan values
    print('        Removing nan values and the corresponding coordinates...')
    nan_mask = ~np.isnan(values)
    print('        # non-nan values',np.count_nonzero(nan_mask))
    values   = values[nan_mask]
    sampled_coords_array = np.array(COORDS)
    coords   = sampled_coords_array[nan_mask].tolist()
    
    if COORDS_RETURN_TYPE=='tuple':
        coords =  [tuple(sublist) for sublist in coords]
        
    print('        final shape of coords is ',np.shape(coords))
    print('        final shape of values is ',np.shape(values))
    return coords, values

def make_variogram(COORDS, VALUES, NBINS, MAXLAG, DX=1.0, BIN_FUNCTION='even',ESTIMATOR='matheron'):
    """
    Estimator options:
    1. matheron [Matheron, default]
    2. 1st-order structure function
    3. 2nd-order structure function
    4. 3rd-order structure function
    5. 4th-order structure function
    6. 5th-order structure function
    7. cressie [Cressie-Hawkins]
    8. dowd [Dowd-Estimator]
    9. genton [Genton]
    10. minmax [MinMax Scaler]
    11. entropy [Shannon Entropy]
    """
    print('        creating variogram...')
    print('        MAXLAG= ',MAXLAG,'grid points')
    V        = skg.Variogram(COORDS, VALUES,n_lags=NBINS,maxlag = MAXLAG, bin_func=BIN_FUNCTION,estimator=ESTIMATOR)
    bins     = V.bins*DX # convert from integer coordinates to physical coordinates (km)
    #print('        upper edges of bins: ',bins,'\n')
    bins = np.subtract(bins, np.diff([0] + bins.tolist()) / 2)
    #print('        mid points of bins: ',bins)
    exp_variogram =  V.experimental
    #matrix_for_saving = np.array([bins,exp_variogram]).T
    return V , bins, exp_variogram#, matrix_for_saving
    
def retrieve_histogram(VARIOGRAM,DX=1.0):
    print('        retreiving counts of pairwise obs per lag class ...')
    bins_upper_edges = VARIOGRAM.bins*DX
    counts = np.fromiter((g.size for g in VARIOGRAM.lag_classes()), dtype=int)
    widths = np.diff([0] + bins_upper_edges.tolist())
    bins_middle_points   = np.subtract(bins_upper_edges, np.diff([0] + bins_upper_edges.tolist()) / 2)
    #print('        widths of lag classes are: ',widths)
    #print('length of bins_middle_points:',len(bins_middle_points))
    #print('length of width:',len(widths))
    return bins_middle_points, counts, widths

def parse_variogram_filename(filename):
    # Extracting fields using regex
    pattern = (r"ensemble_(\d+)_"
               r"([a-zA-Z_]+)_sampling_(\d+)_samples_"
               r"binwidth_(\d+km)_estimator_([a-zA-Z_]+)_"
               r"([a-zA-Z0-9\.\-]+)_G(\d+)_"
               r"([a-zA-Z]+)_points_threshold_([0-9e\.\-]+)_"
               r"([a-zA-Z]+)_levtype_([a-zA-Z]+)_lev_(-?\d+)_"
               r"(\d+)\.csv")
    
    match = re.search(pattern, filename)
    
    if not match:
        return "Filename pattern does not match"
    
    ensemble_number = int(match.group(1))
    sampling_type = match.group(2)
    num_samples = int(match.group(3))
    binwidth = match.group(4)
    estimator = match.group(5)
    simulation_name = match.group(6)
    grid = f"G{match.group(7)}"
    variogram_region = match.group(8)
    segmentation_threshold = float(match.group(9))
    variable_name = match.group(10)
    level_type = match.group(11)
    level = int(match.group(12))
    time = pd.to_datetime(match.group(13), format='%Y%m%d%H%M%S')
    
    return {
        'ensemble_number': ensemble_number,
        'sampling_type': sampling_type,
        'num_samples': num_samples,
        'binwidth': binwidth,
        'estimator': estimator,
        'simulation_name': simulation_name,
        'grid': grid,
        'variogram_region': variogram_region,
        'segmentation_threshold': segmentation_threshold,
        'variable_name': variable_name,
        'level_type': level_type,
        'level': level,
        'time': time
    }

def average_variograms(file_list):
    # Initialize an empty DataFrame to hold the cumulative sum of variograms
    cumulative_variograms = None
    count_files = 0

    for file in file_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        #print(df)

        # Ensure the file has the correct columns
        if 'bins' in df.columns and 'exp_variogram' in df.columns:
            if cumulative_variograms is None:
                # Initialize cumulative_variograms with the first file
                cumulative_variograms = df.copy()
            else:
                # Add the variograms from this file to the cumulative sum
                cumulative_variograms['exp_variogram'] += df['exp_variogram']

            count_files += 1
        else:
            print(f"File {file} does not have the required columns.")

    if count_files == 0:
        raise ValueError("No valid files were provided.")

    # Calculate the average by dividing by the number of files
    cumulative_variograms['exp_variogram'] /= count_files

    return cumulative_variograms

def grab_intersection_gbig_gsmall_RAMS(VARIABLE,RAMS_G1_or_G2_FILE,RAMS_G3_FILE):
    z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(RAMS_G1_or_G2_FILE,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    #print(np.min(z))
    #print(np.max(z))
    #z2, z_name2, z_units2, z_time2 = read_vars_WRF_RAMS.read_variable(RAMS_G3_FILE,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    print('        done getting the variable ',VARIABLE[0],' with shape: ',np.shape(z),'\n')
    print('        subsetting the larger domain...\n')
    # read the variables for which you want the variogram
    ds_big   = xr.open_dataset(RAMS_G1_or_G2_FILE,engine='h5netcdf',phony_dims='sort')[['GLAT','GLON']]
    ds_small = xr.open_dataset(RAMS_G3_FILE,engine='h5netcdf',phony_dims='sort')[['GLAT','GLON']]
    dim1, dim2 = ds_big.GLAT.dims
    #print(ds_big)
    #print(ds_small)
    #ds_big = ds_big.rename_dims({'phony_dim_0': 'y','phony_dim_1': 'x'})
    #ds_small = ds_small.rename_dims({'phony_dim_0': 'y','phony_dim_1': 'x'})
    min_lat_big = ds_big.GLAT.min().values
    max_lat_big = ds_big.GLAT.max().values
    min_lon_big = ds_big.GLON.min().values
    max_lon_big = ds_big.GLON.max().values
    print('        min and max lat for big domain = ',min_lat_big,' ',max_lat_big)
    print('        min and max lon for big domain = ',min_lon_big,' ',max_lon_big)
    print('        ----')
    min_lat_small = ds_small.GLAT.min().values
    max_lat_small = ds_small.GLAT.max().values
    min_lon_small = ds_small.GLON.min().values
    max_lon_small = ds_small.GLON.max().values
    print('        min and max lat for small domain = ',min_lat_small,' ',max_lat_small)
    print('        min and max lon for small domain = ',min_lon_small,' ',max_lon_small)
    print('        ----')
    #subset by lat/lon - used so only region covered by inner grid is compared
    ds = xr.Dataset({VARIABLE[0]: xr.DataArray(data   = z,  dims   = [dim1,dim2])})
    ds = ds.assign(GLAT=ds_big.GLAT)
    ds = ds.assign(GLON=ds_big.GLON)
    #print(ds)
    ds = ds.where((ds.GLAT>=min_lat_small) & (ds.GLAT<=max_lat_small) & (ds.GLON>=min_lon_small) & (ds.GLON<=max_lon_small), drop=True)
    #print(ds)
    min_lat = ds.GLAT.min().values
    max_lat = ds.GLAT.max().values
    min_lon = ds.GLON.min().values
    max_lon = ds.GLON.max().values
    
    print('        min and max lat for modified domain = ',min_lat,' ',max_lat)
    print('        min and max lon for modified domain = ',min_lon,' ',max_lon)
    print('        ----')
    
    #print(ds)
    print('        shape of small domain: ',np.shape(ds_small.GLAT))
    print('        shape of big domain: ',np.shape(ds_big.GLAT))
    print('        shape of modified domain: ',np.shape(ds.GLAT))
    #return z, z_name, z_units, z_time
    return ds.variables[VARIABLE[0]].values, z_name, z_units, z_time

def grab_intersection_gbig_gsmall_WRF(VARIABLE,WRF_G1_or_G2_FILE,WRF_G3_FILE):
    z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(WRF_G1_or_G2_FILE,VARIABLE[0],'WRF',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    print('        done getting the variable ',VARIABLE[0],' with shape: ',np.shape(z),'\n')
    print('        subsetting the larger domain...\n')
    # read the variables for which you want the variogram
    ds_big   = xr.open_dataset(WRF_G1_or_G2_FILE)[['XLAT','XLONG']].squeeze()
    ds_small = xr.open_dataset(WRF_G3_FILE)[['XLAT','XLONG']].squeeze()
    #print(ds_big.XLAT)
    dim1, dim2 = ds_big.XLAT.dims
    min_lat_big = ds_big.XLAT.min().values
    max_lat_big = ds_big.XLAT.max().values
    min_lon_big = ds_big.XLONG.min().values
    max_lon_big = ds_big.XLONG.max().values
    print('        min and max lat for big domain = ',min_lat_big,' ',max_lat_big)
    print('        min and max lon for big domain = ',min_lon_big,' ',max_lon_big)
    print('        ----')
    min_lat_small = ds_small.XLAT.min().values
    max_lat_small = ds_small.XLAT.max().values
    min_lon_small = ds_small.XLONG.min().values
    max_lon_small = ds_small.XLONG.max().values
    print('        min and max lat for small domain = ',min_lat_small,' ',max_lat_small)
    print('        min and max lon for small domain = ',min_lon_small,' ',max_lon_small)
    print('        ----')
    #subset by lat/lon - used so only region covered by inner grid is compared
    ds = xr.Dataset({VARIABLE[0]: xr.DataArray(data   = z,  dims   = [dim1,dim2])})
    #print(ds)
    print('-----')
    ds = ds.assign_coords(XLAT=ds_big.XLAT)
    #print(ds)
    ds = ds.assign_coords(XLONG=ds_big.XLONG)
    #print(ds)
    ds = ds.where((ds.XLAT>=min_lat_small) & (ds.XLAT<=max_lat_small) & (ds.XLONG>=min_lon_small) & (ds.XLONG<=max_lon_small), drop=True)
    #print(ds)
    min_lat = ds.XLAT.min().values
    max_lat = ds.XLAT.max().values
    min_lon = ds.XLONG.min().values
    max_lon = ds.XLONG.max().values
    
    print('        min and max lat for modified domain = ',min_lat,' ',max_lat)
    print('        min and max lon for modified domain = ',min_lon,' ',max_lon)
    print('        ----')
    
    #print(ds)
    print('        shape of small domain: ',np.shape(ds_small.XLAT))
    print('        shape of big domain: ',np.shape(ds_big.XLAT))
    print('        shape of modified domain: ',np.shape(ds.XLAT))
    #return z, z_name, z_units, z_time
    return ds.variables[VARIABLE[0]].values, z_name, z_units, z_time

def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S'), pd_time

def get_time_from_WRF_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][11:30] # Grab time string from WRF file
    pd_time = pd.to_datetime(cur_time[0:9]+' '+cur_time[11:18].replace('_', ':'))
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')

def get_time_from_a_file(INPUT_FILE,MODEL_NAME):
    if MODEL_NAME=='RAMS':
        return get_time_from_RAMS_file(INPUT_FILE)
    if MODEL_NAME=='WRF':
        return get_time_from_WRF_file(INPUT_FILE)

def find_closest_datetime_index(datetime_list, target_datetime):
    """
    Find the index of the closest datetime in the datetime_list to the target_datetime.
    """
    closest_datetime = min(datetime_list, key=lambda x: abs(x - target_datetime))
    closest_index = datetime_list.index(closest_datetime)
    return closest_index

def compute_moran(DISTANCE_INTERVAL, COORDS, VALUES):
    # Create binary spatial weights matrix based on distance interval
    w = libpysal.weights.DistanceBand(COORDS, threshold=DISTANCE_INTERVAL, binary=True, silence_warnings=True)
    # Compute Moran's I
    moran = Moran(VALUES, w)
    return DISTANCE_INTERVAL, moran.I, moran.EI, moran.VI_norm, moran.p_norm, moran.z_norm


def arrange_images_with_wildcard(input_folder, output_file, wildcard_pattern, non_target_string):
    # Get a list of PNG images in the input folder matching the wildcard pattern
    if non_target_string:
        image_files = sorted([f for f in glob.glob(os.path.join(input_folder, wildcard_pattern)) if f.lower().endswith('.png') and non_target_string not in f])[1::2]
    else:
        image_files = sorted([f for f in glob.glob(os.path.join(input_folder, wildcard_pattern)) if f.lower().endswith('.png')])[1::2]

    print('found ',len(image_files),' images')
    for fil in image_files:
        print(fil)
    # Check if there are any matching images
    if not image_files:
        print(f"Error: No PNG images matching the wildcard pattern '{wildcard_pattern}' found in the folder.")
        return

    # Calculate the number of rows and columns for the matrix
    num_images = len(image_files)
    num_cols = int(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Create a new image with dimensions for the matrix and reduced white space
    img_width, img_height = Image.open(image_files[0]).size
    margin = 60  # Adjust this value to control the margin
    result_image = Image.new('RGB', (num_cols * (img_width - margin), num_rows * (img_height - margin)))

    # Loop through the matching images and paste them onto the result image with reduced white space
    for i in range(num_images):
        img = Image.open(image_files[i])

        # Calculate the position with margin to paste the image
        col = i % num_cols
        row = i // num_cols
        position = (col * (img_width - margin), row * (img_height - margin))

        # Paste the image onto the result image
        result_image.paste(img, position)

    # Save the result image
    result_image.save(output_file)
    
    
def make_plan_view_RAMS_WRF(WHICH_TIME, VARIABLE, SIMULATION, DOMAIN, CMAP, SAMPLE_SIZE, SAVEFILE, CONDITION_MASK=None, COORDS = None, CONDITION_INFO=None, SCATTER_COLOR='k', MASKED_PLOT=False, SHOW_INNER_DOMAINS=False, CBAR=True):

    units_dict = {'Tk':'$K$','QV':'$kg kg^{-1}$','RH':'percent','WSPD':'$m s^{-1}$','U':'$m s^{-1}$',\
              'V':'$m s^{-1}$','W':'$m s^{-1}$','MCAPE':'$J kg^{-1}$','MCIN':'$J kg^{-1}$','THETA':'$K$','QTC':'$kg kg^{-1}$',\
                  'SHF':'$W m^{-2}$', 'LHF':'$W m^{-2}$','MAXCOL_W':'$m s^{-1}$','THETAV':'$K$','ITC':'$mm$'}
    
    vmin_vmax_dict = {'Tk':[290,331,1],'QV':[0.006,0.024,0.001],'RH':[70,101,1],'WSPD':[1,20,1],'U':[1,20,1],\
              'V':[1,20,1],'W':[-5,21,1],'MCAPE':[100,3100,100],'MCIN':[0,310,10],'THETA':[290,331,1],
                     'SHF':[-100,310,10],'LHF':[-100,310,10]}
 
    contour_min=vmin_vmax_dict[VARIABLE[0]][0]
    print('contour_min: ',contour_min)
    contour_max=vmin_vmax_dict[VARIABLE[0]][1]
    print('contour_max: ',contour_max)
    contour_spacing=vmin_vmax_dict[VARIABLE[0]][2]
    print('contour_spacing: ',contour_spacing)
    
    
    if SIMULATION=='AUS1.1-R':
        simulation_full_name = 'Australia'
    if SIMULATION=='DRC1.1-R':
        simulation_full_name = 'Congo'
    if SIMULATION=='PHI1.1-R':
        simulation_full_name = 'Philippines_West'
    if SIMULATION=='PHI2.1-R':
        simulation_full_name = 'Philippines_East'
    if SIMULATION=='WPO1.1-R':
        simulation_full_name = 'Western_Pacific'
    if SIMULATION=='BRA1.1-R':
        simulation_full_name = 'Brazil1'
    if SIMULATION=='BRA1.2-R':
        simulation_full_name = 'Brazil2'
    if SIMULATION=='USA1.1-R':
        simulation_full_name = 'Gulf_Coast'
    if SIMULATION=='RSA1.1-R':
        simulation_full_name = 'South_Africa'
    if SIMULATION=='ARG1.1-R':
        simulation_full_name = 'Argentina1'
    if SIMULATION=='ARG1.2-R':
        simulation_full_name = 'Argentina2'

    if SIMULATION=='AUS1.1-R':
        grid_boxes_rams = {'G2': [(128.43218994140625, -14.15191650390625), (133.06552124023438, -14.142359733581543), (133.02102661132812, -9.837047576904297), (128.45965576171875, -9.84489917755127)], 'G3': [(129.5452880859375, -13.191335678100586), (131.8562774658203, -13.187128067016602), (131.84426879882812, -11.20703125), (129.55003356933594, -11.210875511169434)]}
    if SIMULATION=='DRC1.1-R':
        grid_boxes_rams = {'G2': [(21.134906768798828, -6.180349349975586), (26.289892196655273, -6.163394451141357), (26.264724731445312, -1.8193483352661133), (21.137409210205078, -1.8290973901748657)], 'G3': [(24.061450958251953, -5.501791477203369), (25.749422073364258, -5.493662357330322), (25.736658096313477, -3.095113515853882), (24.053871154785156, -3.101199150085449)]}
    if SIMULATION=='PHI1.1-R':
        grid_boxes_rams = {'G2': [(117.5504379272461, 15.917317390441895), (120.23397827148438, 15.915518760681152), (120.255126953125, 18.567716598510742), (117.53333282470703, 18.569684982299805)], 'G3': [(118.46419525146484, 16.512176513671875), (120.03064727783203, 16.508451461791992), (120.04146575927734, 18.08207130432129), (118.46177673339844, 18.086000442504883)]}
    if SIMULATION=='PHI2.1-R':
        grid_boxes_rams = {'G2': [(126.4293441772461, 14.675756454467773), (131.07505798339844, 14.69363784790039), (131.1012420654297, 18.108726501464844), (126.37310028076172, 18.088594436645508)], 'G3': [(127.29903411865234, 15.166404724121094), (129.81304931640625, 15.178471565246582), (129.81573486328125, 17.62335777282715), (127.26988220214844, 17.6102237701416)]}
    if SIMULATION=='WPO1.1-R':
        grid_boxes_rams = {'G2': [(135.5248260498047, 12.084915161132812), (139.48826599121094, 12.099800109863281), (139.501708984375, 15.29932689666748), (135.484375, 15.282389640808105)], 'G3': [(135.78146362304688, 12.294074058532715), (137.4358673095703, 12.305987358093262), (137.427001953125, 14.300797462463379), (135.7588348388672, 14.287872314453125)]}
    if SIMULATION=='BRA1.1-R':
        grid_boxes_rams = {'G2': [(-61.3571891784668, -4.338754653930664), (-58.65502166748047, -4.344385147094727), (-58.65354537963867, -1.6485878229141235), (-61.3496208190918, -1.6451811790466309)], 'G3': [(-60.97938919067383, -3.8747100830078125), (-59.22237014770508, -3.8782856464385986), (-59.22057342529297, -2.1246743202209473), (-60.97501754760742, -2.122084617614746)]}
    if SIMULATION=='BRA1.2-R':
        grid_boxes_rams = {'G2': [(-61.99944305419922,-5.999470233917236), (-56.99296951293945,-5.999470233917236), (-56.99296951293945,-1.801897644996643), (-61.99944305419922,-1.801897644996643)],'G3':[(-60.804481506347656,-5.602199077606201), (-58.69898223876953,-5.602199077606201), (-58.69898223876953,-3.2987847328186035), (-60.804481506347656,-3.2987847328186035)]}
    if SIMULATION=='USA1.1-R':
        grid_boxes_rams = {'G2': [(-96.52090454101562, 27.413938522338867), (-92.47908782958984, 27.413936614990234), (-92.41642761230469, 30.569290161132812), (-96.58356475830078, 30.569290161132812)], 'G3': [(-95.95916748046875, 28.246612548828125), (-93.4371109008789, 28.25029182434082), (-93.4184799194336, 30.039934158325195), (-95.9847412109375, 30.036081314086914)]}
    if SIMULATION=='RSA1.1-R':
        grid_boxes_rams = {'G2': [(27.49627685546875, -27.29660987854004), (32.55283737182617, -27.24605941772461), (32.42838668823242, -23.461658477783203), (27.53109359741211, -23.507389068603516)], 'G3': [(28.736751556396484, -26.92152976989746), (31.79244041442871, -26.885950088500977), (31.732847213745117, -24.680686950683594), (28.73419189453125, -24.714256286621094)]}
    if SIMULATION=='ARG1.1-R':
        grid_boxes_rams = {'G2': [(-64.0031509399414,-34.36425018310547), (-60.94878387451172,-34.36425018310547), (-60.94878387451172,-31.815420150756836), (-64.0031509399414,-31.815420150756836)],'G3':[(-63.49833679199219,-34.02104949951172), (-61.38158416748047,-34.02104949951172), (-61.38158416748047,-32.274147033691406), (-63.49833679199219,-32.274147033691406)]}
    if SIMULATION=='ARG1.2-R':
        grid_boxes_rams = {'G2': [(-67.0828476, -34.26351166), (-60.11714554, -34.26351166), (-60.11714554, -29.89814568), (-67.0828476, -29.89814568)], 'G3': [(-65.86869049, -33.27977753), (-62.64379883, -33.27977753), (-62.64379883, -31.10369492), (-65.86869049, -31.10369492)]}

    if SIMULATION=='AUS1.1-W':
        grid_boxes_rams = {'G2': [(128.43218994140625, -14.15191650390625), (133.06552124023438, -14.142359733581543), (133.02102661132812, -9.837047576904297), (128.45965576171875, -9.84489917755127)], 'G3': [(129.5452880859375, -13.191335678100586), (131.8562774658203, -13.187128067016602), (131.84426879882812, -11.20703125), (129.55003356933594, -11.210875511169434)]}
    if SIMULATION=='DRC1.1-W':
        grid_boxes_rams = {'G2': [(21.134906768798828, -6.180349349975586), (26.289892196655273, -6.163394451141357), (26.264724731445312, -1.8193483352661133), (21.137409210205078, -1.8290973901748657)], 'G3': [(24.061450958251953, -5.501791477203369), (25.749422073364258, -5.493662357330322), (25.736658096313477, -3.095113515853882), (24.053871154785156, -3.101199150085449)]}
    if SIMULATION=='PHI1.1-W':
        grid_boxes_rams = {'G2': [(117.5504379272461, 15.917317390441895), (120.23397827148438, 15.915518760681152), (120.255126953125, 18.567716598510742), (117.53333282470703, 18.569684982299805)], 'G3': [(118.46419525146484, 16.512176513671875), (120.03064727783203, 16.508451461791992), (120.04146575927734, 18.08207130432129), (118.46177673339844, 18.086000442504883)]}
    if SIMULATION=='PHI2.1-W':
        grid_boxes_rams = {'G2': [(126.4293441772461, 14.675756454467773), (131.07505798339844, 14.69363784790039), (131.1012420654297, 18.108726501464844), (126.37310028076172, 18.088594436645508)], 'G3': [(127.29903411865234, 15.166404724121094), (129.81304931640625, 15.178471565246582), (129.81573486328125, 17.62335777282715), (127.26988220214844, 17.6102237701416)]}
    if SIMULATION=='WPO1.1-W':
        grid_boxes_rams = {'G2': [(135.5248260498047, 12.084915161132812), (139.48826599121094, 12.099800109863281), (139.501708984375, 15.29932689666748), (135.484375, 15.282389640808105)], 'G3': [(135.78146362304688, 12.294074058532715), (137.4358673095703, 12.305987358093262), (137.427001953125, 14.300797462463379), (135.7588348388672, 14.287872314453125)]}
    if SIMULATION=='BRA1.1-W':
        grid_boxes_rams = {'G2': [(-61.3571891784668, -4.338754653930664), (-58.65502166748047, -4.344385147094727), (-58.65354537963867, -1.6485878229141235), (-61.3496208190918, -1.6451811790466309)], 'G3': [(-60.97938919067383, -3.8747100830078125), (-59.22237014770508, -3.8782856464385986), (-59.22057342529297, -2.1246743202209473), (-60.97501754760742, -2.122084617614746)]}
    if SIMULATION=='USA1.1-W':
        grid_boxes_rams = {'G2': [(-96.52090454101562, 27.413938522338867), (-92.47908782958984, 27.413936614990234), (-92.41642761230469, 30.569290161132812), (-96.58356475830078, 30.569290161132812)], 'G3': [(-95.95916748046875, 28.246612548828125), (-93.4371109008789, 28.25029182434082), (-93.4184799194336, 30.039934158325195), (-95.9847412109375, 30.036081314086914)]}
    if SIMULATION=='RSA1.1-W':
        grid_boxes_rams = {'G2': [(27.49627685546875, -27.29660987854004), (32.55283737182617, -27.24605941772461), (32.42838668823242, -23.461658477783203), (27.53109359741211, -23.507389068603516)], 'G3': [(28.736751556396484, -26.92152976989746), (31.79244041442871, -26.885950088500977), (31.732847213745117, -24.680686950683594), (28.73419189453125, -24.714256286621094)]}

    print('Contour plotting ',VARIABLE,'\n')
    
    #fig    = plt.figure(figsize=(8,8))
    #ax = plt.gca()
    fig, ax = plt.subplots(nrows=1,ncols=1,subplot_kw={'projection': crs.PlateCarree()},figsize=(8,8))
    print('    working on simulation: ',SIMULATION)
    #if model_name=='RAMS':
    if isinstance(WHICH_TIME, dict):
        print('dictionary passed as argument: getting filenames from the dictionary')
        selected_fil = WHICH_TIME[SIMULATION]
    else:
        selected_fil = find_RAMS_file(SIMULATION=SIMULATION,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)
    #if model_name=='WRF':
    #        selected_fil =  variogram_helper_functions.find_WRF_file(SIMULATION=simulation,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)
    
    # get the required variable
    z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(selected_fil,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    
    # grab lat-lon for plotting
    rams_lons, _, _, _ =  read_vars_WRF_RAMS.read_variable(selected_fil,'LON','RAMS',output_height=False,interpolate=False)
    rams_lats, _, _, _ =  read_vars_WRF_RAMS.read_variable(selected_fil,'LAT','RAMS',output_height=False,interpolate=False)
    
    # grab terrain for plotting
    rams_terr, _, _, _ =  read_vars_WRF_RAMS.read_variable(selected_fil,'TERR_HGT','RAMS',output_height=False,interpolate=False)
    
    #y_dim,x_dim = np.shape(z)

    if DOMAIN=='1':
        dx=1.6
    if DOMAIN=='2':
        dx=0.4
    if DOMAIN=='3':
        dx=0.1
    
    #xx = np.arange(0,dx*x_dim,dx)
    #yy = np.arange(0,dx*y_dim,dx)
    timestep_pd     = pd.to_datetime(z_time,format='%Y%m%d%H%M%S')
    
    # if coordinates are given, make a scatter plot 
    if COORDS:
        print('will make scatter plot of given coordinates...')
        main_cont =plt.contourf(xx,yy,z,levels=30,cmap=CMAP,extend='both')
        y_coords, x_coords = zip(*COORDS)
        plt.scatter(np.array(x_coords)*dx, np.array(y_coords)*dx, color=SCATTER_COLOR, marker='o',s=.07) 
        CONDITION_INFO = None

    # make a scatter plot of random points (to be used for the variogram estimation) based on the conditions given, 
    # such as total integrated condensate exceeding some value
    if CONDITION_INFO:
        print('conditional information given; will generate random points for scatter plot...')
        if CONDITION_INFO[0]=='environment':
            # will fetch points from the 'environment' (QTC)
            print('        getting random coordinates over ',CONDITION_INFO[0],' points')
            print('        conditioned on total condensate')
            if VARIABLE[1]<0:
                conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC','RAMS',output_height=False,interpolate=True,level=0,interptype='model')
            else:
                conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC','RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
            if CONDITION_INFO[2]:
                print('        smoothing the condition field')
                conditional_field = smooth2d(conditional_field, passes=CONDITION_INFO[3], meta=False)
                #conditional_field = gaussian_filter(conditional_field, sigma=1) 
            if MASKED_PLOT:
                masked_z = np.ma.masked_where(conditional_field > CONDITION_INFO[1], z)
                main_cont =plt.contourf(xx,yy,masked_z,levels=30,cmap=CMAP,extend='both')
            else:
                main_cont =plt.contourf(xx,yy,z,levels=30,cmap=CMAP,extend='both')
            print('        min, max for the condensate field is ',np.min(conditional_field),' ',np.max(conditional_field))
            coords = produce_random_coords_conditional(SAMPLE_SIZE, conditional_field, CONDITION_STATEMENT=lambda x: x < CONDITION_INFO[1])
        if CONDITION_INFO[0]=='storm': 
            print('        getting random coordinates over ',CONDITION_INFO[0],' points')
            print('        conditioned on total condensate')
            if VARIABLE[1]<0:
                conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC','RAMS',output_height=False,interpolate=True,level=0,interptype='model')
            else:
                conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC','RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
            if CONDITION_INFO[2]:
                print('        smoothing the condition field')
                conditional_field = smooth2d(conditional_field, passes=CONDITION_INFO[3], meta=False)
                #conditional_field = gaussian_filter(conditional_field, sigma=1) 
            if MASKED_PLOT:
                masked_z = np.ma.masked_where(conditional_field <=CONDITION_INFO[1], z)
                main_cont =plt.contourf(xx,yy,masked_z,levels=30,cmap=CMAP,extend='both')
            else:
                main_cont =plt.contourf(xx,yy,z,levels=30,cmap=CMAP,extend='both')
            print('        min, max for the condensate field is ',np.min(conditional_field),' ',np.max(conditional_field))
            coords = produce_random_coords_conditional(SAMPLE_SIZE, conditional_field, CONDITION_STATEMENT=lambda x: x >= CONDITION_INFO[1])
        if CONDITION_INFO[0]=='all':
            print('getting random coordinates over ',CONDITION_INFO[0],' points')
            coords = produce_random_coords(x_dim,y_dim,SAMPLE_SIZE)   
            main_cont =plt.contourf(xx,yy,z,levels=30,cmap=CMAP,extend='both')
        
        # Create scatter plot
        y_coords, x_coords = zip(*coords)
        plt.scatter(np.array(x_coords)*dx, np.array(y_coords)*dx, color=SCATTER_COLOR, marker='o',s=.07)
        
    else:
        if CONDITION_MASK is not None:
            condition_mask = np.where(CONDITION_MASK<-500,np.nan,1.0)
            main_cont     = ax.contourf(rams_lons,rams_lats,z*condition_mask,levels=np.arange(contour_min,contour_max,contour_spacing),transform=crs.PlateCarree(),cmap=CMAP,extend='both')
        else:
        #main_cont =plt.contourf(xx,yy,z,levels=30,cmap=CMAP,extend='both')
            #main_cont     = ax.contourf(rams_lons,rams_lats,z,levels=np.arange(contour_min,contour_max,contour_spacing),transform=crs.PlateCarree(),cmap=CMAP,extend='both')
            main_cont     = ax.contourf(rams_lons,rams_lats,z,levels=np.arange(contour_min,contour_max,contour_spacing),transform=crs.PlateCarree(),cmap=CMAP,extend='both')
        rams_terr_cont= ax.contour (rams_lons,rams_lats,rams_terr,levels=[500.],transform=crs.PlateCarree(),linewidths=1.0,colors="saddlebrown")
            
    if VARIABLE[2]:
        if  VARIABLE[2]=='pressure':
            level_units = ' mb'
            lev = int(VARIABLE[1])
        if  VARIABLE[2]=='model':
            level_units = ''
            lev = int(VARIABLE[1]+1)
        title_string = simulation_full_name+'\n'+timestep_pd.strftime('%Y-%m-%d %H:%M:%S')
        #title_string = simulation_full_name+' '+VARIABLE[3]+' ('+units_dict[VARIABLE[0]]+')'+' at '+VARIABLE[2]+' level '+str(lev)+level_units+' for G'+DOMAIN+'\n'+timestep_pd.strftime('%Y-%m-%d %H:%M:%S')
    else:
        title_string = simulation_full_name+'\n'+timestep_pd.strftime('%Y-%m-%d %H:%M:%S')
        #title_string = simulation_full_name+' '+VARIABLE[3]+' ('+units_dict[VARIABLE[0]]+')'+' for G'+DOMAIN+'\n'+timestep_pd.strftime('%Y-%m-%d %H:%M:%S')
    
    plt.title(title_string,fontsize=30)
    #plt.xlabel('x (km)',fontsize=16)
    #plt.ylabel('y (km)',fontsize=16)

    if CBAR:
        import matplotlib.ticker as ticker

        cbar = plt.colorbar(main_cont, orientation='horizontal')
        cbar.set_label(label=units_dict[VARIABLE[0]], size=25)
        cbar.ax.tick_params(labelsize=25)

        # Format tick labels as regular numbers
        cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{x * 1e3:.0f}")
        cbar.update_ticks()

        # Add single ×10⁻³ label on the side
        cbar.ax.text(1.02, 1.2, r'$\times 10^{-3}$',
                    transform=cbar.ax.transAxes,
                    fontsize=25, ha='left', va='bottom')
    
    gl = ax.gridlines()#color="gray",alpha=0.5, linestyle='--',draw_labels=True,linewidth=2)
    ax.coastlines(resolution='10m',linewidth=1)
    gl.xlines = False
    gl.ylines = False
    LATLON_LABELS=True
    print('LATLON labels are on')
    gl.top_labels = True 
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xlabel_style = {'size': 18, 'color': 'gray'}#, 'weight': 'bold'}
    gl.ylabel_style = {'size': 18, 'color': 'gray'}#, 'weight': 'bold'}
    
    
    if SHOW_INNER_DOMAINS:
        for loc, poly in grid_boxes_rams.items():
            pts = []
            lons, lats = [], []
            for lon, lat in poly:
                pt  = Point(lon, lat)
                pts.append( pt )
                lons.append(pt.x)
                lats.append(pt.y)

            shp = Polygon(pts)
            #axs[2].scatter(lons, lats,transform=crs.PlateCarree())
            ax.add_geometries([shp], crs=crs.PlateCarree(),fc='none', ec="k")#,fc=None, alpha =0.1)

    plt.tight_layout()
    if SAVEFILE:
        savepng = '/home/isingh/code/variogram_data/PNGs'
        if not os.path.exists(savepng):
            os.makedirs(savepng)
        if VARIABLE[2]:
            if CBAR:
                filename = 'plan_view_publication_RAMS_colorbar_'+SIMULATION+'_G'+DOMAIN+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'.png'
            else:
                filename = 'plan_view_publication_RAMS_'+SIMULATION+'_G'+DOMAIN+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'.png'
        else:
            if CBAR:
                filename = 'plan_view_publication_RAMS_colorbar_'+SIMULATION+'_G'+DOMAIN+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'.png'
            else:
                filename = 'plan_view_publication_RAMS_'+SIMULATION+'_G'+DOMAIN+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'.png'

        print('saving to png file: ',filename)
        plt.savefig(savepng+'/'+filename,dpi=200)
    else:
        print('will not save the png file')
    #plt.close()
    print('\n\n')
