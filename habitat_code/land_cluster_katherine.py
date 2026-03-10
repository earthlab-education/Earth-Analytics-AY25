# %% [markdown]
# # Land cover classification at the Missisippi Delta
# 
# In this notebook, you will use a k-means **unsupervised** clustering
# algorithm to group pixels by similar spectral signatures. **k-means** is
# an **exploratory** method for finding patterns in data. Because it is
# unsupervised, you don’t need any training data for the model. You also
# can’t measure how well it “performs” because the clusters will not
# correspond to any particular land cover class. However, we expect at
# least some of the clusters to be identifiable as different types of land
# cover.
# 
# You will use the [harmonized Sentinel/Landsat multispectral
# dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).
# You can access the data with an [Earthdata
# account](https://www.earthdata.nasa.gov/learn/get-started) and the
# [`earthaccess` library from
# NSIDC](https://github.com/nsidc/earthaccess):

# %% [markdown]
# ## STEP 1: Set up
# 
# ### Step 1a: Load libraries and set GDAL parameters
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Import all libraries you will need for this analysis</li>
# <li>Configure GDAL parameters to help avoid connection errors:
# <code>python      os.environ["GDAL_HTTP_MAX_RETRY"] = "5"      os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"</code></li>
# </ol></div></div>

# %%
### reproducible file paths
import os
import pathlib

### serializing and unserializing objects (save objects to disk and load later)
import pickle

### regular expressions
import re

### throw up warnings
import warnings

### projecting coordinate systems for spatial data and mapping
import cartopy.crs as ccrs

### access satellite imagery through the NASA API
import earthaccess

### spatial data analysis
import earthpy as et

### vector/shapefiles
import geopandas as gpd

### visualizations
import geoviews as gv

### visualization
import hvplot.pandas
import hvplot.xarray

### arrays 
import numpy as np

### tables
import pandas as pd

### rasters
import rioxarray as rxr
import rioxarray.merge as rxrmerge
import xarray as xr

### progress bar
from tqdm.notebook import tqdm
from ipywidgets import IntProgress
from IPython.display import display

### polygons
from shapely.geometry import Polygon

### kmeans clustering
from sklearn.cluster import KMeans

### set GDAL parameters
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

### don't show non-critical warnings
warnings.simplefilter('ignore')

# %% [markdown]
# ### Step 1b: Run the caching decorator
# 
# Below you can find code for a caching **decorator** which you can use in
# your code. To use the decorator:
# 
# ``` python
# @cached(key, override)
# def do_something(*args, **kwargs):
#     ...
#     return item_to_cache
# ```
# 
# This decorator will **pickle** the results of running the
# `do_something()` function, and only run the code if the results don’t
# already exist. To override the caching, for example temporarily after
# making changes to your code, set `override=True`. Note that to use the
# caching decorator, you must write your own function to perform each
# task!
# 
# You might notice that typically in these assignments, we start by creating a data_dir to store our data files. Here, our caching decorator is making the data directory for us.

# %%
### make the caching decorator
def cached(func_key, override=False):
    """
    A decorator to cache function results
    
    Parameters
    ==========
    key: str
      File basename used to save pickled results
    override: bool
      When True, re-compute even if the results are already stored
    """
    def compute_and_cache_decorator(compute_function):
        """
        Wrap the caching function
        
        Parameters
        ==========
        compute_function: function
          The function to run and cache results
        """
        def compute_and_cache(*args, **kwargs):
            """
            Perform a computation and cache, or load cached result.
            
            Parameters
            ==========
            args
              Positional arguments for the compute function
            kwargs
              Keyword arguments for the compute function
            """
            ### Add an identifier from the particular function call
            if 'cache_key' in kwargs:
                key = '_'.join((func_key, kwargs['cache_key']))
            else:
                key = func_key

            ### define a file path based on the directory structure in earthpy
            path = os.path.join(
                
                ### earthpy directory
                et.io.HOME, 
                
                ### earthpy dataset
                et.io.DATA_NAME, 
                
                ### make a subdirectory called "jars"
                'jars', 
                
                ### use f-string (formatted string) to create a string by embedding the value
                ### of the variable "key" into the string 
                ### use .pickle file extension (a pickle file is a serialized python objecT)
                f'{key}.pickle')
            
            ### Check if the cache exists already or if we should override caching
            if not os.path.exists(path) or override:
                
                ### Make jars directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                ### Run the compute function as the user did
                result = compute_function(*args, **kwargs)
                
                ### Pickle the object (save to file)
                ### open the file at filename
                with open(path, 'wb') as file:
                    
                    ### save the result without needing to recompute when loading
                    ### it back into Python
                    pickle.dump(result, file)
            
            ### if the file already exists/we are not overriding the cache
            else:
               
                ### Unpickle the object (load the cached result)
                with open(path, 'rb') as file:
                    
                    ### use pickle.load to unserialize the file back into a python object
                    result = pickle.load(file)
                    
            return result
        
        return compute_and_cache
    
    return compute_and_cache_decorator

# %% [markdown]
# ## STEP 2: Study site
# 
# For this analysis, you will use a watershed from the [Water Boundary
# Dataset](https://www.usgs.gov/national-hydrography/access-national-hydrography-products),
# HU12 watersheds (WBDHU12.shp).
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Download the Water Boundary Dataset for region 8 (Mississippi)</li>
# <li>Select watershed 080902030506</li>
# <li>Generate a site map of the watershed</li>
# </ol>
# <p>Try to use the <strong>caching decorator</strong></p></div></div>
# 
# We chose this watershed because it covers parts of New Orleans an is
# near the Mississippi Delta. Deltas are boundary areas between the land
# and the ocean, and as a result tend to contain a rich variety of
# different land cover and land use types.

# %%
### assign the hydrologic unit code
HUC_LEVEL = 12

### download, unzip, and read shapefile, using cache decorator
@cached(f'wbd_08_hu{HUC_LEVEL}_gdf')

### make a function to read in the file
def read_wbd_file(wbd_filename, cache_key):

    ### define the URL we're pulling data from
    wbd_url = (

        ### add base URL
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/"

        ### insert the name of the specific file we want 
        f"{wbd_filename}.zip"
    )

    ### download data and unzip it into directory
    wbd_dir = et.data.get_data(url = wbd_url)

    ### path to shapefile in the dir
    wbd_path = os.path.join(wbd_dir,
                            'Shape',
                            f'WBDHU{HUC_LEVEL}.shp')
    
    ### read shp as gdf
    wbd_gdf = gpd.read_file(wbd_path,
                            
                            ### use pyogrio library
                            engine = 'pyogrio')
    
    ### give us the gdf for the watershed boundaries
    return wbd_gdf


# %%
### open the shapefile using the read_wbd_file function that we created
wbd_gdf = read_wbd_file("WBD_08_HU2_Shape",
                        f'hu{HUC_LEVEL}')

# %%
wbd_gdf

# %%
### filter the shapefile to the specific watershed we're using

### define the gdf for the watershed by subsetting the gdf of the whole watershed dataset
delta_gdf = wbd_gdf[wbd_gdf[
                            
    ### filter the gdf to the row(s) with the watershed we want
    ### use "dissolve" to merge the geometries of all the rows matching the target watershed
    f'huc{HUC_LEVEL}'].isin(['080902030506'])].dissolve()




### check it out
delta_gdf

# %%
### Make a site map with satellite imagery in the background
(
    ### project the delta_gdf to Mercator
    delta_gdf.to_crs(ccrs.Mercator())

    ### use hvplot
    .hvplot(

        ### make the watershed transparent
        alpha = 0.35, fill_color = "white",

        ### add satellite basemap
        tiles = "EsriImagery",

        ### plot in Mercator
        crs = ccrs.Mercator())
    
    ### set plot size
    .opts(width = 600, height = 300)
)


# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-response"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div></div><div class="callout-body-container callout-body"><p>Write a 2-3 sentence <strong>site description</strong> (with
# citations) of this area that helps to put your analysis in context.</p></div></div>
# 
# 
# **YOUR SITE DESCRIPTION HERE**

# %% [markdown]
# ## STEP 3: Multispectral data
# 
# ### Step 3a: Search for data
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Log in to the <code>earthaccess</code> service using your Earthdata
# credentials:
# <code>python      earthaccess.login(persist=True)</code></li>
# <li>Modify the following sample code to search for granules of the
# HLSL30 product overlapping the watershed boundary from May to October
# 2023 (there should be 76 granules):
# <code>python      results = earthaccess.search_data(          short_name="...",          cloud_hosted=True,          bounding_box=tuple(gdf.total_bounds),          temporal=("...", "..."),      )</code></li>
# </ol></div></div>

# %%
### Log in to earthaccess
earthaccess.login(persist = True)

# %%
### Search for HLS granules we want
results = earthaccess.search_data(

    ### specify which dataset and spatial resolution we want 
    short_name = "HLSL30",


    ### specify that we're using cloud data
    cloud_hosted = True,


    ### use the bounding box from our watershed boundary
    bounding_box = tuple(delta_gdf.total_bounds),


    ### set the temporal range of the data
    temporal = ("2024-06", "2024-08")
)

# %%
results

# %% [markdown]
# ### Step 3b: Compile information about each granule
# 
# I recommend building a GeoDataFrame, as this will allow you to plot the
# granules you are downloading and make sure they line up with your
# shapefile. You could also use a DataFrame, dictionary, or a custom
# object to store this information.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>For each search result:
# <ol type="1">
# <li>Get the following information (HINT: look at the [‘umm’] values for
# each search result):
# <ul>
# <li>granule id (UR)</li>
# <li>datetime</li>
# <li>geometry (HINT: check out the shapely.geometry.Polygon class to
# convert points to a Polygon)</li>
# </ul></li>
# <li>Open the granule files. I recommend opening one granule at a time,
# e.g. with (<code>earthaccess.open([result]</code>).</li>
# <li>For each file (band), get the following information:
# <ul>
# <li>file handler returned from <code>earthaccess.open()</code></li>
# <li>tile id</li>
# <li>band number</li>
# </ul></li>
# </ol></li>
# <li>Compile all the information you collected into a GeoDataFrame</li>
# </ol></div></div>

# %%
### make a function to process all the granules from the earthaccess search
### and extract information for each granule

### define the function
def get_earthaccess_links(results):


    ### make and display a progress bar
    f = IntProgress(min = 0, max = len(results), description = 'Open granules')
    display(f)


    ### use a regular expression to extract tile_id and band from .tif files
    url_re = re.compile(
        r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')

    ### accumulate gdf rows from each granule
    link_rows = []

    ### loop over granules to extract info
    for granule in results:


        ### locate metadata (UMM = universal metadata model)
        info_dict = granule['umm']


        ### pull out unique identifier for the granule
        granule_id = info_dict['GranuleUR']


        ### extract date/time 
        datetime = pd.to_datetime(
            info_dict['TemporalExtent']['RangeDateTime']['BeginningDateTime'])


        ### extact boundary coordinates for granule
        points = (
            info_dict
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
            ['Boundary']['Points']
        )

        ### make polygon using coordinate points for granule
        geometry = Polygon(
            [(point['Longitude'],
              point['Latitude']) for point in points]
        )


        ### get url and open granule
        files = earthaccess.open([granule])

        ### loop through each file in the granule
        for file in files:


            ### use url regular expression to get url
            match = url_re.search(file.full_name)

            ### if match is found, append data to link_rows gdf we initialized
            if match is not None:
                link_rows.append(

                    ### makes a gdf with the granule's data and geometry
                    gpd.GeoDataFrame(
                        dict(

                            ### timestamp
                            datetime = [datetime],
                            
                            ### unique ID
                            tile_id = [match.group('tile_id')],

                            ### band name
                            band = [match.group('band')],

                            ### url
                            url = [file],

                            ### polygon
                            geometry = [geometry]
                        ),

                        ### set crs
                        crs = "EPSG:4326"

                    )
                )

        ### update progress bar after each granule is done
        f.value += 1

    ### combine into a single gdf   
    file_df = pd.concat(link_rows).reset_index(drop = True)


    ### return the final gdf file
    return file_df


# %%
granule = results[0]
granule

# %%
info_dict = granule['umm']
info_dict

# %%
### run the function to get granule search results
file_df = get_earthaccess_links(results)

# %%
type(file_df)

# %% [markdown]
# ### Step 3c: Open, crop, and mask data
# 
# This will be the most resource-intensive step. I recommend caching your
# results using the `cached` decorator or by writing your own caching
# code. I also recommend testing this step with one or two dates before
# running the full computation.
# 
# This code should include at least one **function** including a
# numpy-style docstring. A good place to start would be a function for
# opening a single masked raster, applying the appropriate scale
# parameter, and cropping.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>For each granule:
# <ol type="1">
# <li><p>Open the Fmask band, crop, and compute a quality mask for the
# granule. You can use the following code as a starting point, making sure
# that <code>mask_bits</code> contains the quality bits you want to
# consider: ```python # Expand into a new dimension of binary bits bits =
# ( np.unpackbits(da.astype(np.uint8), bitorder=‘little’)
# .reshape(da.shape + (-1,)) )</p>
# <p># Select the required bits and check if any are flagged mask =
# np.prod(bits[…, mask_bits]==0, axis=-1) ```</p></li>
# <li><p>For each band that starts with ‘B’:</p>
# <ol type="1">
# <li>Open the band, crop, and apply the scale factor</li>
# <li>Name the DataArray after the band using the <code>.name</code>
# attribute</li>
# <li>Apply the cloud mask using the <code>.where()</code> method</li>
# <li>Store the DataArray in your data structure (e.g. adding a
# GeoDataFrame column with the DataArray in it. Note that you will need to
# remove the rows for unused bands)</li>
# </ol></li>
# </ol></li>
# </ol></div></div>

# %%
### apply cached decorator to function
@cached('delta_reflectance_da_df')

### write function that computes reflectance data using 
### search results (df of urls) and watershed boundary
def compute_reflectance_da(search_results, boundary_gdf):

    """
    Connect to files using VSI, crop them, apply a cloud mask, and wrangle

    Return a single reflectance DataFrame with bands as columns
    and centroid coordinates and datetime as the index

    Parameters
    ==========
    search_results:list
        Search result links to the files (urls)
    boundary_gdf: gpd.GeoDataFrame
        Boundary used to crop the data
    """

    #### function to open raster from url, apply scale factor, and crop and mask data
    def open_dataarray(url, boundary_proj_gdf, scale = 1, masked = True):

        ### open raster data
        da = rxr.open_rasterio(url, masked = masked).squeeze() * scale

        ### reproject the boundary if needed to match the raster crs
        if boundary_proj_gdf is None:
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)

        ### crop raster to bounding box
        cropped = da.rio.clip_box(*boundary_proj_gdf.total_bounds)

        return cropped

    ### write function to apply a cloud mask
    def compute_quality_mask(da, mask_bits = [1, 2, 3]):

        """Mask out low quality data by bit"""

        ### unpack the bits to a new axis
        bits = (

            ### unpack each number into individual bits
            np.unpackbits(

                ### convert to 8-bit unsigned integer format
                da.astype(np.uint8),

                ### set the order of the bits
                bitorder = "little"
            
            ### reshape to match original data with an extra dimension for the bits
            ).reshape(da.shape + (-1, ))
        )

        ### grab bits we want and check if their flagged
        mask = np.prod(

            ### open bits
            bits[
                ...,
                            mask_bits] == 0,
                            axis = -1)

        ### return the mask
        return mask

    ### grab metadata
    file_df = get_earthaccess_links(search_results)

    ### store results for each granule
    granule_da_rows = []

    ### store projected boundary
    boundary_proj_gdf = None

    ### group the data by each granule
    group_iter = file_df.groupby(

        ### datetime and tile_id
        ['datetime', 'tile_id'])


    ### loop through each image and its metadata
    for (datetime, tile_id), granule_df in tqdm(group_iter):

        ### print status bar
        print(f'Processing granule {tile_id} {datetime}')

        ### find each granule's cloud mask file (fmask) url
        cloud_mask_url = (
            granule_df.loc[granule_df.band == 'Fmask', 'url']
            .values[0])

        ### open granule cloud mask
        cloud_masked_cropped_da = open_dataarray(cloud_mask_url, boundary_proj_gdf, masked = False)

        ### compute cloud mask
        cloud_mask = compute_quality_mask(cloud_masked_cropped_da)

        ### loop through each spectral band to open, crop, and mask the band
        da_list = []
        df_list = []

        ### loop through each band in the granule
        for i, row in granule_df.iterrows():

            ### only loop through the spectral bands
            if row.band.startswith('B'):

                ### open band's raster and scale to reflectance
                band_cropped = open_dataarray(
                    row.url, boundary_proj_gdf, scale = 0.0001)
                
                ### name the raster by the band
                band_cropped.name = row.band

                ### apply the cloud mask to the raster
                row['da'] = band_cropped.where(cloud_mask)

                ### append the row to granule_da_rows
                granule_da_rows.append(row.to_frame().T)


    ### reassemble the metadata df
    return pd.concat(granule_da_rows)

# %%
### apply the function
reflectance_da_df = compute_reflectance_da(results, delta_gdf)

# %%
### check out the dataframe
reflectance_da_df

# %% [markdown]
# ### Step 3d: Merge and Composite Data
# 
# You will notice for this watershed that:   
# 1. The raster data for each date are spread across 4 granules  
# 2. Any given image is incomplete because of clouds
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# 
# *   1. For each band:  
#     *   a. For each date:  
#         *   i. Merge all 4 granules  
#         *   ii. Mask any negative values created by interpolating from the nodata value of -9999 (`rioxarray`) should account for this, but doesn't appear to when merging. If you leave these values in, they will create problems later on
#     *   b. Concatenate the merged DataArrays along a new date dimension  
#     *   c. Take the mean in the date dimension to create a composite image that fills cloud gaps  
#     *   d. Add the band as a dimensions, and give the DataArray a name  
# *   2. Concatenate along the band dimension
# 

# %%
### apply cache decorator
@cached('delta_reflectance_da')

### create a function to merge and composite reflectance data from multiple granules
### end result: single, composite reflectance image for each spectral band
def merge_and_composite_arrays(granule_da_df):
    
    ### initialize a list to store composites after procesing
    da_list = []
    
    ### loop over each spectral band
    for band, band_df in granule_da_df.groupby('band'):

        ### list for storing merged data arrays (one per date)
        merged_das = []

        ### loop over date/time of image acquisition and merge granules for each date
        for datetime, date_df in band_df.groupby('datetime'):

            ### merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
           
            ### mask negative values (could be no data or invalid data)
            merged_da = merged_da.where(merged_da > 0)
            
            ### append to merged_das list we initialized
            merged_das.append(merged_da)
                   
        ### composite images across dates
        composite_da = xr.concat(merged_das,
                                 
                                 ### make a datetime dimension
                                 ### calculate median value across the datetimes for the pixel
                                 dim = 'datetime').median('datetime')

        ### assign band number to attribute of composite data array
        composite_da['band'] = int(band[1:])

        ### name the composite data array
        composite_da.name = 'reflectance'
        
        ### add processed and composite data array to list
        da_list.append(composite_da)

    ### concatenates composite data arrays for each band along band dimension
    return xr.concat(da_list, dim = 'band')


# %%
### call function to get final composite reflectance data 
reflectance_da = merge_and_composite_arrays(reflectance_da_df)

# %%
reflectance_da

# %% [markdown]
# ## STEP 4: K-means clustering
# 
# Cluster your data by spectral signature using the k-means algorithm.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Convert your DataArray into a <strong>tidy</strong> DataFrame of
# reflectance values (hint: check out the <code>.to_dataframe()</code> and
# <code>.unstack()</code> methods)</li>
# <li>Filter out all rows with no data (all 0s or any N/A values)</li>
# <li>Fit a k-means model. You can experiment with the number of groups to
# find what works best.</li>
# </ol></div></div>

# %%
### Convert spectral DataArray to a tidy DataFrame
model_df = (reflectance_da
            
            ### flattern the array into a long dataframe
            .to_dataframe()

            ### select the reflectance column
            .reflectance

            ### make the table wide: each row will be a pixel location
            ### and each column is a spectral band with the reflectance value
            .unstack('band')
            )

model_df

### filter out rows with no data
model_df = model_df.drop(columns = [10, 11]).dropna()
model_df

# %%
### data directory for csv
data_dir = os.path.join(
    # Home directory
    pathlib.Path.home(),
    # Earth analytics data directory
    'earth-analytics',
    'data',
    # Project directory
    'landcover-kmeans',
)
os.makedirs(data_dir, exist_ok=True)

### make path for csv
csv_path = os.path.join(data_dir, 'spectral_data.csv')

# %%
### save csv of spectral data to that path
model_df.to_csv(csv_path, index = True)

# %%
min_values = model_df.min()
max_values = model_df.max()

print(min_values)
print(max_values)

# %% [markdown]
# Now we're reading to fit the k-means clustering model. We can run the fit and prediction functions at the same time because we don't have target data.

# %%
### initialize k-means model 
k_means = KMeans(n_clusters = 5)

### fit model and predict
prediction = k_means.fit_predict(model_df.values)

### add the predicted values back to the model dataframe
model_df['clusters'] = prediction
model_df

# %% [markdown]
# ## STEP 5: Plot
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Create a plot that shows the k-means clusters next to an RGB image of
# the area. You may need to brighten your RGB image by multiplying it by
# 10. The code for reshaping and plotting the clusters is provided for you
# below, but you will have to create the RGB plot yourself!</p>
# <p>So, what is <code>.sortby(['x', 'y'])</code> doing for us? Try the
# code without it and find out.</p></div></div>

# %%
### make data array with bands to use for rgb: red, green, and blue
rgb = reflectance_da.sel(band = [4, 3, 2])


# %%
### plot rgb
(
    rgb.hvplot.rgb(y = 'y',
                   x = 'x',
                   bands = 'band',
                   data_aspect = 1,
                   xaxis = None,
                   yaxis = None)
)

# %%
### stretch values
rgb_uint8 = (rgb * 255).astype(np.uint8).where(rgb != np.nan)
rgb_uint8

# %%
### plot rgb from 0-255
(
    rgb_uint8.hvplot.rgb(y = 'y',
                   x = 'x',
                   bands = 'band',
                   data_aspect = 1,
                   xaxis = None,
                   yaxis = None)
)

# %%
### enhance brightness
rgb_uint8_bright = rgb_uint8 * 10
rgb_uint8_bright

# %%
### plot brighter rgb
(
    rgb_uint8_bright.hvplot.rgb(y = 'y',
                   x = 'x',
                   bands = 'band',
                   data_aspect = 1,
                   xaxis = None,
                   yaxis = None)
)

# %%
### cap saturation
rgb_sat = rgb_uint8_bright.where(rgb_uint8_bright < 255, 255)

### replot
(
    rgb_sat.hvplot.rgb(y = 'y',
                   x = 'x',
                   bands = 'band',
                   data_aspect = 1,
                   xaxis = None,
                   yaxis = None)
)

# %%
### plot the clusters from kmeans
(
    model_df.clusters.to_xarray().hvplot(
        x = 'x',
        y = 'y',
        data_aspect = 1, 
        xaxis = None, 
        yaxis = None)

)

# %%
### plot the clusters from kmeans, deal with sorting
(
    model_df.clusters.to_xarray().sortby(['x', 'y']).hvplot(
        x = 'x',
        y = 'y',
        data_aspect = 1, 
        xaxis = None, 
        yaxis = None)

)

# %%
### plot the k-means clusters
(
    rgb_sat.hvplot.rgb(y = 'y',
                   x = 'x',
                   bands = 'band',
                   data_aspect = 1,
                   xaxis = None,
                   yaxis = None)
    + 
    model_df.clusters.to_xarray().sortby(['x', 'y']).hvplot(
        cmap = "Colorblind", aspect = 'equal') 
)

# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Don’t forget to interpret your plot!</p></div></div>

# %% [markdown]
# **YOUR PLOT HEADLINE AND DESCRIPTION HERE**

# %% [markdown]
# ## Hierarchical clustering

# %%
### load packages 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# %%
### open data

### data directory for csv
data_dir = os.path.join(
    # Home directory
    pathlib.Path.home(),
    # Earth analytics data directory
    'earth-analytics',
    'data',
    # Project directory
    'landcover-kmeans',
)
os.makedirs(data_dir, exist_ok=True)

### make path for csv
csv_path = os.path.join(data_dir, 'spectral_data.csv')

### open csv
spectral_dat = pd.read_csv(csv_path)

# %%
spectral_dat

# %%
### get spectral bands
bands_cols = spectral_dat.iloc[:, 2:10]
bands_cols

# %%
agg_clustering = AgglomerativeClustering(n_clusters = 4)
y_pred = agg_clustering.fit_predict(bands_cols)


