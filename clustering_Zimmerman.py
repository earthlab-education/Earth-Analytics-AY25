# %% [markdown]
# # Land cover classification at the Mississppi Delta
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
# You will use the [harmonized Sentinal/Landsat multispectral
# dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).
# You can access the data with an [Earthdata
# account](https://www.earthdata.nasa.gov/learn/get-started) and the
# [`earthaccess` library from
# NSIDC](https://github.com/nsidc/earthaccess):
# 
# ## STEP 1: SET UP
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Import all libraries you will need for this analysis</li>
# <li>Configure GDAL parameters to help avoid connection errors:
# <code>python      os.environ["GDAL_HTTP_MAX_RETRY"] = "5"      os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"</code></li>
# </ol></div></div>

# %%
import os
import pickle
import re
import warnings

import cartopy.crs as ccrs
import earthaccess
import earthpy as et
import geopandas as gpd
import geoviews as gv
import hvplot.pandas
import hvplot.xarray
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrmerge
from tqdm.notebook import tqdm
import xarray as xr
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import pathlib 

os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

warnings.simplefilter('ignore')

# %% [markdown]
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

# %%
def cached(func_key, override=True):
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
            # Add an identifier from the particular function call
            if 'cache_key' in kwargs:
                key = '_'.join((func_key, kwargs['cache_key']))
            else:
                key = func_key

            path = os.path.join(
                et.io.HOME, et.io.DATA_NAME, 'jars', f'{key}.pickle')
            
            # Check if the cache exists already or override caching
            if not os.path.exists(path) or override:
                # Make jars directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # Run the compute function as the user did
                result = compute_function(*args, **kwargs)
                
                # Pickle the object
                with open(path, 'wb') as file:
                    pickle.dump(result, file)
            else:
                # Unpickle the object
                with open(path, 'rb') as file:
                    result = pickle.load(file)
                    
            return result
        
        return compute_and_cache
    
    return compute_and_cache_decorator

# %%
# Create Reproducible File Paths
data_dir = os.path.join(
    pathlib.Path.home(),
    'earth-analytics',
    'data',
    'clustering')
os.makedirs(data_dir, exist_ok=True)

# %% [markdown]
# ## STEP 2: STUDY SITE
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
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-response"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div></div><div class="callout-body-container callout-body"><p>Write a 2-3 sentence <strong>site description</strong> (with
# citations) of this area that helps to put your analysis in context.</p></div></div>

# %% [markdown]
# **SITE DESCRIPTION**
# 
# Water Boundary Dataset Region 8 covers the Mississippi, and our watershed (080902030506) is part of the Mississippi Delta.
# 
# The Mississippi River Delta is the confluence of the Mississippi River with the Gulf of Mexico in Louisiana. It is the 7th largest river delta on earth and one of the largest coastal wetlands in the United States. Covering almost three million acres, the delta can be traced back almost 100 million years and it has payed a key role in the evolution of the Louisiana coastline. 
# 
# The Mississippi River Delta is home to more that two million people and is of rich cultural and ecological significance. The Delta has a diverse array of natural habitats significant to the region [[1]]. "Louisiana's wetlands are one of the nation's most productive and important natural assets. Consisting of natural levees, barrier islands, forests, swamps, and fresh, brackish and saline marshes, the region is home to complex ecosystems and habitats that are necessary for sustaining its unique and vibrant nature"[[2]](https://web.archive.org/web/20140521180243/http://www.lca.gov/Learn.aspx).
# 
# A report released by the USGS in 2009 looking and changing land cover change in the lower Mississippi Valley identfied the following land cover types [[3]](https://pubs.usgs.gov/of/2009/1280/pdf/of2009-1280.pdf): 
# * Water                     
# * Forest
# * Developed                 
# * Grassland/Shrubland
# * Mechanically Disturbed    
# * Agriculture
# * Mining                    
# * Wetland
# * Barren                    
# * Non-Mechanically Disturbed
# 
# A 2024 article in Landscape Ecology surveyed the delta's land cover types from 2008 to 2021. The study found that while the areas of cultivated land did not change, the land cover type of 72% of the land did change in a process the likened to a 'shifting moasic'[[4]](https://link.springer.com/article/10.1007/s10980-024-01797-0).
# 
# **DATA DESCRIPTION**
# The Watershed Boundary Dataset (WBD) is available in shapefile and file geodatabase formats[[5]](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/).
# 
# This nationwide dynamic service displays the WBD at scales from 1:18M and below. It provides ability to either show all WBD levels at all scale ranges or the most appropriate HUC level at a given scale (default setting). The data is updated quarterly and it supports access to attributes and dynamic styling (the ability to change the style representation in ESRI clients). This map service is also available as a OGC Web Map Service (WMS) and a Web Feature Service (WFS)[[6]](https://www.usgs.gov/national-hydrography/access-national-hydrography-products)].
# 
# ### References:
# 1. [Wikipedia - Mississippi River Delta, 2024](https://en.wikipedia.org/wiki/Mississippi_River_Delta)
# 2. Louisiana Office of Coastal Protection and Restoration, US Army Corps of Engineers, "Louisana's Coastal Area - Ecosystem Restoration", 2014.
# 3. Karstensen, K. A., & Sayler, K. (2009). Land-cover change in the Lower Mississippi Valley, 1973-2000 (No. 2009-1280). US Geological Survey.
# 4. Heintzman, L.J., McIntyre, N.E., Langendoen, E.J. et al. Cultivation and dynamic cropping processes impart land-cover heterogeneity within agroecosystems: a metrics-based case study in the Yazoo-Mississippi Delta (USA). Landsc Ecol 39, 29 (2024). https://doi.org/10.1007/s10980-024-01797-0.
# 5. US Department of Interior, USGS, 'National Map: Staged Products Directory", last modified 4 Aug 2023. [Wastershed Boundary Dataset, 2021](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/)
# 6. USGS, US Department of the Interior, "USGS Hydrography Products", 2021. [USGS Hydrography Products, 2021](https://www.usgs.gov/national-hydrography/access-national-hydrography-products)

# %%
@cached('wbd_08')
def read_wbd_file(wbd_filename, huc_level, cache_key):
    # Download and unzip
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Hydrography/WBD/HU2/Shape/"
        f"{wbd_filename}.zip")
    wbd_dir = et.data.get_data(url=wbd_url)
                  
    # Read desired data
    wbd_path = os.path.join(wbd_dir, 'Shape', f'WBDHU{huc_level}.shp')
    wbd_gdf = gpd.read_file(wbd_path, engine='pyogrio')
    return wbd_gdf

huc_level = 12
wbd_gdf = read_wbd_file(
    "WBD_08_HU2_Shape", huc_level, cache_key=f'hu{huc_level}')


print(wbd_gdf.columns)

delta_gdf = (
    wbd_gdf[wbd_gdf[f'huc{huc_level}']
    .isin(['080902030506'])]
    .dissolve()
)

(
    delta_gdf.to_crs(ccrs.Mercator())
    .hvplot(
        alpha=.2, fill_color='white', 
        tiles='EsriImagery', crs=ccrs.Mercator())
    .opts(width=600, height=300)
)

# %% [markdown]
# ## STEP 3: MULTISPECTRAL DATA
# 
# ### Search for data
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
# Log in to earthaccess
earthaccess.login(persist=True)

# Search for HLS tiles
results = earthaccess.search_data(  
    short_name="HLSL30",          
    cloud_hosted=True,          
    bounding_box=tuple(delta_gdf.total_bounds),  # create a bounding box
    temporal=("2024-06", "2024-08"), # how do we find out the correct syntax/format for code inputs?
)

# Conform the Contents
num_granules = len(results)
print(f'Number of Granules found', {num_granules})

# %%
print(results)

# %% [markdown]
# ### Compile information about each granule
# 
# Each granule is a dictionary containing metadata such as:
# 
#     GranuleUR → A unique identifier for the granule
#     TemporalExtent → The time when the data was captured
#     SpatialExtent → The geographic area covered
#     umm (Unified Metadata Model) → Stores all related metadata
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
def get_earthaccess_links(results): 
    # Define the regex pattern - you can do this in chat gpt or the Regex Generator
    url_re = re.compile(
        r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')

    # Loop through each granule
    link_rows = [] # Create a list for metadata
    url_dfs = []   # do we actually use this anywhere??
    for granule in tqdm(results): # Note tqdm is progress bars for loops and long-running processes
        # Get granule information by extracting metadata
        info_dict = granule['umm'] # Retrieve metadata stored in the 'umm' field
        granule_id = info_dict['GranuleUR']
        datetime = pd.to_datetime( # Convert starting timestamp to a datetime object
            info_dict
            ['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        # Extract geospatial information
        points = (
            info_dict
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
            ['Boundary']['Points'])
        geometry = Polygon(
            [(point['Longitude'], point['Latitude']) for point in points])
        
        # Get URL
        files = earthaccess.open([granule])

        # Build metadata DataFrame
        for file in files:
            match = url_re.search(file.full_name)
            if match is not None:
                 # Create a geodata frame for each file
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group('tile_id')],
                            band=[match.group('band')],
                            url=[file],
                            geometry=[geometry]
                        ),
                        crs="EPSG:4326"
                    )
                )

    # Concatenate metadata DataFrame (combine all extracted data)
    file_df = pd.concat(link_rows).reset_index(drop=True)
    return file_df

# %% [markdown]
# ### Open, crop, and mask data
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
from tqdm import tqdm  # Progress bar for loops
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import numpy as np

def get_earthaccess_links(results):
    """
    Extracts metadata and URLs from Earthdata search results.
    """
    url_re = re.compile(r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')
    
    link_rows = []
    
    for granule in tqdm(results):
        info_dict = granule['umm']
        granule_id = info_dict['GranuleUR']
        datetime = pd.to_datetime(info_dict['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        points = info_dict['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
        geometry = Polygon([(point['Longitude'], point['Latitude']) for point in points])
        
        # Open files from Earthdata
        files = earthaccess.open([granule])
        
        for file in files:
            match = url_re.search(file.full_name)
            if match is not None:
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group('tile_id')],
                            band=[match.group('band')],
                            url=[file],
                            geometry=[geometry]
                        ),
                        crs="EPSG:4326"
                    )
                )
    
    # Combine all extracted metadata into a single DataFrame
    file_df = pd.concat(link_rows).reset_index(drop=True)
    return file_df

@cached('delta_reflectance_da_df')
def compute_reflectance_da(search_results, boundary_gdf):
    """
    Processes remote sensing imagery by applying cloud masks, cropping, and assembling data into a DataFrame.
    """
    def open_dataarray(url, boundary_proj_gdf, scale=1, masked=True):
        """
        Opens a raster file, optionally masks and scales it, and clips it to the study boundary.
        """
        da = rxr.open_rasterio(url, masked=masked).squeeze() * scale  # Open and apply scale factor
        
        # Reproject the boundary if needed
        if boundary_proj_gdf is None:
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)
        
        # Clip the raster to the bounding box of the boundary
        cropped = da.rio.clip_box(*boundary_proj_gdf.total_bounds)
        return cropped
    
    def compute_quality_mask(da, mask_bits=[1, 2, 3]):
        """
        Applies a cloud mask by filtering out low-quality data based on specific bit flags.
        """
        bits = (
            np.unpackbits(
                da.astype(np.uint8), bitorder='little'
            ).reshape(da.shape + (-1,))
        )
        
        # Keep pixels where none of the selected mask bits are flagged
        mask = np.prod(bits[..., mask_bits] == 0, axis=-1)
        return mask
    
    # Extract file metadata and URLs
    file_df = get_earthaccess_links(search_results)
    
    granule_da_rows = []  # List to store processed data arrays
    boundary_proj_gdf = None  # Initialize reprojected boundary

    # Group images by timestamp and tile ID
    group_iter = file_df.groupby(['datetime', 'tile_id'])
    for (datetime, tile_id), granule_df in tqdm(group_iter):
        print(f'Processing granule {tile_id} {datetime}')
        
        # Extract cloud mask from Fmask band
        cloud_mask_url = (
            granule_df.loc[granule_df.band == 'Fmask', 'url']
            .values[0]
        )
        cloud_mask_cropped_da = open_dataarray(cloud_mask_url, boundary_proj_gdf, masked=False)
        
        # Compute cloud mask
        cloud_mask = compute_quality_mask(cloud_mask_cropped_da)
        
        # Process each spectral band
        for i, row in granule_df.iterrows():
            if row.band.startswith('B'):  # Filter only spectral bands
                band_cropped = open_dataarray(row.url, boundary_proj_gdf, scale=0.0001)
                band_cropped.name = row.band  # Name the raster
                
                # Apply cloud mask
                row['da'] = band_cropped.where(cloud_mask)
                granule_da_rows.append(row.to_frame().T)
    
    # Compile processed data into a single DataFrame
    return pd.concat(granule_da_rows)

# Execute the function and store the result
reflectance_da_df = compute_reflectance_da(results, delta_gdf)


# %% [markdown]
# ### Merge and Composite Data
# 
# You will notice for this watershed that: 1. The raster data for each
# date are spread across 4 granules 2. Any given image is incomplete
# because of clouds
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li><p>For each band:</p>
# <ol type="1">
# <li><p>For each date:</p>
# <ol type="1">
# <li>Merge all 4 granules</li>
# <li>Mask any negative values created by interpolating from the nodata
# value of -9999 (<code>rioxarray</code> should account for this, but
# doesn’t appear to when merging. If you leave these values in they will
# create problems down the line)</li>
# </ol></li>
# <li><p>Concatenate the merged DataArrays along a new date
# dimension</p></li>
# <li><p>Take the mean in the date dimension to create a composite image
# that fills cloud gaps</p></li>
# <li><p>Add the band as a dimension, and give the DataArray a
# name</p></li>
# </ol></li>
# <li><p>Concatenate along the band dimension</p></li>
# </ol></div></div>

# %%
@cached('delta_reflectance_da')
def merge_and_composite_arrays(granule_da_df):
    # Merge and composite and image for each band
    da_list = []
    for band, band_df in tqdm(granule_da_df.groupby('band')):
        merged_das = []
        for datetime, date_df in tqdm(band_df.groupby('datetime')):
            # Merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
            # Mask negative values
            merged_da = merged_da.where(merged_da>0)
            # Add to your data array
            merged_das.append(merged_da)
            
        # Composite images across dates
        composite_da = xr.concat(merged_das, dim='datetime').median('datetime')
        composite_da['band'] = int(band[1:])
        composite_da.name = 'reflectance'
        da_list.append(composite_da)
     # Combine   
    return xr.concat(da_list, dim='band')

reflectance_da = merge_and_composite_arrays(reflectance_da_df)
reflectance_da

# %% [markdown]
# ## STEP 4: K-MEANS
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
# Convert spectral DataArray to a tidy DataFrame
    # Steps:
    # Flatten the DataArray so that each pixel's value is a row.
    # Extract coordinates (e.g., longitude, latitude, time).
    # Reshape the data so each band becomes a column.
    # Convert to a pandas DataFrame.

reflectance_df = (
    reflectance_da.to_dataframe()  # Convert to long format
    .reset_index()  # Convert index (band, x, y) into columns
    .pivot(index=["y", "x"], columns="band", values="reflectance")  # Make bands columns
    .reset_index()  # Flatten DataFrame
)
# Display the final tidy DataFrame
print(reflectance_df)


# %%
reflectance_df = reflectance_da.to_dataframe().reflectance.unstack('band')

# Drop columns 10-11 and drop Nan
reflectance_df = reflectance_df.drop(columns=[10, 11], axis=1)

# Drop rows with NaN values
reflectance_df = reflectance_df.dropna()

# Running the fit and predict functions at the same time for a k means model.
# We can do this since we don't have test data.
prediction = KMeans(n_clusters=6).fit_predict(reflectance_df.values)

# Add the predicted values back to the model DataFrame
reflectance_df['clusters'] = prediction
reflectance_df

# %%
reflectance_df

# %%
# double check the data
min_values = reflectance_df.min()
max_values = reflectance_df.max()

print(min_values)
print(max_values)

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray as rxr
import earthpy as et
import earthpy.plot as ep
import xarray as xr

### packages for scikit-learn
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans ### for kmeans clustering
from sklearn.decomposition import PCA ### for doing PCA
from sklearn.metrics import silhouette_score ### calculate silhouette score

### import toy data
import seaborn as sns

# %%
# NOTE: I am doing this mostly out of curiousity, not because it is essentail for classification.

# Initial Data visualisation (You can learn more about kmeans and other scikit-learn stuff at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
### how many types of clusters  are there?
np.unique(reflectance_df.clusters) 

### make a pairplot to look at continuous variables
sns.pairplot(reflectance_df, hue = "clusters")

# %%
# try clustering on different variables
### make a k-means model with 8 clusters (here, the hyperparameter k = 8)
k_means = KMeans(n_clusters = 8)

### fit the model to the data (specify which variables we want)
k_means.fit(reflectance_df[[1,
                        #  '2',
                        #  '3',
                        #  '4',
                        #  '5',
                        #  '6',
                        #  '7',
                         9]])

### add the cluster labels to the dataframe
reflectance_df['k_means_labels'] = k_means.labels_

### check it out
reflectance_df

# %%
# Visuaize
sns.pairplot(reflectance_df, 
             hue="k_means_labels",
             vars=[1, 2, 3, 4, 5, 6, 7, 9])

# Add title
plt.title("Pairplot of Reflectance Values in the Mississippi Delta")

# Display the plot
plt.show()

# %% [markdown]
# # PCA

# %%
### Run PCA
### n_components: we tell it how many components to identify (pull out the n most important components)
### if we don't set n_components, it will keep all the components
pca = PCA()

### fit the PCA to the data (specify which variables we want)
pca.fit(reflectance_df[[1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        9,
                        ]]
                        )
### check out the components -- this will spit out the coefficients for the linear combinations of the variables
pca.components_

# %%
### check out the variation explained by the PCAs
pca.explained_variance_ratio_

# %% [markdown]
# ### The first component explains a very large 96% of the variation in components

# %%
### extract the first PCA component
b = pca.components_[0]

### make column in penguins df for component
reflectance_df['component'] = (

    ### multiply the original variables by their weight in the first component
    reflectance_df[[1]].values * b[0] 
    + reflectance_df[[2]].values * b[1]
    + reflectance_df[[3]].values * b[2]
    + reflectance_df[[4]].values * b[3]
    + reflectance_df[[5]].values * b[4]
    + reflectance_df[[6]].values * b[5]
    + reflectance_df[[7]].values * b[6]
    + reflectance_df[[9]].values * b[7]
)

### check it out
reflectance_df

# %%
### make a k-means model with 3 clusters (here, the hyperparameter k = 3
k_means_pca = KMeans(n_clusters = 3)

### fit the model to just the first principal component
k_means_pca.fit(reflectance_df[['component']])

### add the cluster labels to the dataframe
reflectance_df['k_means_labels_pca'] = k_means_pca.labels_

### check it out
reflectance_df

# %%
# Your existing pairplot
pairplot = sns.pairplot(reflectance_df, 
                         hue="k_means_labels_pca",
                         vars=[1, 2, 3, 4, 5, 6, 7, 9])

# Add a title
plt.suptitle("Pairplot of Reflectance Data", fontsize=16)

# Show the plot
plt.show()

# %%

### this time we're looking at the silhouette score, so we want to accumulate it into a list
silhouette = []

### make list of k values to loop through
k_list = list(range(2, 8))

### loop through different k values
for k in k_list:

    ### make model with k clusters
    k_means = KMeans(n_clusters = k, n_init = 'auto')

    ### identify the variables to include
    model_vars = (
        reflectance_df
        [[1, 2, 3, 4, 5, 6, 7, 9]])

    ### fit the model
    k_means.fit(model_vars)
        
    ### calculate silhouette score and add it to the list we initialized (along with the corresponding k value)
    silhouette.append(silhouette_score(model_vars, k_means.labels_))

### check it out
silhouette

# %% [markdown]
# ## STEP 5: PLOT
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Create a plot that shows the k-means clusters next to an RGB image of
# the area. You may need to brighten your RGB image by multiplying it by
# 10. The code for reshaping and plotting the clusters is provided for you
# below, but you will have to create the RGB plot yourself!</p>
# <p>So, what is <code>.sortby(['x', 'y'])</code> doing for us? Try the
# code without it and find out.</p></div></div>

# %%
# Select R, G, B and transform to uint8
rgb = reflectance_da.sel(band=[4, 3, 2])

# restore the brigthness with control
rgb_uint8 = (rgb * 255).astype(np.uint8).where(rgb!=np.nan)
rgb_bright = rgb_uint8 * 10
rgb_sat = rgb_bright.where(rgb_bright < 255, 255)


# Visualize with `hvplot`
(
    rgb_sat.hvplot.rgb( 
        x='x', y='y', bands='band',
        data_aspect=1,
        xaxis=None, yaxis=None)
    + 
    reflectance_df.clusters.to_xarray().sortby(['x', 'y']).hvplot(
        cmap="Colorblind", aspect='equal') 
)

# %%
# Make and array with just RGB (which is bands 4, 3, 2)
rgb = reflectance_da.sel(band=[4, 3, 2])
rgb_uint8 = (rgb * 255).astype(np.uint8).where(rgb!=np.nan)
rgb_bright = rgb_uint8 * 10
rgb_sat = rgb_bright.where(rgb_bright < 255, 255)

# Plot it

# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Don’t forget to interpret your plot!</p></div></div>

# %% [markdown]
# # Diverse Array of Land Use Types Charactaristic of the Mississippi Delta
# 
# 
# While there are six separate clusters used in my k-means plot. Some represent very large areas (Dark Blue and Pink) while other do not appear to be largely unused (Black) It is important to note that this clusters represent spectral groups that MAY correspond with certain land cover types. 
# 
# * Dark Blue (0) - appears to be associated with very green flat patches of land (swamp?)
# * Orange (1) - appears along borders of what appear to be waterways or topographical ridges 
# * Dark green (2) - limited presence in certain areas on the North end of the image
# * Light Blue (3) - loosely correleated with waterways
# * Pink (4) - Large and flatter open areas (grasses?)
# * Black (5) - Uncommon, ,osty along areas of orange
# 
# Since this data is only being compared to an RBG image, the next sensible step would be to compare it to a satalite image of the area. Ground truth data would also support improved classification. 


