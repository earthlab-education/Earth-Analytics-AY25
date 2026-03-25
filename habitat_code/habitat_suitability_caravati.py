# %% [markdown]
# # Habitat suitability under climate change
# 
# Our changing climate is changing where plant species can live,
# and conservation and restoration practices will need to take
# this into
# account.
# 
# In this coding challenge, you will create a habitat suitability model
# for a terrestrial plant species of your choice that lives in the contiguous United States
# (CONUS). We have this limitation because the downscaled climate data we
# suggest, the [MACAv2 dataset](https://www.climatologylab.org/maca.html),
# is only available in the CONUS – if you find other downscaled climate
# data at an appropriate resolution, you are welcome to choose a different
# study area. If you don’t have anything in mind, you can take a look at
# [*Sorghastrum nutans*](https://www.gbif.org/species/2704414), a grass native to North America. In the past 50
# years, its range has moved
# northward.
# 
# Your suitability assessment will be based on combining multiple data
# layers related to soil, topography, and climate, then applying a fuzzy logic model across the different data layers to generate habitat suitability maps. 
# 
# You will need to create a **modular, reproducible, workflow** using functions and loops.
# To do this effectively, we recommend planning your code out in advance
# using a technique such as a pseudocode outline or a flow diagram. We
# recommend breaking each of the blocks below out into multiple steps. It
# is unnecessary to write a step for every line of code unless you find
# that useful. As a rule of thumb, aim for steps that cover the major
# structures of your code in 2-5 line chunks.

# %% [markdown]
# ## Step 0. Set-Up

# %%
#suppress version warning
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)

# %%
# Import Libraries

# File & Path Management
from getpass import getpass     # Securely prompt for passwords
import os                       # OS utilities (paths, environment)
from glob import glob           # File pattern matching
import pathlib                  # Filesystem paths
from pathlib import Path        # Object-oriented filesystem paths
import zipfile                  # Read/write ZIP archives

# Data Handling & Computation
import math                     # Basic math functions
from math import ceil, floor    # Specific math functions
import numpy as np              # Numerical operations
import pandas as pd             # DataFrames and tabular data
import rioxarray as rxr         # Raster data I/O with xarray
import rioxarray.merge as rxrm  # Merge multiple raster datasets
import xrspatial                # Spatial analysis with xarray
import xarray as xr             # N-dimensional arrays with labels

# Geospatial Libraries
import fiona                    # Reading/writing vector data
import geopandas as gpd         # Geospatial DataFrames
import pyproj                   # Coordinate projections and transformations
from rasterio.features import rasterize               # Rasterize vector geometries
from shapely.geometry import MultiPolygon, Polygon  # Geometric objects
from shapely.geometry import box                        # Create bounding boxes

# Data Access & APIs
import earthaccess              # NASA Earthdata access
import requests                 # HTTP requests
from pygbif import occurrences as occ  # GBIF occurrences API
from pygbif import species      # GBIF species API
import time                     # Timing operations
from tqdm import tqdm           # Progress bars

# Visualization
import geoviews as gv           # Geospatial visualizations with HoloViews
import holoviews as hv          # Interactive visualizations
from holoviews import opts      # Plotting options
import hvplot.pandas            # HoloViews plotting for pandas
import hvplot.xarray            # HoloViews plotting for xarray
import matplotlib.pyplot as plt # Static 2D plotting

# Initialize HoloViews
hv.extension('bokeh')           # Enable Bokeh backend for HoloViews

# %% [markdown]
# ## STEP 1: Study overview
# 
# Before you begin coding, you will need to design your study.
# 
# ### Step 1a: Select a species
# Select the terrestrial plant species you want to study, and research its habitat parameters in scientific studies or other reliable sources. Individual studies may not have the breadth needed for this purpose, so take a look at reviews or overviews of the data. Do **not** just look at an AI-generated summary! In the US, the National Resource Conservation Service can have helpful fact sheets about different species. University Extension programs are also good resources for summaries.</p>
# <p>Based on your research, select soil, topographic, and climate variables that you can use to determine if a particular location and time period is a suitable habitat for your species.</p></div></div>
# 
# **Reflect and respond**: 
# Write a description of your species. What habitat is it found in? What is its geographic range? What, if any, are conservation threats to the species? What data will shed the most light on habitat suitability for this species? 
# 
# What core scientific question do you hope to answer about potential future changes in habitat suitability? Don't forget to cite your sources!

# %% [markdown]
# Whitebark Pine is a protected "pioneer" species in alpine habitats that both relies on wildlife interactions for survival and provides essential resources for other species. My expectation is suitability will move up the peaks I am looking at and in extreme cases might move off the top of the peak entirely as temperatures rise.
# 
# Research links
# - https://www.nps.gov/articles/000/whitebark-pine-klamath-network.htm
# - https://www.nrcs.usda.gov/plantmaterials/idpmcpg10870.pdf
# - https://www.fws.gov/species/whitebark-pine-pinus-albicaulis
# - https://www.conifers.org/pi/Pinus_albicaulis.php

# %%
# set base directory
habitat_dir = Path.home() / "Data" / "Earth Analytics" / "habitat-proj"
habitat_dir.mkdir(parents=True, exist_ok=True)

# %%
# Specific directory for species data
gbif_dir = habitat_dir / 'gbif-data'
gbif_dir.mkdir(exist_ok=True)

# %% [markdown]
# > EDIT SPECIES NAME HERE

# %%
#set species name
species_name = "Pinus albicaulis"

#get species meta data
backbone = species.name_backbone(species_name)
backbone

# %%
# reset credentials if needed
reset = False

# Request and store username
if (not ('GBIF_USER'  in os.environ)) or reset:
    os.environ['GBIF_USER'] = input('GBIF username:')

# Securely request and store password
if (not ('GBIF_PWD'  in os.environ)) or reset:
    os.environ['GBIF_PWD'] = getpass('GBIF password:')
    
# Request and store account email address
if (not ('GBIF_EMAIL'  in os.environ)) or reset:
    os.environ['GBIF_EMAIL'] = input('GBIF email:')

'GBIF_EMAIL' in os.environ and 'GBIF_USER' in os.environ and 'GBIF_PWD' in os.environ

# %%
#download species data

species_key = backbone['usageKey']

species_slug = species_name.replace(" ","_").lower()
gbif_pattern = gbif_dir / f"{species_slug}_gbif.csv"

if not gbif_pattern.exists():

    # only submit a download request to GBIF once
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:

        # submit a query to GBIF
        gbif_query = occ.download([

            # add your species key here
            f"speciesKey = {species_key}",

            # filter out results that are missing coordinates
            "hasCoordinate = True",

            # choose a year to include (optional)
            # "year = 2024",
        ])
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]

    # Wait for the download to build
    download_key = os.environ['GBIF_DOWNLOAD_KEY']

    # use the occurrence command module in pygbif to get the metadata
    wait = occ.download_meta(download_key)['status']

    # check if the status of the download = "SUCCEEDED"
    # wait and loop through until it finishes
    while not wait=='SUCCEEDED':
        wait = occ.download_meta(download_key)['status']

        # don't want to re-query the API in the loop too frequently
        time.sleep(5)

    # Download GBIF data when it's ready
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'], 
        path=gbif_dir)

    # Unzip GBIF data using the zipfile package
    with zipfile.ZipFile(download_info['path']) as z:
        z.extractall(path=gbif_dir)

# %%
# Find the extracted .csv file path (take the first result)
gbif_path = next(gbif_dir.glob("*.csv"))
gbif_path.rename(gbif_pattern)

# %%
#confirm path
gbif_path

# %%
# format to df
gbif_df = pd.read_csv(
    gbif_pattern, 
    delimiter='\t',
    index_col='gbifID'
    # optional, set columns to be saved
    # usecols=['gbifID', 'decimalLatitude', 'decimalLongitude', 'month']
    )

#test
gbif_df

# %%
#convert to gdf
gbif_gdf = (
    gpd.GeoDataFrame(
       gbif_df, 
        geometry=gpd.points_from_xy(
           gbif_df.decimalLongitude, 
           gbif_df.decimalLatitude), 
        crs="EPSG:4326")
    # Select the desired columns
    #[['month','geometry']]
)

#test
gbif_gdf

# %%
#look at columns
gbif_gdf.columns

# %%
#look at crs

gbif_gdf.crs

# %%
#look at species distribution
species_overlay = gbif_gdf.hvplot(
    geo=True,
    color='red',
    size=5,
    alpha=0.7,
    tiles = 'EsriImagery',
    # xlim = (-150,-90),
    # ylim = (25,60)
)

#plot
species_overlay


# %% [markdown]
# ### Step 1b: Select study sites
# Based on your research and/or range maps you find online, select at least 2 sites where your species occurs. These could be national parks, national forests, national grasslands or other protected areas, or some other area you're interested in. You can access protected area polygons from the [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview), [national grassland units](https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip), etc.
# 
# When selecting your sites, you might want to look for places that are marginally habitable for this species, since those locations will be most likely to show changes due to climate.
# 
# Generate a site map for each location.

# %% [markdown]
# **Reflect and Respond**: 
# Write a site description for each of your sites, or for all of your sites as a group if you have chosen a large number of linked sites. What
# differences or trends in habitat suitability over time do you expect to see among your sites?

# %% [markdown]
# Lassen National Forrest and Shasta National Forrest are both protected areas containing mountain peaks with large numbers of observed observations. The mountainous, rocky soil and changes in elevation should provide an interesting comparison.

# %%
#set directory for sites
sites_dir = habitat_dir / "study-sites"
sites_dir.mkdir(exist_ok=True)

# %% [markdown]
# >SET STUDY STATE HERE

# %%
#study state
STATE = "CA"

# %%
#download protected area files for borders
# item id and filename for download url
ITEM_ID = "6759abcfd34edfeb8710a004"
FILENAME = f"PADUS4_1_State_{STATE}_GDB_KMZ.zip"

# url
url = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}?name={FILENAME}"

#set path
output_path = sites_dir / FILENAME

# call API
if not output_path.exists():
    with requests.get(url, stream = True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

#test path
output_path

# %%
# unzip
zip_path = Path(output_path)

extract_dir = sites_dir / f"{STATE}_files"

if not extract_dir.exists() or not any(extract_dir.iterdir()):
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zr:
        zr.extractall(extract_dir)

# %%
#set path
pa_path = extract_dir / f"PADUS4_1_State{STATE}.gdb"

#list layers
layers = fiona.listlayers(pa_path)
layers

# %%
#filter by state
pa_shp = gpd.read_file(pa_path, layer = f"PADUS4_1Fee_State_{STATE}")
pa_shp

# %%
#look at crs
pa_shp.crs

# %%
#correct crs
pa_shp = pa_shp.to_crs(epsg = 4326)

# %%
#deal with multipolygons
pa_shp['geometry'] = pa_shp['geometry'].apply(
    lambda geom: geom.make_valid() if not isinstance(geom,
                                                     MultiPolygon) and not geom.is_valid else geom)

# %%
#deal with invalid geometries
pa_shp = pa_shp[pa_shp.geometry.is_valid]

# %%
#deal with na entries
pa_shp = pa_shp.dropna(subset=['geometry'])

# %% [markdown]
# > Uncomment to plot, takes a few mins

# %%
# # plot CA protected areas
# pa_shp.hvplot(
#     geo = True,
#     tiles = 'EsriImagery',
#     title = f'{STATE} Protected Areas',
#     fill_color = None,
#     line_color = 'white',
#     frame_width = 600
# )

# %%
#look at columns
pa_shp.columns

# %%
#subset species data by protected area borders
whitebark = gpd.overlay(gbif_gdf, pa_shp, how = "intersection")

# %%
#count occurrences per PA
val_counts = whitebark['Unit_Nm'].value_counts()

#check
val_counts

# %%
#look at filtered
whitebark

# %%
# get unique PA names
pa_occs_unique = whitebark['Unit_Nm'].unique()

# filter PAs by observation
pa_occs = pa_shp[pa_shp['Unit_Nm'].isin(pa_occs_unique)]

# %%
#look at columns
pa_occs.columns

# %%
#look at length
len(pa_occs)

# %%
#plot protected areas with occurrences
pa_plot = pa_occs.hvplot(
    geo=True,
    tiles = 'EsriImagery',
    color = None,
    alpha = .7,
    line_color = 'red',
    frame_width = 600,
    # hover_cols = ['Loc_Nm']
).opts(legend_position=None)

#plot occurrences within protected areas
occ_plot = whitebark.hvplot(
    geo = True,
    color = 'blue',
    size = 5,
    alpha = .75,
    hover_cols = ['dateIdentified', 'Unit_Nm']
)

#combine plots
overlay_plot = pa_plot * occ_plot
overlay_plot

# %% [markdown]
# Unit_Nm
# - Inyo National Forest                 295
# - Eldorado National Forest             136
# - Shasta National Forest               119
# - Toiyabe National Forest              107
# - Lassen Volcanic National Park        103
# 

# %%
#filter by specific PA
inyo_gdf = pa_shp[pa_shp['Unit_Nm'] == 'Inyo National Forest']
inyo_gdf

# %%
#plot
inyo_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'Inyo Protected Areas',
    fill_color = None,
    line_color = 'white',
    frame_width = 200
)

# %%
lavo_gdf = pa_shp[pa_shp['Unit_Nm'] == 'Lassen Volcanic National Park']
lavo_gdf

# %%
lavo_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'LAVO Protected Areas',
    fill_color = None,
    line_color = 'white')

# %%
#set bounds from testing
lavo_box = box(-121.6, 40.38, -120.00, 41.6)

#clip PA
lavo_clipped = gpd.clip(lavo_gdf, lavo_box)

#plot clipped
lavo_clipped.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'Clipped LAVO Protected Areas',
    fill_color = None,
    line_color = 'white')


# %%
#multiple filters needed for non-unique names
shasta_broad_gdf = pa_shp[pa_shp['Unit_Nm'] == 'Shasta National Forest']
shasta_gdf = shasta_broad_gdf[shasta_broad_gdf['Loc_Nm'] == "Shasta-Trinity National Forest"]
shasta_gdf

# %%
shasta_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'Shasta Protected Areas',
    fill_color = None,
    line_color = 'white')

# %%
shasta_box = box(-122.65, 41.20, -122.00, 41.6)

shasta_clipped = gpd.clip(shasta_gdf, shasta_box)

shasta_clipped.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'Clipped Shasta Protected Areas',
    fill_color = None,
    line_color = 'white')

# %% [markdown]
# Lassen Volcanic National Park (LAVO) frontrunner and Shasta and/or Inyo. Ideally clip Shasta by min elevation to isolate and focus on the mountain peak region.

# %%
# combine plots of interest
sites_gdf = gpd.GeoDataFrame(pd.concat([lavo_clipped, shasta_clipped], ignore_index=True))
sites_gdf

# %%
#plot
sites_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = f'Shasta & LAVO Protected Areas',
    fill_color = None,
    line_color = 'white')

# %% [markdown]
# ### Step 1c: Select time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, you will need to access climate data from 2035-2065.
# 
# **Reflect and Respond**: Select at least two 30-year time periods to compare, such as historical and 30 years into the future. These time periods should help you to answer your scientific question.

# %% [markdown]
# I am planning to use historical/present data over 1981-2010 and a future time period of 2070-2099.

# %% [markdown]
# ### Step 1d: Select climate models
# 
# There is a great deal of uncertainty among the many global climate
# models available. One way to work with the variety is by using an
# **ensemble** of models to try to capture that uncertainty. This also
# gives you an idea of the range of possible values you might expect! To
# be most efficient with your time and computing resources, you can use a
# subset of all the climate models available to you. However, for each
# scenario, you should attempt to include models that are:
# 
# -   Warm and wet
# -   Warm and dry
# -   Cold and wet
# -   Cold and dry
# 
# for each of your sites.
# 
# To figure out which climate models to use, you will need to access
# summary data near your sites for each of the climate models. You can do
# this using the [Climate Futures Toolbox Future Climate Scatter
# tool](https://climatetoolbox.org/tool/Future-Climate-Scatter). There is
# no need to write code to select your climate models, since this choice
# is something that requires your judgement and only needs to be done
# once.
# 
# If your question requires it, you can also choose to include multiple
# climate variables, such as temperature and precipitation, and/or
# multiple emissions scenarios, such as RCP4.5 and RCP8.5.
# 
# **Reflect and respond**: Choose at least 4 climate models that cover the range of possible future climate variability at your sites. Which models did you choose, and how did you make that decision?

# %% [markdown]
# HOT & WET
# - CanESM2
# 
# HOT & DRY
# - MIROC-ESM
# 
# COLD & DRY
# - bcc-csm1-1
# 
# HOT & DRY
# - GFDL-ESM2G
# 
# COLD MOIST (Optional)
# - MRI - CGCM3

# %% [markdown]
# ## STEP 2: Data access
# 
# ### Step 2a: Soil data
# 
# The [POLARIS dataset](http://hydrology.cee.duke.edu/POLARIS/) is a
# convenient way to uniformly access a variety of soil parameters such as
# pH and percent clay in the US. It is available for a range of depths (in
# cm) and split into 1x1 degree tiles.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download POLARIS data for a particular location, soil parameter,
# and soil depth. Your function should account for the situation where
# your site boundary crosses over multiple tiles, and merge the necessary
# data together.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include soil parameters that will help you to
# answer your scientific question. We recommend using a soil depth that
# best corresponds with the rooting depth of your species.</p></div></div>

# %% [markdown]
# Starting with theta_s, or "saturated volumetric water content”, which measures how much of the soil volume is filled with water when the soil is fully saturated. This is particularly relevant as we look at sparse, volcanic soils at higher altitudes and are interested in measuring water and snowpack retainability for the whitebark pine. Theta_s also responds to climate predictions as temperature, precipitation, and snowmelt contribute. High theta_s soils act like a sponge whereas low theta_s soils dry much more quickly. 
# 
# Starting with mean and 15-30 soil depth. The species' roots can grow deeper but I want to minimize risk of skewing the model with low soil depth and volcanic rock.

# %%
### Download and process soil data

# %% [markdown]
# > MODIFY SOIL PARAMETERS HERE

# %%
#set soil parameters
soil_var = "theta_s"
soil_stat = "mean"
soil_depth = "15_30"

# bounding box
xmin, ymin, xmax, ymax = lavo_clipped.total_bounds

# initialize
tiles = []

#loop through lat and lon in range of bounds
for lat_min in range(floor(ymin), ceil(ymax)):
    for lon_min in range(floor(xmin), ceil(xmax)):

        # calc max lat lon
        lat_max, lon_max = lat_min + 1, lon_min + 1

        # theta_s url 
        url = (
            "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
            f"{soil_var}/{soil_stat}/{soil_depth}/"
            f"lat{lat_min}{lat_max}_lon{lon_min}{lon_max}.tif"
        )

        # open raster and append to tile
        tiles.append(rxr.open_rasterio(url))

soil_da = rxrm.merge_arrays(tiles).rio.clip_box(*lavo_clipped.total_bounds)

soil_da.plot()

# %%
def get_soil_data_new(gdf, soil_var, soil_stat, soil_depth, buffer=0, area_name=None, force_download=False):
    """
    Download POLARIS soil data tiles for a specific region and then merge.

    Args:
        gdf: GDF of study area(s)
        soil_var: str, variable of interest
        soil_stat: str, "mean", "mode", "p5", "p50", "p95" 
        soil_depth: str, soil depth in cm, e.g. "0_5", "5_15", "15_30"
        buffer: float, optional, expand bounding box
        area_name: str, optional, unique area name
        force_download: boolean, re-run even if file exists 

    Returns:
        xarray.DataArray of soil data for study areas.
    """
    soil_cache = habitat_dir / "soil-data"
    soil_cache.mkdir(exist_ok=True)

    # create a unique filename based on inputs
    prefix = f"{area_name}_" if area_name else ""
    cache_file = soil_cache / f"{prefix}{soil_var}_{soil_stat}_{soil_depth}.tif"
        
    # load the file and exist function if exists already
    if cache_file.exists() and not force_download:
        print("Loading cached raster...")
        return rxr.open_rasterio(cache_file)

    print(f"Downloading and merging soil data")
          
    # initialize tiles
    tiles = []

    # make a single polygon for interior/edge selection before downloading
    study_union = gdf.union_all()
    xmin, ymin, xmax, ymax = study_union.bounds
    xmin, ymin, xmax, ymax = xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer
    
    for lat_min in range(floor(ymin), ceil(ymax)):
        for lon_min in range(floor(xmin), ceil(xmax)):

            # calc max lat lon
            lat_max, lon_max = lat_min + 1, lon_min + 1

            # theta_s url 
            url = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
                f"{soil_var}/{soil_stat}/{soil_depth}/"
                f"lat{lat_min}{lat_max}_lon{lon_min}{lon_max}.tif"
            )

            try:
                da = rxr.open_rasterio(url)
                tiles.append(da)
            except Exception as e:
                print(f"Warning: tile {lat_min},{lon_min} not found. Skipping. {e}")

    if not tiles:
        raise ValueError("No POLARIS tiles intersect study area")
    
    merged = rxrm.merge_arrays(tiles)
    soil_da = merged.where(merged > 0).rio.clip_box(*gdf.total_bounds)
    soil_da.rio.to_raster(cache_file)

    area_label = area_name.replace(" ", "_") if area_name else "site"
    soil_da.name = f"{area_label}_{soil_var}_{soil_stat}"
    soil_da.attrs['layer_type'] = 'soil'
    soil_da.attrs['soil_var'] = soil_var
    soil_da.attrs['soil_stat'] = soil_stat
    soil_da.attrs['soil_depth'] = soil_depth

    return soil_da

# %%
soil_var = "theta_s"
soil_stat = "mean"
soil_depth = "15_30"
a_name = "shasta"

shasta_soil = get_soil_data_new(shasta_clipped, soil_var, soil_stat, soil_depth, area_name=a_name)

# %%
shasta_soil_plot = shasta_soil.plot()
shasta_soil_plot

# %%
soil_var = "theta_s"
soil_stat = "mean"
soil_depth = "15_30"
a_name = "lavo"

lavo_soil = get_soil_data_new(lavo_clipped, soil_var, soil_stat, soil_depth, area_name=a_name)

# %%
lavo_soil_plot = lavo_soil.plot()
lavo_soil_plot

# %%
def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title,
              bar_label, plot_cmap, boundary_clr, tif_file = False):
    
    """
    Create site plot
    
    Args:
    site_da: xarray.DataArray, input site raster
    site_gdf: GDF, site boundary gdf
    plots_dir: str, path of plots for saving
    site_fig_name: str, site figure name
    plot_title: str, plot title
    bar_label: str, plot bar var name
    plot_cmap: str, colormap for plot
    boundary_clr: str, color for plot
    tif_file: boolean, indicate site file

    
    ReturnsL
    matplotlib.pyplot.plot: plot of site values
    """
    fig = plt.figure(figsize = (8,6))
    ax = plt.axes()

    #conditional
    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked=True)

    #plot
    site_plot = site_da.plot(cmap = plot_cmap,
                             cbar_kwargs = {'label': bar_label})
    
    site_gdf.boundary.plot(ax = plt.gca(), color = boundary_clr)

    plt.title(site_da.name)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


    fig.savefig(f"{plots_dir}/{site_fig_name}.png")

    return site_plot

# %%
plots_dir = habitat_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

# %%
lavo_plot = plot_site(site_da = lavo_soil, 
            site_gdf = lavo_clipped, 
            plots_dir = plots_dir,
            site_fig_name = "lavo_soil", 
            plot_title = "Lavo Soil",
            bar_label = "theta_s", 
            plot_cmap = "viridis", 
            boundary_clr = "black")

# %% [markdown]
# ### Step 2b: Topographic data
# 
# Depending on your species habitat needs/environmental parameters, you might be interested in elevation, slope, and/or aspect. You can access reliable elevation data from the [SRTM
# dataset](https://www.earthdata.nasa.gov/data/instruments/srtm),
# available through the [earthaccess
# API](https://earthaccess.readthedocs.io/en/latest/quick-start/). Once you have elevation data, you can calculate slope and aspect.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download SRTM elevation data for a particular location and
# calculate any additional topographic variables you need such as slope or
# aspect.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include topographic parameters that will help you
# to answer your scientific question.</p></div></div>
# 
# > **Warning**
# >
# > Be careful when computing the slope from elevation that the units of
# > elevation match the projection units (e.g. meters and meters, not
# > meters and degrees). You will need to project the SRTM data to
# > complete this calculation correctly.

# %%
### Download and process topographic data

# %%
earthaccess.login(persist=True)

# %%
#search for data
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %% [markdown]
# SRTMGL1 NASA Shuttle Radar Topography Mission Global 1 arc second V003 (~30m resolution)
# 
# SRTMGL3 NASA Shuttle Radar Topography Mission Global 3 arc second V003 (~90m resolution)
# 
# 

# %%
shasta_clipped

# %%
def get_topo_data(gdf, shortname, buffer=.005, force_download=False):

    # topo dir
    topo_dir = habitat_dir / "topography-data"
    topo_dir.mkdir(exist_ok=True)

    # file pattern
    unit_name = gdf['Unit_Nm'].iloc[0]
    study_area_name = unit_name.split(' ')[0]
    topo_raster_pattern = topo_dir / f"{study_area_name}_{shortname}_topo"
    topo_raster_pattern.mkdir(exist_ok=True)

    hgt_files = list(topo_raster_pattern.glob("*.hgt"))

    if hgt_files and not force_download:
        print("Loading cached topo raster")

    else:
        print("Downloading topo files")

        #bbox
        bounds = tuple(gdf.total_bounds)
        xmin, ymin, xmax, ymax = bounds
        topo_bounds = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

        # search
        srtm_search = earthaccess.search_data(
            short_name = shortname,
            bounding_box = topo_bounds
            )
        # download
        download_zips = earthaccess.download(
            srtm_search,
            topo_raster_pattern
            )

        # extract
        hgt_files = []
        for zf in download_zips:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(topo_raster_pattern)
            Path(zf).unlink()

        # now just glob for .hgt files in the folder
        hgt_files = list(topo_raster_pattern.glob("*.hgt"))

    da_list = []
    for hgt_file in hgt_files:
        tile_da = rxr.open_rasterio(hgt_file, mask_and_scale=True).squeeze()
        da_list.append(tile_da)
        
    merged_da = rxr.merge.merge_arrays(da_list)

    merged_da.name = f"{study_area_name}_{shortname}_topo"
    merged_da.attrs['layer_type'] = 'topo'
    merged_da.attrs['shortname'] = shortname

    return merged_da


# %%
shasta_topo = get_topo_data(shasta_clipped, "SRTMGL3")
lavo_topo = get_topo_data(lavo_clipped, "SRTMGL3")

# %%
lavo_topo.plot(cmap='terrain')
lavo_clipped.boundary.plot(ax=plt.gca(), color = 'black')

# %%
shasta_aspect = xrspatial.aspect(shasta_topo) % 360
lavo_aspect = xrspatial.aspect(lavo_topo) % 360

# %% [markdown]
# 0 - North
# 
# 90 - East
# 
# 180 - South
# 
# 270 - West

# %%
lavo_aspect.plot(cmap = 'terrain')

# %%
# reproject to projected crs for slope
shasta_rpj = shasta_topo.rio.reproject("EPSG: 5070")
lavo_rpj = lavo_topo.rio.reproject("EPSG: 5070")

# %%
shasta_rpj_slope = xrspatial.slope(shasta_rpj)
lavo_rpj_slope = xrspatial.slope(lavo_rpj)

#re-reproject
shasta_slope = shasta_rpj_slope.rio.reproject("EPSG: 4326")
lavo_slope = lavo_rpj_slope.rio.reproject("EPSG: 4326")


# %%
shasta_slope.name = "Shasta_slope"
shasta_slope.attrs['layer_type'] = "slope"

lavo_slope.name = "Lassen_slope"
lavo_slope.attrs['layer_type'] = "slope"

shasta_aspect.name = "Shasta_aspect"
shasta_aspect.attrs['layer_type'] = "aspect"

lavo_aspect.name = "Lassen_aspect"
lavo_aspect.attrs['layer_type'] = "aspect"

# %%
shasta_slope.plot(cmap = 'terrain')
shasta_clipped.boundary.plot(ax=plt.gca(), color = 'black')

# %%
lavo_slope.plot(cmap = 'terrain')
lavo_clipped.boundary.plot(ax=plt.gca(), color = 'black')

# %% [markdown]
# ### Step 2c: Climate model data
# 
# You can use MACAv2 data for historical and future climate data. Be sure
# to compare at least two 30-year time periods (e.g. historical vs. 10
# years in the future) for at least four of the CMIP models. Overall, you
# should be downloading at least 8 climate rasters for each of your sites,
# for a total of 16. **You will *need* to use loops and/or functions to do
# this cleanly!**.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download MACAv2 data for a particular climate model, emissions
# scenario, spatial domain, and time frame. Then, use loops to download
# and organize the 16+ rasters you will need to complete this section. The
# <a
# href="http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html">MACAv2
# dataset is accessible from their Thredds server</a>. Include an
# arrangement of sites, models, emissions scenarios, and time periods that
# will help you to answer your scientific question.</p></div></div>

# %% [markdown]
# HOT & WET
# - CanESM2
# 
# HOT & DRY
# - MIROC-ESM
# 
# COLD & DRY
# - bcc-csm1-1
# 
# HOT & DRY
# - GFDL-ESM2G
# 
# COLD MOIST (Optional)
# - MRI - CGCM3

# %%
def kel_to_cel(temperature):
    """
    Convert temp from Kelvin to Celsius
    
    Args:
    temp: temperature in Kelvin
    
    Returns:
    Temperature in Celsius
    """

    return temperature - 273.15

# %%
def convert_long(lon):
    """
    Convert [0,360] to [-180,180]
    
    Args:
    lon: longitude in [0,360]
    
    Returns:
    Longitude in [-180,180]
    """

    return xr.where(lon > 180, lon - 360, lon)

# %%
climate_dir = habitat_dir / "climate-data"
climate_dir.mkdir(exist_ok=True)

# %%
def process_maca(site_list,
                 years_list,
                 models_list,
                 vars_list,
                 output_list=None,
                 rcp_value="rcp45",
                 frequency = "monthly",
                 maca_data_dir=climate_dir):
    
    """
    Process MACA urls for study sites, models, and times
    
    Args:
    
    Returns:
    
    """
    
    var_lookup = {
    "pr": "precipitation",
    "tasmax": "air_temperature",
    "tasmin": "air_temperature"
    }   

    if output_list is None:
        output_list = []

    #loop over each site in site_list
    for site in tqdm(site_list, desc="Site", position=0):
       
        unit_name = site['Unit_Nm'].iloc[0]
        
        #loop over each period in years_list
        for date_range in tqdm(years_list, desc="Range", position=1):
            
            start_year = int(date_range.split("_")[0])
            scenario = "historical" if start_year < 2006 else rcp_value
            
            #loop over each model in models_list
            for model in tqdm(models_list, desc="Model", position=2):

                #loop over each climate variable
                for climate_var in vars_list:
                
                #define the MACA url
                    
                    
                    maca_url = (
                        "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
                        "/MACAV2"
                        f"/{model}"
                        f"/macav2metdata_{climate_var}"
                        f"_{model}_r1i1p1"
                        f"_{scenario}"
                        f"_{date_range}_CONUS"
                        f"_{frequency}.nc"
                        )

                    #set paths
                    maca_full_path = maca_data_dir / f"maca_{model}_{climate_var}_{scenario}_{date_range}_{frequency}.nc"

                    #download once
                    print(f"Trying: model={model}, var={climate_var}, scenario={scenario}, date_range={date_range}")
                    if not os.path.exists(maca_full_path):
                        # print("Downloading")

                        # open and squeeze data
                        maca_ds = xr.open_dataset(maca_url).squeeze()
                        maca_da = maca_ds[var_lookup[climate_var]]
                        
                        # convert to kelvin if temperature
                        if var_lookup[climate_var] == "air_temperature":                            
                            maca_da = kel_to_cel(maca_da)

                        maca_da.name = climate_var
                        maca_da.to_netcdf(maca_full_path)                    
                    else:
                        maca_ds = xr.open_dataset(maca_full_path).squeeze()
                        maca_da = maca_ds[climate_var]
                        

                    # change longitude values
                    maca_da = maca_da.assign_coords(
                                lon=convert_long(maca_da.lon))
                    
                    # define spatial bounds
                    site_maca_crs = site.to_crs("EPSG:4326")
                    bounds_maca = site_maca_crs.total_bounds

                    # set spatial dimensions
                    maca_da = maca_da.rio.set_spatial_dims(
                                x_dim = "lon",
                                y_dim = "lat"
                                )
                    
                    maca_da = maca_da.rio.write_crs("EPSG:4326")

                    # crop to site boundaries
                    maca_clipped = maca_da.rio.clip_box(*bounds_maca)
                    
                    # add cropped da to dict w/ metadata and save to list
                    result = dict(
                    site_name = unit_name,
                    climate_model = model,
                    climate_var = climate_var,
                    date_range = date_range,
                    da = maca_clipped,
                    scenario = scenario,
                    frequency = frequency
                                )
                    
                    output_list.append(result)

    # return list of dictionaries, each with cropped and processed climate data
    return output_list

# %%
def split_years(start_year, end_year, chunk_size=5):
    """
    Split a year range into chunks and return strings like in maca format 19XX_19YY
    """
    chunks = []
    current_start = start_year

    while current_start <= end_year:
        current_end = min(current_start + chunk_size - 1, end_year)
        chunks.append(f"{current_start}_{current_end}")
        current_start += chunk_size

    return chunks

# %%
early_years = split_years(2006,2025)
late_years = split_years(2061,2080)
years_list=early_years+late_years

# %%
site_list = [lavo_clipped, shasta_clipped]


models_list =[
    #HOT & WET
    "CanESM2",
    
    #HOT & DRY
    "MIROC-ESM",

    #COLD & DRY
    "bcc-csm1-1",

    #HOT & DRY
    "GFDL-ESM2G",
]

vars_list = ["pr", "tasmax", "tasmin"]

# %%
maca_output = process_maca(site_list=site_list,
                           years_list=years_list,
                           models_list=models_list,
                           vars_list=vars_list)

# %%
def generate_mean_climate_das(site_name, climate_var, maca_list, early_years, late_years):
    """
    Generate mean DataArrays for a single climate variable, split by site, time period, and model.

    Returns a nested dict: { period_name : { model_name : DataArray } }
    """
    periods = {"early": early_years, "late": late_years}
    da_list = []

    # get all unique models in the dataset
    models = set(entry['climate_model'] for entry in maca_list)

    for period_name, years_list in periods.items():
        
        for model in models:
            # select entries matching site, model, variable, and time period
            matching_entries = [
                entry for entry in maca_list
                if (entry['site_name'] == site_name and
                    entry['climate_model'] == model and
                    entry['climate_var'] == climate_var and
                    entry['date_range'] in years_list)
            ]

            if matching_entries:
                # combine along time and take mean
                combined = xr.concat([entry['da'] for entry in matching_entries], dim="time")
                
                # average along time
                mean_da = combined.mean(dim="time", keep_attrs=True)
                
                # assign lat/lon if they exist
                if 'lat' in combined.coords and 'lon' in combined.coords:
                    mean_da = mean_da.assign_coords(lat=combined['lat'], lon=combined['lon'])
                
                mean_da = mean_da.squeeze(drop=True)

                # give a unique, descriptive name
                mean_da.name = f"{site_name.replace(' ','_')}_{climate_var}_{model}_{period_name}"
                mean_da.attrs['var'] = climate_var
                mean_da.attrs['model'] = model
                mean_da.attrs['period'] = period_name

                da_list.append(mean_da)

    return da_list

# %%
def select_climate_da(da_list, period=None, model=None, var=None):
    return [
        da for da in da_list
        if (period is None or da.attrs.get('period') == period)
        and (model is None or da.attrs.get('model') == model)
        and (var is None or da.attrs.get('var') == var)
    ][0]  # returns the first match

# %%
early_years = split_years(2006,2025)
late_years = split_years(2061,2080)

lavo_tasmax = generate_mean_climate_das(
    site_name="Lassen Volcanic National Park",
    climate_var="tasmax",
    maca_list=maca_output,
    early_years=early_years,
    late_years=late_years
)

# Access one model
lavo_tasmax_early_bcc = select_climate_da(lavo_tasmax, period="early", model = "bcc-csm1-1")
lavo_tasmax_early_bcc


# %%
early_years = split_years(2006,2025)
late_years = split_years(2061,2080)

shasta_tasmax = generate_mean_climate_das(
    site_name="Shasta National Forest",
    climate_var="tasmax",
    maca_list=maca_output,
    early_years=early_years,
    late_years=late_years
)

shasta_tasmax_early_bcc = select_climate_da(shasta_tasmax, period="early", model = "bcc-csm1-1")
shasta_tasmax_early_bcc


# %%
shasta_tasmax_early_bcc.plot()
shasta_clipped.boundary.plot(ax=plt.gca(), color = 'black')

# %% [markdown]
# **Reflect and respond**: Make sure to include a description of the climate data and how you selected your models. Include a citation of the MACAv2 data.

# %% [markdown]
# Your response here:

# %%


# %% [markdown]
# ## STEP 3: Harmonize data
# To use all your environmental and climate data layers together, you need to harmonize the different rasters you've downloaded and processed. 
# 
# As a first step, make sure that the grids for all the rasters match each other. Check out the <a href="https://corteva.github.io/rioxarray/stable/examples/reproject_match.html#Reproject-Match"><code>ds.rio.reproject_match()</code> method</a> from <code>rioxarray</code>. Make sure to use the data source that has the highest resolution as a template!</p></div></div>
# 
# > **Warning**
# >
# > If you are reprojecting data (as you need to here), the order of
# > operations is important! Recall that reprojecting will typically tilt
# > your data, leaving narrow sections of the data at the edge blank.
# > However, to reproject efficiently it is best for the raster to be as
# > small as possible before performing the operation. We recommend the
# > following process:
# >
# >     1. Crop the data, leaving a buffer around the final boundary
# >     2. Reproject to match the template grid (this will also crop any leftovers off the image)

# %%
### Align the grids of the different data layers
lavo_da_list = [
    lavo_soil,
    lavo_topo,
    lavo_slope,
    lavo_aspect
] + lavo_tasmax

# %%
### Align the grids of the different data layers
shasta_da_list = [
    shasta_soil,
    shasta_topo,
    shasta_slope,
    shasta_aspect
] + shasta_tasmax

# %%
def reproj_bounds(da_list, bounds_gdf, buffer=.05):
    reproj_da_list = []
    
    bounds = tuple(bounds_gdf.total_bounds)
    (xmin, ymin, xmax, ymax) = bounds
    buffer_bounds = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

    for da in tqdm(da_list):

        orig_attrs = da.attrs.copy()
        
        cropped_da = da.rio.clip_box(*buffer_bounds)

        reproj_da = (cropped_da.rio.reproject_match(da_list[0]))

        reproj_da.attrs.update(orig_attrs)

        reproj_da_list.append(reproj_da)

    return reproj_da_list

# %%
lavo_reproj_das = reproj_bounds(lavo_da_list, lavo_clipped)
shasta_reproj_das = reproj_bounds(shasta_da_list, shasta_clipped)


# %%
#lavo_reproj_das

# %%
#plot function
def plot_reproj_axs(da_list, bounds_gdf,  fig_size_mod=5, max_cols=4):
    
    n_plots = len(da_list)
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                         figsize = (fig_size_mod*n_cols,fig_size_mod*n_rows))

    if len(da_list) == 1:
        axes = [axes]
    
    axes = np.array(axes).ravel()
    
    for ax, data in zip(axes, da_list):

        if data.ndim == 3:
            data=data.squeeze(drop=True)

        data.plot(ax=ax, cmap='viridis', add_colorbar=False,)

        bounds_gdf.plot(ax = ax, facecolor = 'none',
                        edgecolor = 'white', linewidth=1)
        
        ax.set_aspect("equal", adjustable='box')
        # ax.set_axis_off()
        
        if data.name:
            ax.set_title(data.name, fontsize=12)
            
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01)
    plt.show()

# %%
plot_reproj_axs(lavo_reproj_das, lavo_clipped)

# %%
plot_reproj_axs(shasta_reproj_das, shasta_clipped)

# %% [markdown]
# ## STEP 4: Develop a fuzzy logic model
# 
# A fuzzy logic model is one that is built on expert knowledge rather than
# training data. You may wish to use the
# [`scikit-fuzzy`](https://pythonhosted.org/scikit-fuzzy/) library, which
# includes many utilities for building this sort of model. In particular,
# it contains a number of **membership functions** which can convert your
# data into values from 0 to 1 using information such as, for example, the
# maximum, minimum, and optimal values for soil pH.
# 
# To train a fuzzy logic habitat suitability model:</p>
# <pre><code>1. Find the optimal values for your species for each variable you are using (e.g. soil pH, slope, and current annual precipitation). 
# 2. For each **digital number** in each raster, assign a **continuous** value from 0 to 1 for how close that grid square/pixel is to the optimum range (1 = optimal, 0 = incompatible). 
# 3. Combine your layers by multiplying them together. This will give you a single suitability number for each grid square.
# 4. Optionally, you may apply a suitability threshold to make the most suitable areas pop on your map.</code></pre></div></div>
# 
# > **Tip**
# >
# > If you use mathematical operators on a raster in Python, it will
# > automatically perform the operation for every number in the raster.
# > This type of operation is known as a **vectorized** function. **DO NOT
# > DO THIS WITH A LOOP!**. A vectorized function that operates on the
# > whole array at once will be much easier and faster.

# %%
### Create fuzzy logic model for habitat suitability
def run_fuzzy(harmonized_rasters, site_name, model, 
                  optimal_vals, tolerance_ranges):

    periods = ["early", "late"]
    results = {} 

    raster_dir = habitat_dir / "fuzzy_rasters"
    raster_dir.mkdir(exist_ok=True)

    for period in periods:
        suitability_layers = []
        output_file = raster_dir / f"fuzzy_{site_name}_{period}_{model}.tif"

        for raster in harmonized_rasters:    
            r = raster.squeeze()
            var = raster.attrs.get('layer_type')
            
            if var is None:
                name_lower = raster.name.lower() if raster.name else ""
                if "topo" in name_lower:
                    var = "topo"          
                elif "slope" in name_lower:
                    var = "slope"
                elif "aspect" in name_lower:
                    var = "aspect"
                elif model.lower() in name_lower and period.lower() in name_lower:
                    var = 'tasmax'
                elif "theta_s" in name_lower:
                        var = "theta_s"
                else:
                        continue

            opt = optimal_vals[var]
            tol = tolerance_ranges[var]

            diff = r - opt
            sq_diff = diff**2
            scaled_diff = sq_diff/(2*tol**2)
            negative_scaled = -scaled_diff
            suitability = np.exp(negative_scaled)

            suitability_layers.append(suitability)
        
        if not suitability_layers:
            raise ValueError(f"No rasters matched the filters for {site_name}, {period}, {model}")
        combined_suitability = suitability_layers[0]

        for layer in suitability_layers[1:]:
            combined_suitability *= layer

        combined_suitability.name = f"{site_name.replace(' ', '_')}_fuzzy_{period}_{model}"
        combined_suitability.attrs['layer_type'] = 'fuzzy_suitability'
        combined_suitability.attrs['site'] = site_name
        combined_suitability.attrs['period'] = period
        combined_suitability.attrs['model'] = model
        
        combined_suitability.rio.to_raster(output_file)

        results[period] = combined_suitability

    results['difference'] = results['late'] - results['early']
    results['difference'].name = f"{site_name.replace(' ', '_')}_fuzzy_change_{model}"
    results['difference'].attrs['layer_type'] = 'fuzzy_suitability_change'
    results['difference'].attrs['site'] = site_name
    results['difference'].attrs['model'] = model

    return results

# %%
# define optimal environmental parameters
wb_optimal_values = {
    'topo': 2500,
    'theta_s': .55,
    'slope': 30,
    'aspect': 180,
    'tasmax': 14,
}

# define tolerances
wb_tolerance_ranges = {
    'topo': 1200,
    'theta_s': .11,
    'slope': 15,
    'aspect': 60,
    'tasmax': 2,
}


# %%
#define parameters
harm_rasters = lavo_reproj_das
period = "early"
climate_model = "canesm2"
site_name = "lavo"
boundary_gdf = lavo_clipped

#call function
fuzzy_lavo_canesm2 = run_fuzzy(harmonized_rasters=harm_rasters, site_name=site_name,
                                    model=climate_model, optimal_vals=wb_optimal_values, tolerance_ranges=wb_tolerance_ranges)

# %%
harm_rasters = shasta_reproj_das
period = "early"
climate_model = "canesm2"
site_name = "shasta"
boundary_gdf = shasta_clipped

#call function
fuzzy_shasta_canesm2 = run_fuzzy(harmonized_rasters=harm_rasters, site_name=site_name,
                                    model=climate_model, optimal_vals=wb_optimal_values, tolerance_ranges=wb_tolerance_ranges)

# %% [markdown]
# ## Results
# > I would like to be able to run all models and time periods through run_fuzzy but for now I run it model by model.

# %%
### Create plots
def plot_fuzzy_results(fuzzy_dict, boundary_gdf, cmap='viridis', diff_cmap='coolwarm',
                           xlabel='Longitude', ylabel='Latitude', cbar_label_main='Suitability', 
                           cbar_label_diff='Suitability Change'):
    """
    Plot fuzzy results (early, late, difference) side by side.
    
    Args:
        fuzzy_dict: dict from run_fuzzy_NEW
        boundary_gdf: GeoDataFrame for overlay
        cmap: colormap for suitability
        diff_cmap: colormap for difference layer
    """

    keys = list(fuzzy_dict.keys())
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=(7*n, 4))

    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        da = fuzzy_dict[key]

        # choose colormap
        if key == "difference":
            cmap_use = diff_cmap
            cbar_label = cbar_label_diff
        else:
            cmap_use = cmap
            cbar_label = cbar_label_main

        #turn off autocolorbar
        im = da.plot(ax=ax, cmap=cmap_use, add_colorbar=False)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        
        # overlay boundary
        boundary_gdf.boundary.plot(ax=ax, color='white', linewidth=1)

        # titles + axis labels
        ax.set_title(key.capitalize())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

# %%
#plot lavo
plot_fuzzy_results(fuzzy_lavo_canesm2, lavo_clipped)


# %%
# plot shasta
plot_fuzzy_results(fuzzy_shasta_canesm2, shasta_clipped)


# %% [markdown]
# Interpret your plots here:


