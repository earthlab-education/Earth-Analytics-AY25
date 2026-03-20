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
# <u>Scientific Question</u>
# Will the habitat suitability change for Douglas Fir in the late century climate change scenario 2070-2099? And is there a difference in these changes between sites with varying habitat charecteristics?
# 
# Douglas Fir (Pseudotsuga menziesii) is the second tallest tree species in the world and is very important for both human use and many ecological functions (NPS, 2017). The Douglas Fir is a member of the Pine Family (Pinaceae). It is important to distinguish for this analysis which variety of Douglas Fir we are examining. The study sites selected are west of the Cascades so we assume the variety identified to be Coastal Douglas Fir (P. menziesii var. menziesii) vs the Rocky Mountain Douglas Fir (P. menziesii var. glauca) which is found to the east of the Cascades. Their location is in part how they are distinguished and do differ in their habitat needs including soil and precipitation based on these geographies (USFS, 1990). In this analysis , we will use habitat characteristics for Coastal Douglas Fir.
# 
# Coastal Douglas Fir thrives in well-drained, moist, and deep loamy soils and prefers the soil to be slightly acidic (pH 5 to 6) to achieve its best growth ability (USFS, 1990). While it does prefer moist conditions, it does not tolerate saturated soils (USDA NRCS, 2002). It prefers loamy (well-balanced) soils but can tolerate gravelly soils and limited clay as long as soils are not compacted. It occurs on soils derived from a variety of parent materials including glacial and volcanic origins (USFS, 1990).
# 
# Coastal Douglas Fir is a dominant species in temperate conifer forests of the Pacific Northwest and commonly occurs in mixed forests with species such as western hemlock; silver, noble, and grand firs; and western cedar (NPS, 2017).
# 
# The geographic range of Coastal Douglas Fir extends along the Pacific coast of North America from central British Columbia through Washington and Oregon and south into northern and central California. It is most abundant west of the Cascade Range where maritime climates provide adequate precipitation and moderate temperatures (NPS, 2017).
# 
# While Coastal Douglas Fir remains widespread and ecologically dominant, climate change may alter its distribution through increased drought stress, changing precipitation patterns, and increased wildfire frequency (Klamerus-Iwan et al., 2025). Land use change and historical logging practices have also altered forest structure across parts of its range (USFS, 1990).
# 
# To model habitat suitability for Coastal Douglas Fir, several environmental variables were selected that strongly influence its growth and distribution. These include elevation and topographic variables (slope and aspect), which influence drainage and sun exposure; climate variables such as precipitation and temperature; and soil characteristics such as soil pH that affect nutrient availability (Klamerus-Iwan et al., 2025). These variables help determine whether a landscape provides suitable habitat conditions for Douglas Fir.
# 
# 
# 
# <u>Habitat Characteristics Relevant to Habitat Suitability Modeling</u>
# 
# Key environmental variables that influence Coastal Douglas Fir habitat suitability include:
# 
# - Elevation: Commonly found from sea level up to approximately 1,500 m in the Pacific Northwest
# 
# - Temperature: Prefers mild maritime climates with moderate annual temperatures and limited extreme cold. However, Douglas Fir is more heat intolerant, experiencing physiological stress above 80°F but it can tolerate down to 10°F. Douglas Fir's optimal temperature is 68°F.
# 
# - Precipitation: Thrives in regions receiving approximately 600–2500 mm of annual precipitation, with much of the moisture occurring during winter months
# 
# - Soil pH: Optimal growth occurs in slightly acidic soils, generally between pH 5.0–6.0
# 
# - Slope: Frequently found on moderate slopes where drainage is good
# 
# - Aspect: Prefers North & Northwest facing slopes in its more southern region (warmer, drier) and South & Southwest facing slopes in northern regions (cooler, wetter)

# %% [markdown]
# <u>Data Citations</u>
# 
# - Chaney, N. W., Wood, E. F., McBratney, A. B., Hempel, J. W., Nauman, T. W., Brungard, C. W., & Odgers, N. P. (2016). POLARIS: A 30-meter probabilistic soil series map of the contiguous United States. Geoderma, 274, 54–67. https://doi.org/10.1016/j.geoderma.2016.03.025
# 
# - NASA Jet Propulsion Laboratory (JPL). (2013). NASA Shuttle Radar Topography Mission Global 1 arc second [Data set]. NASA Land Processes Distributed Active Archive Center. https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1.003
# 
# - Global Biodiversity Information Facility (GBIF). (2024). GBIF occurrence download. https://www.gbif.org
# 
# - U.S. Forest Service. (2020). Administrative forest boundaries dataset. U.S. Department of Agriculture.
# 
# - Abatzoglou, J. T., & Brown, T. J. (2012). A comparison of statistical downscaling methods suited for wildfire applications. International Journal of Climatology, 32(5), 772–780. https://doi.org/10.1002/joc.2313

# %%
### Import libraries
# File paths
import os
import pathlib 
from glob import glob
from pathlib import Path

# Work with GBIF data
import pygbif.occurrences as occ
import pygbif.species as species
from getpass import getpass

# Unzipping
import zipfile 
import time

# Work with spatial data
import geopandas as gpd 
import xrspatial
import xarray as xr

# Work with dataframes
import pandas as pd
import numpy as np

# Plot and visualize
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import panel as pn
import cartopy.crs as ccrs
import rioxarray as rxr
import rioxarray.merge as rxrm
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt
from rasterio.crs import CRS

# Access NASA earth data (topography)
import earthaccess

# Soil calculations
from math import floor, ceil

# Deal with invalid geometries
from shapely.geometry import MultiPolygon, Polygon

# for API use
import requests

# Search for locations by name - this might take a moment
from osmnx import features as osm

# Progress bars
from tqdm.auto import tqdm
from tqdm.notebook import tqdm 

# %%
### Set up file paths
# Set up directory
data_dir = os.path.join(

    pathlib.Path.home(),

    # Earth Analytics
    'Documents',
    'education',
    'earth-data-analytics',
    'spring-2026-data',

    # Create project directory
    'douglas-fir-habitat-suitability'
)

# Make the dir once
os.makedirs(data_dir, exist_ok=True)

# %%
# Set a dir for the gbif data
gbif_dir = os.path.join(data_dir, 'gbif_douglas_fir')

# %%
### Login to GBIF account and do not save credentials
# Reset credentials
reset_credentials = False

# Request and store username
if (not ('GBIF_USER'  in os.environ)) or reset_credentials:
    os.environ['GBIF_USER'] = input('GBIF username:')

# Securely request and store password
if (not ('GBIF_PWD'  in os.environ)) or reset_credentials:
    os.environ['GBIF_PWD'] = getpass('GBIF password:')
    
# Request and store account email address
if (not ('GBIF_EMAIL'  in os.environ)) or reset_credentials:
    os.environ['GBIF_EMAIL'] = input('GBIF email:')

# %%
# Check that it is logged in
'GBIF_PWD' in os.environ

# %%
# Set species name
species_name = "Pseudotsuga menziesii var. menziesii"

# Species info from GBIF
species_info = species.name_lookup(species_name,
                                   rank = 'Species')

# Grab the first result
first_result = species_info['results'][0]
first_result

# %%
# Get the species key
species_key = first_result['nubKey']

# Check that
first_result['species'], species_key

# Assign the species code
species_key = 2685796

# %%
# Make the file path
gbif_pattern = os.path.join(gbif_dir, '*.csv')

# Download it once
if not glob(gbif_pattern):

    # Submit the query
    gbif_query = occ.download([
        f"speciesKey = {species_key}",
        "hasCoordinate = True",
    ])

    # Only download once
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
        download_key = os.environ['GBIF_DOWNLOAD_KEY']
        time.sleep(5)

    # Download the data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'],
        path = data_dir
    )

    # Unzip the file
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path = gbif_dir)
# Find csv file path
gbif_path = glob(gbif_pattern)[0]

# %%
# Check out the info
occ.download_meta("0000077-260221153910048")

# %%
# Read csv and look at dataframe
gbif_df = pd.read_csv(
    gbif_path,
    delimiter = '\t'
)

# Check it out
gbif_df

# %%
# Look at columns
gbif_df.columns


# %%
# Make it spatial
gbif_gdf = (
    gpd.GeoDataFrame(
        gbif_df,
        geometry = gpd.points_from_xy(
            gbif_df.decimalLongitude,
            gbif_df.decimalLatitude
        ),
        crs = 'EPSG:4326'
    )
)

# Check it out
gbif_gdf

# %%
# Plot it
gbif_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Douglas Fir occurrencces in GBIF',
    fill_color = None,
    line_color = 'purple',
    frame_width = 600
)

# %% [markdown]
# ### Step 1b: Select study sites
# Based on your research and/or range maps you find online, select at least 2 sites where your species occurs. These could be national parks, national forests, national grasslands or other protected areas, or some other area you're interested in. You can access protected area polygons from the [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview), [national grassland units](https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip), etc.
# 
# When selecting your sites, you might want to look for places that are marginally habitable for this species, since those locations will be most likely to show changes due to climate.
# 
# Generate a site map for each location.

# %% [markdown]
# ### Load Sites

# %%
# Make directory for sites
site_dir = Path(data_dir) / "douglas_fir_sites"
site_dir.mkdir(parents=True, exist_ok=True)

# %%
# Create python path for zip file to live
zip_path = site_dir / "BdyAdm_LSRS_AdministrativeForest.zip"
print(zip_path)


# %%
# Unzip
# Create folder for the data
extract_folder = zip_path.parent

# Create the folder if it doesn't exist
extract_folder.mkdir(parents=True, exist_ok=True)

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# %%
# Open the shapefile from forest boundaries
shp_path = extract_folder / "S_USA.BdyAdm_LSRS_AdministrativeForest.shp"

# Read the shapefile
gdf = gpd.read_file(shp_path)

# Check it out
print(gdf.head())
print(gdf.columns)

# %%
# Filter for Willamette and Olympic
willamette_gdf = gdf[gdf['FORESTNAME'] == 'Willamette National Forest']
olympic_gdf = gdf[gdf['FORESTNAME'] == 'Olympic National Forest']

# %%
# Check that they are geodataframes
type(willamette_gdf)
type(olympic_gdf)

# %%
# Check out the forest
print(willamette_gdf)
print(olympic_gdf)

# %%
# Check the crs
willamette_gdf.crs
olympic_gdf.crs

# %%
# Align crs to GBIF
willamette_gdf = willamette_gdf.to_crs(epsg = 4326)
olympic_gdf = olympic_gdf.to_crs(epsg = 4326)

# %%
# Check the polygon for Willamette
willamette_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Willamette National Forest',
    fill_color = None,
    line_color = "white",
    frame_width = 300,
    frame_height = 500
)

# %%
# Check the polygon for Olympic
olympic_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Olympic National Forest',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %%
# Intersect with GBIF data with Willamette
douglas_willamette = gpd.overlay(gbif_gdf, willamette_gdf, how = 'intersection')

# Check out the occerences
douglas_willamette

# %%
# Intersect with GBIF data with Olympic
douglas_olympic = gpd.overlay(gbif_gdf, olympic_gdf, how = 'intersection')

# Check it out
douglas_olympic

# %%
# Check the occurences for Olympic
douglas_willamette.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Douglas Fir Occurances - Willamette National Forest',
    fill_color = None,
    line_color = "white",
    frame_width = 400,
    frame_height = 500
)

# %%
# Check the occurences for Olympic
douglas_olympic.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Douglas Fir Occurances - Olympic National Forest',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %% [markdown]
# **Reflect and Respond**: 
# Write a site description for each of your sites, or for all of your sites as a group if you have chosen a large number of linked sites. What
# differences or trends in habitat suitability over time do you expect to see among your sites?

# %% [markdown]
# **Willamette National Forest**
# 
# Willamette National Forest is located on the western slopes of the Cascade Mountains in western Oregon. This national forest has over 1 million acres and has steep terrain, volcanic soils, and dense conifer forests. The region experiences a maritime-influenced climate with wet winters, dry summers, and significant annual precipitation that supports the highly productive forests dominated by species such as Douglas Fir (National Forest Federation, 2026). These factors make Douglas Fir a dominant tree species in the region and is the official state tree of Oregon.
# 
# **Olympic National Forest**
# 
# Olympic National Forest is located on the Olympic Peninsula in northwestern Washington and encompasses around 600,000 acres total (National Forest Federation, 2026). The forest is a part of a temperate rainforest ecosystem, which is quite rare globally. The region receives very high precipitation due to Pacific storm systems and the unique topography of the Olympic Mountains. This creates moist forest conditions that support large conifer species including Douglas Fir (NPS, 2025). 

# %% [markdown]
# ### Step 1c: Select time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, you will need to access climate data from 2035-2065.
# 
# **Reflect and Respond**: Select at least two 30-year time periods to compare, such as historical and 30 years into the future. These time periods should help you to answer your scientific question.
# 
# The time periods 1970-2000 and 2070-2099 were selected to compare changes over time in the habitat suitability for Douglas Fir and observe potential future scenarios under different climate projections. 

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
# 
# I chose the climate models CanESM2 (warm & wet), HadGEM2-CC365(warm & dry), inmcm4(cold & dry), and IPSL-CM5A-LR(cold & wet). I chose these four models to account for uncertainty in climate model predictions. I used the Climate Futures Toolbox Future Climate Scatter Tool and selected the best fit models for these for scenarios to get a picture of what habitat suitability might look like under potential late century conditions. 

# %% [markdown]
# Future Climate Predictions Model data was taken from the Climate Futures Toolbox using the Climate Scatter Tool. Retrieved from: [https://climatetoolbox.org/tool/future-climate-scatter](https://climatetoolbox.org/tool/future-climate-scatter)

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

# %%
### Download and process soil data
# Make a bounding box for Willamette National Forest

xmin, ymin, xmax, ymax = willamette_gdf.total_bounds

# Initialize tiles to accumulate into
tiles_willamette = [] 

# Loop 
for lat_min in range(floor(ymin), ceil(ymax)):
    for lon_min in range(floor(xmin), ceil(xmax)):

        # Calculate max lat and long for tile
        lat_max, lon_max = lat_min + 1, lon_min + 1 

        # Url for pH data
        ph_url = (
            "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
            "/ph/mean/15_30"
            f"/lat{lat_min}{lat_max}_lon{lon_min}{lon_max}.tif")
        
        # Open raster and append to tiles list
        tiles_willamette.append(rxr.open_rasterio(ph_url))

# Merge all the individual ph tiles
willamette_ph_da = merge_arrays(tiles_willamette).rio.clip_box(*willamette_gdf.total_bounds)

willamette_ph_da.plot()

# %%
# Flatten the values 
ph_vals = willamette_ph_da.values.flatten()

# Histogram
plt.hist(ph_vals, bins = 50, edgecolor = "black")

plt.show()

# %%
print("count of very negative values:", np.sum(ph_vals < 0))

# %%
print(willamette_ph_da.rio.nodata)

# %%
print("count of very negative values:", np.sum(ph_vals == -9999))

# %%
# Deal with weird values
willamette_ph_da = willamette_ph_da.where(willamette_ph_da != willamette_ph_da.rio.nodata)

# Check the range of values now
float(willamette_ph_da.min()), float(willamette_ph_da.max())

# %%
# Look at soil plot with no data values removed
willamette_ph_da.plot()

# %% [markdown]
# ### Write function for soil data

# %%
### Write a function to get the URLs 
def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):
    """
    A function to download the POLARIS urls for site boundary
    to generate dataset for soil parameters
    
    Args:
    soil_prop (str): soil property that we want (pH, soil conductivity, etc)
    stat (str): summary statistic (mean, p5, etc)
    soil_depth (str): soil depth in cm  (15 to 30 cm, 30 - 60 cm, etc)
    gdf_bounds: array of site boundaries

    Returns:
    list: a list of POLARIS urls
    """

    # extract bounding box for the site 
    min_lon, min_lat, max_lon, max_lat = gdf_bounds

    # snap boundaries to whole degrees (floor - rounding to nearest whole number)
    site_min_lon = floor(min_lon)
    site_min_lat = floor(min_lat)
    site_max_lon = ceil(max_lon)
    site_max_lat = ceil(max_lat)

    # Check
    print(site_min_lon, site_max_lon)
    print(site_min_lat, site_max_lat)

    # Initialize outpout list
    all_soils_urls = []

    # Loop through lat and lon to get each tile in study area
    for lon in range(site_min_lon, site_max_lon):
        for lat in range(site_min_lat, site_max_lat):

            # Calculate max lat and lon for all urls
            current_max_lon = lon + 1
            current_max_lat = lat + 1

            # Define the url for Ph data
            soil_template = (
            "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
            "/{soil_prop}/"
            "{stat}/"
            "{soil_depth}/"
            "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"
            )

            # Fill in the template with the parameters for 1 complete URL
            soil_url = soil_template.format(
                soil_prop = soil_prop,
                stat = stat, 
                soil_depth = soil_depth,
                min_lat = lat, 
                max_lat = current_max_lat,
                min_lon = lon, 
                max_lon = current_max_lon
            )

            # Append url to the list
            all_soils_urls.append(soil_url)

    # Return
    return all_soils_urls

# %%
# Try out the function
soil_urls_willamette = create_polaris_urls(soil_prop = "ph",
                                            stat = "mean",
                                            soil_depth = "15_30" ,
                                            gdf_bounds = willamette_gdf.total_bounds)

# %%
# Check it out
soil_urls_willamette

# %%
# Create function to open the raster tiles , mask and scale them, clip them to the site, and merge them
def build_da(urls, bounds):
    """
    Build a data array from list of urls
    
    Args:
    urls (list): list
    bounds (tuple): site boundaries

    return 
    xarray.Dataarray: merged Dataarray
    """

    # Initialize an empty list
    all_das = []

    # Add buffer
    buffer = 0.025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    # Porcess 1 url tile at a time
    for url in urls:

        # Open raster, mask missing data, remove any extra dimensions
        tile_da = rxr.open_rasterio(url,
                                    mask_and_scale=True).squeeze()
        
        # Unpack the bounds and crop tile to buffered boundaries
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)

        # Store cropped tile
        all_das.append(cropped_da)
        
    # Combine into single raster
    merged = merge_arrays(all_das)

    # Return the final raster
    return merged  

# %%
# Use build dataarray function for willamette
soil_da_willamette = build_da(urls = soil_urls_willamette,
                              bounds = willamette_gdf.total_bounds)

# %%
# Check it out
soil_da_willamette.plot()

# %%
# Create a function to save a zarray.DataArray as a raster
def export_raster(da, raster_path, data_dir):

    """
    Export raster to file
    
    Args:
    raster (xarray.DataArray): input raster layer
    raster_path (str): output raster directory
    data_dir (str): path of data directory
    
    Return: None (save to disk)
    
    """
     # Make sure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Remove any existing _FillValue attribute to avoid conflicts
    # Developed with help of ChatGPT & adapted by author
    if "_FillValue" in da.attrs:
        da.attrs.pop("_FillValue")

    # Clear any encoding that might conflict
    # Developed with help of ChatGPT & adapted by author
    da.encoding = {}

    # Build full output path
    output_file = os.path.join(data_dir, os.path.basename(raster_path))

    # Export raster
    da.rio.to_raster(output_file)
    
    print(f"Raster saved to {output_file}")

# %%
# Create a directory for soil rasters for willamette
raster_path = os.path.join(site_dir, "willamette", "soil_ph.tif")

# Export raster
export_raster(da = soil_da_willamette, 
              raster_path = raster_path, 
              data_dir = site_dir)

# %%
### Function for customizable plots
def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title,
              bar_label, plot_cmap, boundary_clr, tif_file = False):
    
    """
    Create custom plot
    
    Args:
    site_da (xarray.AtaArray): input site raster
    site_gdf (geopandas.GeoDataFrame): site boundary gdf
    plots_dir (str): path of plots directory for saving plots
    site_fig_name (str): site figure name
    plot_title (str): plot title
    bar_label (str): plot bar variable name
    plot_cmap (str): colormap for the plot
    boundary_clr (str): color for site boundary
    tif_file (bool): indicate site file

    Returns:
    matplotlib.pyplot.plot: a plot of site values
    """

    # Set up the figure
    fig = plt.figure(figsize = (8, 6))
    ax = plt.axes()

    # Conditional
    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked = True)

    # Plot dataarray values
    site_plot = site_da.plot(cmap = plot_cmap,
                             cbar_kwargs = {'label': bar_label})
    
    # Plot site boundary 
    site_gdf.boundary.plot(ax = plt.gca(), color = boundary_clr)

    # Add title and labels
    plt.title(f'{plot_title}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Save the figure
    fig.savefig(f"{plots_dir}/{site_fig_name}.png")

    return site_plot

# %%
# Create the directory / folder
ph_plots_dir = os.path.join(site_dir, "soil", "willamette", "plots")
os.makedirs(ph_plots_dir, exist_ok=True)

# %%
willamette_soil_plot = plot_site(site_da = soil_da_willamette, 
                                site_gdf = willamette_gdf, 
                                plots_dir = ph_plots_dir, 
                                site_fig_name = "willamette_soil_plot", 
                                plot_title = "Soil pH Mean (depth 15-30cm)- Willamette National Forest",
                                bar_label = "pH", 
                                plot_cmap = "viridis", 
                                boundary_clr = "white", 
                                tif_file = False)

willamette_soil_plot

# %%
# Make overall wrapper function
def download_polaris(site_name, site_gdf, soil_prop, stat, soil_depth,
                     plot_path, plot_title, data_dir, plots_dir):
    
    """
    Retrieve POLARIS data, build DataArray, plot site, and export raster
    
    Args: 
    site_name(str): name of site, used to name exported raster file
    site_gdf (geopandas.GeoDataFrame): boundary of site, used for bounding box 
    soil_prop (str): soil property of interest
    stat (str): summary statistic for POLARIS data
    soil_depth (str): soil depth in cm
    plot_path (str): string used to build plot filename
    plot_title (str): for title of plot
    data_dir (str): path of the directory where raster will be saved
    plots_dir (str): path for plots directory where png will be saved

    Returns:
    xarray.DataArray: soil DataArray for given location
    """

    # Collect the soil URLS
    site_polaris_da = create_polaris_urls(soil_prop, stat, soil_depth, site_gdf.total_bounds)

    # Download rasters, gather into single file
    site_soil_da = build_da(site_polaris_da, tuple(site_gdf.total_bounds))

    # Export as a raster
    export_raster(site_soil_da, f"{site_name}_soil_{soil_prop}.tif", data_dir)

    # Plot site
    plot_site(site_soil_da, site_gdf, plots_dir,
              f'{plot_path}-Soil', f'{plot_title} Soil',
              soil_prop, 'viridis', 'white')
    
    # Return the soil raster
    return site_soil_da

# %%
# Create the directory / folder
ph_raster_dir = os.path.join(site_dir, "soil", "willamette", "rasters")
os.makedirs(ph_plots_dir, exist_ok=True)

# %%
# try out the wrapper function
willamette_soil_da = download_polaris(site_name = "willamette",
                                      site_gdf = willamette_gdf,
                                      soil_prop = "ph",
                                      stat = "mean",
                                      soil_depth = "15_30",
                                      plot_path = "ph_15_30_willamette",
                                      plot_title = "ph_15_30_cm_willamette",
                                      data_dir = ph_raster_dir,
                                      plots_dir = ph_plots_dir)

# %%
# Repeat & use wrapper function for olympic national forest
olympic_soil_da = download_polaris(site_name = "olympic",
                                      site_gdf = olympic_gdf,
                                      soil_prop = "ph",
                                      stat = "mean",
                                      soil_depth = "15_30",
                                      plot_path = "ph_15_30_olympic",
                                      plot_title = "ph_15_30_cm_olympic",
                                      data_dir = ph_raster_dir,
                                      plots_dir = ph_plots_dir)

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
# Create a data directory for the topographic data
elevation_dir = os.path.join(data_dir, "topography")
os.makedirs(elevation_dir, exist_ok=True)

# Make a subdirectory for the willamette data
willamette_topo_dir = os.path.join(elevation_dir, "willamettte")
os.makedirs(willamette_topo_dir, exist_ok=True)

# Make a subdirectory for the olympic data
olympic_topo_dir = os.path.join(elevation_dir, "olympic")
os.makedirs(olympic_topo_dir, exist_ok=True)

# %%
# Set up Earth Access
earthaccess.login()

# %%
# Search for SRTM data
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
# File pattern for data
willamette_srtm_pattern = os.path.join(willamette_topo_dir, '*.hgt.zip')

# Study area for topo data
willamette_elev_bounds = tuple(willamette_gdf.total_bounds)

# Add buffer
buffer = 0.025
willamette_xmin, willamette_ymin, willamette_xmax, willamette_ymax = willamette_elev_bounds
willamette_elev_bounds_buffer = (willamette_xmin - buffer,
                                 willamette_ymin - buffer,
                                 willamette_xmax + buffer,
                                 willamette_ymax + buffer)

# Look at the results
srtm_files = glob(willamette_srtm_pattern)

if not srtm_files:

    # Search for data
    willamette_srtm_search = earthaccess.search_data(
        short_name = 'SRTMGL3',
        bounding_box = willamette_elev_bounds_buffer
    )

    # Download data
    willamette_srtm_results = earthaccess.download(
        willamette_srtm_search,
        willamette_topo_dir
    )

# Add text if files already downloaded
else:
    print("SRTM files already downloaded")
    willamette_srtm_results = srtm_files


# %%
# Check it out
willamette_srtm_results

# %%
willamette_srtm_da_list = []
for srtm_path in glob(willamette_srtm_pattern):
    tile_da = rxr.open_rasterio(srtm_path, mask_and_scale = True).squeeze()
    srtm_cropped_da = tile_da.rio.clip_box(*willamette_elev_bounds_buffer)
    willamette_srtm_da_list.append(srtm_cropped_da)

# Merge 
willamette_srtm_da = merge_arrays(willamette_srtm_da_list)

willamette_srtm_da.plot(cmap='terrain', cbar_kwargs={'label': 'Elevation (meters)'})

willamette_gdf.boundary.plot(ax = plt.gca(), color='black')

# Add title
plt.title("topo_elevation_willamette")

# Check it out
plt.show()

# %%
# Check the crs
print(willamette_srtm_da.rio.crs)

# %%
# Get aspect layer from elevation layer
willamette_aspect = xrspatial.aspect(willamette_srtm_da)

# Convert negative values to 0–360 degrees
willamette_aspect = (willamette_aspect + 360) % 360

# Plot it
willamette_aspect.plot(cmap = 'terrain')

willamette_gdf.plot(ax = plt.gca(), facecolor = 'none', edgecolor = 'black')
plt.show()

# %%
# Reproject DEM to EPSG:5070 (NAD83 / Conus Albers) so horizontal units are in meters - for accurate slope
willamette_rpj = willamette_srtm_da.rio.reproject("EPSG:5070")

# Calculate slope
willamette_slope = xrspatial.slope(willamette_rpj)

# Reproject slope back to lat/long for plotting
willamette_slope_4326 = willamette_slope.rio.reproject("EPSG:4326")

# Plot
willamette_slope_4326.plot(cmap='terrain')

willamette_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='black')

plt.show()

# %% [markdown]
# ### Write a function for topographic data

# %%
# Write a function to get the SRTM files
def download_srtm_files(site_gdf, topo_dir, buffer=0.025):
    """
    A function to download the SRTM tiles for site boundary
    to generate dataset for topographic data.
    Check whether SRTM files already exist in directory.
    If not, it searches Earthaccess for SRTM tiles covering the site boundary
    with a buffer and downloads them
    
    Args:
    site_gdf (geopandas.GeoDataFrame): GeodataFrame containing the study site boundary
    topo_dir (str): directory where SRTM files are stored
    buffer (float): buffer for bounding box (in degrees) to ensure full site coverage.

    Returns:
        list: list of downloaded SRTM file paths
    """

    # Extract bounding box for the site 
    elev_bounds = tuple(site_gdf.total_bounds)

    # Add buffer
    xmin, ymin, xmax, ymax = elev_bounds
    elev_bounds_buffer = (xmin - buffer,
                          ymin - buffer,
                          xmax + buffer,
                          ymax + buffer)
    
    # Ensure directory exists
    os.makedirs(topo_dir, exist_ok=True)

    # Create file path
    srtm_pattern = os.path.join(topo_dir, '*.hgt.zip')
    
    # Find existing files
    srtm_files = glob(srtm_pattern)

    # If no SRTM files exist, search and download them
    if not srtm_files:

        # Search for data
        srtm_search = earthaccess.search_data(short_name = 'SRTMGL3',
                                              bounding_box = elev_bounds_buffer
        )

        # Download data
        srtm_results = earthaccess.download(srtm_search,
                                            topo_dir
        )

    # Add text if files already downloaded
    else:
        print("SRTM files already downloaded")
        srtm_results = srtm_files

    # Return
    return srtm_results

# %%
# Check if function works
willamette_srtm_results = download_srtm_files(
    site_gdf = willamette_gdf, 
    topo_dir = willamette_topo_dir, 
    buffer=0.025
)

# Look at results
willamette_srtm_results

# %% [markdown]
# ##### Merge and Clip Function

# %%
# Create a function to open, merge, and clip files
def open_merge_srtm(srtm_files, site_gdf, buffer=0.025):
        """
        A function to open, merge, and crop the topographic elevation data 
        from the SRTM files into one file. 

        This function opens each tile, then clips it to the study area bounds,
        and merges all clipped tiles into one elevation raster.

        Args:
        srtm_files (list): list of SRTM tile file paths downloaded from Earthaccess
        site_gdf (geopands.GeoDataFrame): study area boundary
        buffer (float): bounding box for buffer in degrees

        Returns:
        (xarray.DataArray): merged Digital Elevation Model (DEM) for study area
        """
        # Create buffered bounds from site boundary
        xmin, ymin, xmax, ymax = site_gdf.total_bounds
        elev_bounds_buffer = (xmin - buffer,
                              ymin - buffer,
                              xmax + buffer,
                              ymax + buffer
                              )
        
        # Create empty list to store clipped tiles
        srtm_da_list = []

        # Loop through each SRTM file path
        for srtm_path in srtm_files:

            # Open the raster tile as a DataArray
            tile_da = rxr.open_rasterio(srtm_path, mask_and_scale = True).squeeze()

            # Clip the tile to bounds with buffer
            srtm_cropped_da = tile_da.rio.clip_box(*elev_bounds_buffer)

            # Add tile to clipped list
            srtm_da_list.append(srtm_cropped_da)

        # After all tiles are processed, merge them
        srtm_da = merge_arrays(srtm_da_list)

        # Return
        return srtm_da

# %%
# Check that it worked
willamette_da = open_merge_srtm(srtm_files= willamette_srtm_results,
                                site_gdf = willamette_gdf)

willamette_da

# %% [markdown]
# ##### Calculate Slope and Aspect Function

# %%
def calc_slope_aspect(srtm_da):
    """
#     Calculate aspect from original DEM and slope from reprojected DEM.

#     Args:
#         srtm_da (xarray.DataArray): merged DEM raster

#     Returns:
#         elev_rpj (xarray.DataArray): DEM reprojected to EPSG:5070
#         slope_da (xarray.DataArray): slope in degrees
#         aspect_da (xarray.DataArray): aspect in degrees from 0 to 360
#     """

    # Calculate aspect from original DEM
    aspect_da = xrspatial.aspect(srtm_da)

    # Convert negative aspect values to 0–360
    aspect_da = (aspect_da + 360) % 360

    # Reproject DEM to meters for slope calculation
    elev_rpj = srtm_da.rio.reproject("EPSG:5070")

    # Calculate slope from reprojected DEM
    slope_da = xrspatial.slope(elev_rpj)

    # Return
    return elev_rpj, slope_da, aspect_da

# %% [markdown]
# ##### TOPO Plot function

# %%
# Create a directory for topographic plots
topo_plots_dir = os.path.join(data_dir, "topography", "plots")
os.makedirs(topo_plots_dir, exist_ok=True)

# Subdirectory for Willamette plots
willamette_topo_plots_dir = os.path.join(topo_plots_dir, "willamette")
os.makedirs(willamette_topo_plots_dir, exist_ok=True)

# Subdirectory for Olympic plots
olympic_topo_plots_dir = os.path.join(topo_plots_dir, "olympic")
os.makedirs(olympic_topo_plots_dir, exist_ok=True)

# %%
### Create function for customizable plots
def plot_topo_layer(site_da, site_gdf, plots_dir, site_fig_name, plot_title,
                    bar_label):
    """
    Plot a topographic raster layer with the terrain caluclations and site boundary

    Args:
        site_da (xarray.DataArray): raster layer to plot
        site_gdf (geopandas.GeoDataFrame): site boundary for overlay
        plots_dir (str): directory to save plot
        site_fig_name (str): figure file name without extension
        plot_title (str): title of plot
        bar_label (str): colorbar label

    Returns:
        matplotlib.axes._axes.Axes:
    """

    # Reproject raster for plotting 
    site_da = site_da.rio.reproject("EPSG:4326")

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot raster
    site_da.plot(
        ax=ax,
        cmap='terrain',
        cbar_kwargs={"label": bar_label}
    )

    # Plot site boundary
    site_gdf.boundary.plot(ax=ax, color='black')

    # Add title and axis labels
    ax.set_title(plot_title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save figure
    plot_path = os.path.join(plots_dir, f"{site_fig_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

    # Return
    return ax

# %% [markdown]
# #### TOPO Workflow Wrapper Function

# %%
def topo_wrapper(site_name, site_gdf, topo_dir, plots_dir, buffer=0.025):
    """
    Run the full topographic workflow for one study site:
    download SRTM, merge DEM, calculate slope/aspect, and plot outputs.

    Args:
        site_name (str): short site name for file naming
        site_gdf (geopandas.GeoDataFrame): study area boundary
        topo_dir (str): directory for SRTM files
        plots_dir (str): directory for saved plots
        buffer (float): buffer in degrees for SRTM search and clipping

    Returns:
        dict: dictionary containing:
            - srtm_files
            - elevation_da
            - elev_rpj
            - slope_da
            - aspect_da
            - elevation_plot
            - slope_plot
            - aspect_plot
    """

    # Ensure plot directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Step 1: Download or find SRTM files
    srtm_files = download_srtm_files(
        site_gdf=site_gdf,
        topo_dir=topo_dir,
        buffer=buffer
    )

    # Step 2: Open, merge, and crop DEM
    elevation_da = open_merge_srtm(
        srtm_files=srtm_files,
        site_gdf=site_gdf,
        buffer=buffer
    )

    # Step 3: Calculate slope and aspect
    elev_rpj, slope_da, aspect_da = calc_slope_aspect(elevation_da)

    # Step 4: Plot elevation
    elevation_plot = plot_topo_layer(
        site_da=elevation_da,
        site_gdf=site_gdf,
        plots_dir=plots_dir,
        site_fig_name=f"{site_name}_elevation",
        plot_title=f"{site_name.title()} Elevation",
        bar_label="Elevation (m)"
    )

    # Step 5: Plot slope
    slope_plot = plot_topo_layer(
        site_da=slope_da,
        site_gdf=site_gdf,
        plots_dir=plots_dir,
        site_fig_name=f"{site_name}_slope",
        plot_title=f"{site_name.title()} Slope",
        bar_label="Slope (degrees)"
    )

    # Step 6: Plot aspect
    aspect_plot = plot_topo_layer(
        site_da=aspect_da,
        site_gdf=site_gdf,
        plots_dir=plots_dir,
        site_fig_name=f"{site_name}_aspect",
        plot_title=f"{site_name.title()} Aspect",
        bar_label="Aspect (degrees)"
    )

    # Return all outputs
    return {
        "srtm_files": srtm_files,
        "elevation_da": elevation_da,
        "elev_rpj": elev_rpj,
        "slope_da": slope_da,
        "aspect_da": aspect_da,
        "elevation_plot": elevation_plot,
        "slope_plot": slope_plot,
        "aspect_plot": aspect_plot
    }

# %%
# Run the wrapper function
willamette_topo = topo_wrapper(
    site_name="willamette",
    site_gdf=willamette_gdf,
    topo_dir=willamette_topo_dir,
    plots_dir=willamette_topo_plots_dir
)

# %%
# Run the wrapper function
olympic_topo = topo_wrapper(
    site_name="olympic",
    site_gdf=olympic_gdf,
    topo_dir=olympic_topo_dir,
    plots_dir=olympic_topo_plots_dir
)

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

# %%
# Create a climate data directories
maca_dir = os.path.join(data_dir, "maca-dir")
os.makedirs(maca_dir, exist_ok=True)

# Create directory for file
maca_pattern = os.path.join(maca_dir, '*.nc')
maca_pattern

# %%
# Temperature in Kelvin
def convert_temperature(temp):

    """
    Convert temperature from Kelvin to Fahrenheit.

    Args:
        temp (xarray.DataArray): temperature in Kelvin

    Returns:
        xarray.DataArray: temperature in Fahrenheit
    """
    return temp * 1.8 - 459.67

# %%
# Convert longitude values
def convert_longitude(longitude):

    """
    Function to convert longitude

    Args:
    longitude:

    Returns:Function to convert longitude
    """
    
    return (longitude - 360) if longitude > 180 else longitude 

# %%
# Define some parameters
site_name = "Willamette"
site_gdf = willamette_gdf
date_range = "2041_2045"
model = "CanESM2"
rcp_value = "rcp45"
climate_var = "pr"

# %%
# Create a maca path
maca_path = os.path.join(
    maca_dir,
    f"maca_{model}_{site_name}_{date_range}_CONUS_month.nc"
)

# %%
# Construct the URL where the climate data lives
maca_url = (
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
    "/MACAV2"
    f"/{model}"
    "/macav2metdata"
    f"_{climate_var}"
    f"_{model}_r1i1p1"
    f"_{rcp_value}"
    f"_{date_range}_CONUS"
    "_monthly.nc"
)

# %%
# Check out the url
maca_url

# %%
# Download it once
if not os.path.exists(maca_path):

    # Open remote dataset
    maca_da = xr.open_dataset(maca_url).squeeze().precipitation

    # Save locally
    maca_da.to_netcdf(maca_path)

    print("downloaded successfully")

else:
    print("file already exists")

# %%
# Open the file
maca_da = xr.open_dataset(maca_path).squeeze().precipitation

# Check CRS
print(maca_da.rio.crs)

# Reproject willamette 
willamette_rpj = willamette_gdf.to_crs(maca_da.rio.crs)

# %%
# Create bounds
bounds_maca = willamette_rpj.total_bounds

# %%
# Convert longitude values
maca_da = maca_da.assign_coords(
    lon= ("lon", [convert_longitude(l) for l in maca_da.lon.values])
)

# %%
maca_da = maca_da.rio.set_spatial_dims(
    x_dim="lon",
    y_dim="lat"

)

# %%
# Crop to bounding box
maca_da_cropped = maca_da.rio.clip_box(*bounds_maca)

# %%
# Create a dictionary containing the site, model info, and cropped climate data
result = dict(
    site_name = site_name,
    climate_model = model, 
    date_range = date_range,
    da = maca_da_cropped
)

# %%
# Plot one month of data
maca_da_cropped.isel(time = 0).plot()

# %%
# Plot the mean
maca_da_cropped.mean(dim = "time").plot()


# %% [markdown]
# ##### Write function for downloading climate data and loop

# %%
# Create function to download climate data
def process_maca_da(site_dict,
                    years_list,
                    model_list,
                    climate_var_list,
                    maca_data_dir):
    """
    Download, process, and crop MACAv2 METDATA monthly climate data
    for multiple sites, years, models, and variables.

    Parameters
    ----------
    site_dict : dict
        Dictionary of site names and GeoDataFrames.
    years_list : list
        List of year ranges to download.
    model_list : list
        List of climate model names.
    climate_var_list : list
        List of climate variables, e.g. ["pr", "tasmax"].
    maca_data_dir : str
        Directory to save downloaded MACA files.

    Returns
    -------
    list
        List of dictionaries containing metadata and cropped DataArrays.
    """

    results = []

    total_runs = (
        len(climate_var_list)
        * len(model_list)
        * len(years_list)
        * len(site_dict)
    )

    pbar = tqdm(total=total_runs, desc="Processing MACA data")

    # Historical 5-year chunks
    historical_periods = [
        "1970_1974", "1975_1979", "1980_1984", "1985_1989",
        "1990_1994", "1995_1999", "2000_2004"
    ]

    for climate_var in climate_var_list:
        for model in model_list:
            for date_range in years_list:

                # Assign scenario
                if date_range in historical_periods:
                    scenario = "historical"
                else:
                    scenario = "rcp45"

                # Map short variable names to actual NetCDF variable names
                if climate_var == "pr":
                    var_name = "precipitation"
                elif climate_var == "tasmax":
                    var_name = "air_temperature"
                else:
                    var_name = climate_var

                # Create local file path
                maca_path = os.path.join(
                    maca_data_dir,
                    f"macav2metdata_{climate_var}_{model}_{scenario}_{date_range}_CONUS_monthly.nc"
                )

                # Build URL using exact monthly file pattern
                maca_url = (
                    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/"
                    f"MACAV2/{model}/"
                    f"macav2metdata_{climate_var}_{model}_r1i1p1_{scenario}_{date_range}_CONUS_monthly.nc"
                )

                # Download file only once
                if not os.path.exists(maca_path):
                    urllib.request.urlretrieve(maca_url, maca_path)
                    print(f"Downloaded: {maca_path}")
                else:
                    print(f"File already exists: {maca_path}")

                # Open local file
                ds = xr.open_dataset(maca_path).squeeze()
                maca_da = ds[var_name]

                # Convert temperature if needed
                if climate_var in ["tas", "tasmin", "tasmax"]:
                    maca_da = convert_temperature(maca_da)

                # Convert longitude values
                maca_da = maca_da.assign_coords(
                    lon=("lon", [convert_longitude(lon) for lon in maca_da.lon.values])
                )

                # Sort longitude values after conversion
                maca_da = maca_da.sortby("lon")

                # Set spatial dimensions
                maca_da = maca_da.rio.set_spatial_dims(
                    x_dim="lon",
                    y_dim="lat"
                )

                # Write CRS if missing
                if maca_da.rio.crs is None:
                    maca_da = maca_da.rio.write_crs("EPSG:4326")

                # Loop through sites and crop
                for site_name, site_gdf in site_dict.items():
                    pbar.set_postfix(
                        site=site_name,
                        model=model,
                        period=date_range,
                        var=climate_var
                    )

                    # Reproject site boundary to match MACA CRS
                    site_rpj = site_gdf.to_crs(maca_da.rio.crs)

                    # Create bounds
                    bounds_maca = site_rpj.total_bounds

                    # Crop to bounding box
                    maca_da_cropped = maca_da.rio.clip_box(*bounds_maca)

                    # Store results
                    result = {
                        "site_name": site_name,
                        "climate_model": model,
                        "date_range": date_range,
                        "scenario": scenario,
                        "climate_var": climate_var,
                        "da": maca_da_cropped
                    }

                    results.append(result)
                    pbar.update(1)

    pbar.close()
    return results

# %%
# Set parameters
site_dict = {
    "Willamette": willamette_gdf,
    "Olympic": olympic_gdf
}

years_list = [
    "1970_1974",
    "1975_1979",
    "1980_1984",
    "1985_1989",
    "1990_1994",
    "1995_1999",
    "2000_2004",
    "2066_2070",
    "2071_2075",
    "2076_2080",
    "2081_2085",
    "2086_2090",
    "2091_2095",
    "2096_2099"
]

model_list = [
    "CanESM2",
    "HadGEM2-CC365",
    "inmcm4",
    "IPSL-CM5A-LR"
]

climate_var_list = [
    "pr",
    "tasmax"
]

# %%
# Check loop/function works on 1
test_results = process_maca_da(
    site_dict={"Willamette": willamette_gdf},
    years_list=["1970_1974"],
    model_list=["CanESM2"],
    climate_var_list=["pr"],
    maca_data_dir=maca_dir
)

# %%
# Check the results
len(test_results)

# %%
# Get climate results
maca_results = process_maca_da(
    site_dict=site_dict,
    years_list=years_list,
    model_list=model_list,
    climate_var_list=climate_var_list,
    maca_data_dir=maca_dir
)

# %%
# Create a function to combine and subset to 30-year periods 
def concat_period(results, site, model, var, period_years):
    """
    Concatenate all DataArrays for a given site/model/variable
    and subset to the target year range.

    Args:
    results (list): Output from process_maca_da(). Each item has 
    site_name, climate_model, climate_var, and a DataArray.
    site (str): Name of the site
    model (str): Climate model name
    var (str): Climate variable
    period_years (tuple): Start and end year, like (1971, 2000)

    Returns:
    xarray.DataArray: All time years combined and clipped to the selected year range
    """
    # Filter relevant chunks
    subset = [
        r["da"]
        for r in results
        if r["site_name"] == site
        and r["climate_model"] == model
        and r["climate_var"] == var
    ]

    # Concatenate along time
    combined = xr.concat(subset, dim="time")

    # Subset to desired years
    start, end = period_years
    combined = combined.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))

    return combined

# %%
# Check to see if it worked
willamette_1971_2000 = concat_period(
    results = maca_results,
    site="Willamette",
    model="CanESM2",
    var="pr",
    period_years=(1971, 2000)
)

willamette_1971_2000

# %%
# Create function to calculate mean 
def mean_raster(da):
    """Return the 30-year mean raster

    Args:
        da (xarray.DataArray): climate raster stack with time dimension

    Returns:
        xarray.DataArray: raster representing the mean value across all time values
    
    """
    return da.mean(dim="time")


# %%
# Create function to write raster 
def write_raster(da, out_path):
    """Write a DataArray to GeoTIFF raster
    
    Args:
        da (xarray.DataArray): Raster to export

        out_path (str): File path for output GeoTIFF

    Returns:
    none
"""
    da.rio.to_raster(out_path)

# %%
### Create a function to generate all climate rasters
# Function developed with help of ChatGPT & adapted by author**

def generate_all_climate_rasters(results, outdir):
    """
    Generate 30-year historical and future mean rasters
    for all sites, models, and climate variables

    Args:
        results (list of dict): Processed climate DataArrays from MACA dataset
        outdir (str): Directory where output rasters will be written

    Returns
        none
    """

    os.makedirs(outdir, exist_ok=True)

    sites = sorted({r["site_name"] for r in results})
    models = sorted({r["climate_model"] for r in results})
    vars_ = sorted({r["climate_var"] for r in results})

    periods = {
        "historical": (1970, 2000),
        "future": (2070, 2099)
    }

    for site in sites:
        for model in models:
            for var in vars_:
                for period_name, years in periods.items():

                    print(f"Processing {site} | {model} | {var} | {period_name}")

                    da = concat_period(
                        results,
                        site=site,
                        model=model,
                        var=var,
                        period_years=years
                    )

                    mean_da = mean_raster(da)

                    out_path = os.path.join(
                        outdir,
                        f"{site}_{model}_{var}_{period_name}_30yr_mean.tif"
                    )

                    write_raster(mean_da, out_path)

                    print(f"Saved: {out_path}")



# %%
# Generate 16 rasters for climate data for two sites - 32 total
generate_all_climate_rasters(
    results=maca_results,
    outdir="final_rasters"
)

# %%
# Check out one of the tifs
willamette_can_pr_mean_historic = rxr.open_rasterio("final_rasters/Willamette_CanESM2_pr_historical_30yr_mean.tif")

willamette_can_pr_mean_historic.plot()
plt.title("Willamette - CanESM2 - PR - Historical 30yr Mean")
plt.show()


# %%
# Check final rasters list
os.listdir("final_rasters")

# %% [markdown]
# **Reflect and respond**: Make sure to include a description of the climate data and how you selected your models. Include a citation of the MACAv2 data.

# %% [markdown]
# Your response here: I chose four models to account for uncertainty in the climate change projections under four climate scenarios. 
# - warm & wet: CanESM2 
# - warm & dry: HadGEM2-CC365
# - cold & dry: inmcm4 
# - cold & wet: IPSL-CM5A-LR 

# %% [markdown]
# Climate projections were obtained from the MACAv2-Livneh statistically downscaled climate dataset available through the Northwest Knowledge Network (Abatzoglou & Brown, 2012).
# 

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

# %% [markdown]
# ### Simpler harmonization workflow without climate

# %%
# Convert to 2D array to match the other rasters (on one tif)
willamette_can_pr_mean_historic = (
    rxr.open_rasterio(
        "final_rasters/Willamette_CanESM2_pr_historical_30yr_mean.tif"
    )
    .squeeze()
)

# %%
# Check shape
willamette_can_pr_mean_historic.shape

# %%
# Check CRS
print(willamette_can_pr_mean_historic.rio.crs)

# %%
# Add CRS to match
willamette_can_pr_mean_historic = willamette_can_pr_mean_historic.rio.write_crs("EPSG:4326")


# %% [markdown]
# ### Create mini function for squeeze and reproject climate rasters

# %%
def mini_load_climate_raster(path, name, crs="EPSG:4326"):
    """
    Load a climate rasters and prepare it for harmonization

    Args:
        path (str): Path to raster file in .tif
        name (str): name to assign to the DataArray
        crs (str): Assign CRS to write if missing 

    Returns:
        clean_climate_raster (xarray.DataArray): Prepared climate raster
    """

    clean_climate_raster = rxr.open_rasterio(path).squeeze()

    if clean_climate_raster.rio.crs is None:
        clean_climate_raster = clean_climate_raster.rio.write_crs(crs)

    clean_climate_raster.name = name

    return clean_climate_raster


# %%
# Repeat for test run for one full set
willamette_can_pr_hist = mini_load_climate_raster(
    "final_rasters/Willamette_CanESM2_pr_historical_30yr_mean.tif",
    "Willamette precip historical CanESM2"
)

willamette_can_pr_future = mini_load_climate_raster(
    "final_rasters/Willamette_CanESM2_pr_future_30yr_mean.tif",
    "Willamette precip future CanESM2"
)

willamette_can_tas_hist = mini_load_climate_raster(
    "final_rasters/Willamette_CanESM2_tasmax_historical_30yr_mean.tif",
    "Willamette temp historical CanESM2"
)

willamette_can_tas_future = mini_load_climate_raster(
    "final_rasters/Willamette_CanESM2_tasmax_future_30yr_mean.tif",
    "Willamette temp future CanESM2"
)

# %%
# Add a name to the arrays for 1 site
willamette_soil_da.name = "Willamette soil pH"
willamette_srtm_da.name = "Willamette elevation"
willamette_slope_4326.name = "Willamette slope"
willamette_aspect.name = "Willamette aspect"

# %%
### Align the grids of the different data layers
willamette_das_list = [
    willamette_soil_da,
    willamette_srtm_da,
    willamette_slope_4326,
    willamette_aspect,
    willamette_can_pr_hist,
    willamette_can_pr_future,
    willamette_can_tas_hist,
    willamette_can_tas_future
]

# %%
# Define the boundaries
willamette_bounds = tuple(willamette_gdf.total_bounds)
willamette_bounds

# Add a small buffer
buffer = 0.025
(willamette_xmin, willamette_ymin, willamette_xmax, willamette_ymax) = willamette_bounds

# Buffer bounding box
willamette_bounds_buffer = (willamette_xmin - buffer,
                           willamette_ymin - buffer,
                           willamette_xmax + buffer,
                           willamette_ymax + buffer)

# %%
# Check out the pre-cropped boundaries
print(willamette_ph_da.rio.bounds())
print(willamette_srtm_da.rio.bounds())

# %%
# Empy list for the cropped and reprojected DAs 
reproj_da_list = []

# Loop through the DAs
for da in tqdm(willamette_das_list):

    # If Willamette is in the name, do this
    if 'Willamette' in da.name:

        # Crop the da
        cropped_da = da.rio.clip_box(*willamette_bounds_buffer)

        # Reproject adn match
        reproj_da = (cropped_da.rio.reproject_match(willamette_ph_da))

        # Remove extra single-band dimension if present
        if reproj_da.ndim == 3:
            reproj_da = reproj_da.squeeze()

        # Add it to the list
        reproj_da_list.append(reproj_da)

reproj_da_list

# %%
# Check they are all the same
for da in reproj_da_list:
    print(da.name, da.shape, da.rio.resolution(), da.rio.crs)

# %%
# Check to make sure they match
print(willamette_ph_da.rio.bounds())
print(reproj_da_list[1].rio.bounds())

# %%
# Create subplots to check
fig, axes = plt.subplots(1, len(reproj_da_list),
                                figsize = (5*len(reproj_da_list), 5))

if len(reproj_da_list) == 1:
    axes = [axes]

for ax, data in zip(axes, reproj_da_list):

    if data.ndim == 3:
        data = data.squeeze()

    data.plot(ax = ax, cmap = 'viridis', add_colorbar = False)

    willamette_gdf.plot(ax = ax, facecolor = 'none', edgecolor = 'white', linewidth = 1)

    ax.set_aspect("equal")
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Write function for load all climate rasters for both sites

# %%
# Create function for loading the climate rasters for harmonization
def load_climate_rasters_clean(site, raster_dir):
    """Function to load the climate rasters, match crs, squeeze to 2D, and name
    
    Args:
        site (str): name of the site used in the raster file name
        raster_dir (str): directory where the GeoTIFF files live for the climate data

    Returns:
        list: list of climate DataArrays
    """

    # Create empty list
    clean_climate_rasters = []

    # Define variable names
    models = ["CanESM2", "HadGEM2-CC365", "IPSL-CM5A-LR", "inmcm4"]
    variables = ["pr", "tasmax"]
    periods = ["historical", "future"]

    # Rename
    var_names = {
        "pr": "precip",
        "tasmax": "temp"
    }

    # Loop over all models, variables, and time periods
    for model in models:
        for variable in variables:
            for period in periods:

                # Set file path
                path = f"{raster_dir}/{site}_{model}_{variable}_{period}_30yr_mean.tif"

                # Squeeze into 2D
                climate_da = rxr.open_rasterio(path).squeeze()

                # Reproject into matching CRS if noe
                if climate_da.rio.crs is None:
                    climate_da = climate_da.rio.write_crs("EPSG:4326")

                # Name the raster
                climate_da.name = f"{site} {var_names[variable]} {period} {model}"

                # Append to empty list
                clean_climate_rasters.append(climate_da)

    # Return
    return clean_climate_rasters

# %%
# Check that it works
willamette_climate = load_climate_rasters_clean(
    site="Willamette",
    raster_dir="final_rasters"
)

# %%
# Check it out
for da in willamette_climate:
    print(da.name)

# %%
# Run it for Olympic
olympic_climate = load_climate_rasters_clean(
    site="Olympic",
    raster_dir="final_rasters"
)

# %%
# Check it out
for da in olympic_climate:
    print(da.name)

# %% [markdown]
# #### Write a function to create da_lists to be harmonized

# %%
# Make function to create lists
def make_site_raster_list(soil_da, elevation_da, slope_da, aspect_da, climate_rasters):
    """
    Combine static environmental rasters and climate rasters for one site.

    Args:
        soil_da (xarray.DataArray): soil raster
        elevation_da (xarray.DataArray): elevation raster
        slope_da (xarray.DataArray): slope raster
        aspect_da (xarray.DataArray): aspect raster
        climate_rasters (list): list of climate rasters

    Returns:
        list: combined list of rasters for one site
    """
    site_raster_list = [soil_da, elevation_da, slope_da, aspect_da] + climate_rasters
    return site_raster_list

# %%
# Create list for willamette
willamette_das_list = make_site_raster_list(
    willamette_soil_da,
    willamette_srtm_da,
    willamette_slope_4326,
    willamette_aspect,
    willamette_climate
)

# %%
# Add names for olympic topo rasters
olympic_soil_da.name = "Olympic soil pH"
olympic_topo["elevation_da"].name = "Olympic elevation"
olympic_topo["slope_da"].name = "Olympic slope"
olympic_topo["aspect_da"].name = "Olympic aspect"

# %%
# Create olympic das list
olympic_das_list = make_site_raster_list(
    olympic_soil_da,
    olympic_topo["elevation_da"],
    olympic_topo["slope_da"],
    olympic_topo["aspect_da"],
    olympic_climate
)

# %%
# Check that it worked
for da in olympic_das_list:
    print(da.name)

# %% [markdown]
# #### Create function to harmonize all layers, all models, all sites

# %%
# Create funciton to harmonize all the layers across both sites, models, time periods
def harmonize_rasters(site_gdf, das_list, template_da, buffer=0.025):
    """
    Crop rasters to buffered site bounds and reproject to match a template grid.

    Args:
        site_gdf (GeoDataFrame): site boundary
        das_list (list): list of rasters for the site
        template_da (DataArray): raster whose grid all others should match
        buffer (float): padding added to bounding box

    Returns:
        list: harmonized rasters
    """

    # Set bounds of area
    bounds = tuple(site_gdf.total_bounds)
    xmin, ymin, xmax, ymax = bounds

    # Create buffer
    bounds_buffer = (
        xmin - buffer,
        ymin - buffer,
        xmax + buffer,
        ymax + buffer
    )

    # Create empty list
    reproj_da_list = []

    # Create loop for both sites
    for da in tqdm(das_list):
        
        # Crop to bounds buffer and double check crs
        cropped_da = da.rio.clip_box(*bounds_buffer, crs=site_gdf.crs)

        # Match the reproject to the template
        reproj_da = cropped_da.rio.reproject_match(template_da)

        # Make sure its 2D by squeeze
        if reproj_da.ndim == 3:
            reproj_da = reproj_da.squeeze()

        # Append to empty list
        reproj_da_list.append(reproj_da)

    # Return list
    return reproj_da_list

# %%
# Create harmonized rasters for willamette
willamette_harmonized = harmonize_rasters(
    willamette_gdf,
    willamette_das_list,
    willamette_soil_da
)

# %%
# Create harmonized raster for olympic
olympic_harmonized = harmonize_rasters(
    olympic_gdf,
    olympic_das_list,
    olympic_soil_da
)

# %%
# Check that everything looks right
for da in olympic_harmonized:
    print(da.name, da.shape, da.rio.resolution(), da.rio.crs)

# %%
# Check that olympic also looks right
len(olympic_harmonized)

# %% [markdown]
# ### Seperate rasters for feeding into fuzzy pipeline

# %%
# Seperate out rasters for Gaussian fuzzy
willamette_gaussian_layers = [
    da for da in willamette_harmonized
    if any(k in da.name.lower() for k in ["soil", "ph", "elev", "srtm", "slope", "pr"])
]

# %%
# Check it out
[da.name for da in willamette_gaussian_layers]

# %%
# Seperate out for aspect fuzzy function
willamette_aspect_layers = [
    da for da in willamette_harmonized
    if "aspect" in da.name.lower()
]

# %%
# Check it out
[da.name for da in willamette_aspect_layers]

# %%
# Seperate out for non-Gaussian temperature function
willamette_temp_layers = [
    da for da in willamette_harmonized
    if "tasmax" in da.name.lower() or "temp" in da.name.lower()
]

# %%
# Check it out
[da.name for da in willamette_temp_layers]

# %%
# Seperate out rasters for Gaussian fuzzy
olympic_gaussian_layers = [
    da for da in olympic_harmonized
    if any(k in da.name.lower() for k in ["soil", "ph", "elev", "srtm", "slope", "pr"])
]

# %%
# Seperate out for aspect fuzzy function
olympic_aspect_layers = [
    da for da in olympic_harmonized
    if "aspect" in da.name.lower()
]

# %%
# Seperate out for non-Gaussian temperature function
olympic_temp_layers = [
    da for da in olympic_harmonized
    if "tasmax" in da.name.lower() or "temp" in da.name.lower()
]

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

# %% [markdown]
# ### Work through fuzzy logic model for 1 layer - soil pH

# %%
### Create fuzzy logic model for habitat suitability (soil pH)
# Define environmental parameter: soil PH
optimal_value = 5.5
tolerance_range = 0.5 

# Define parameters
site_name = "Willamette"
raster_name = "soil_ph_suitability"

# %%
# Run the fuzzy logic model for one layer to see how it works
test_soil_raster = willamette_ph_da

# Fuzzy logic math
difference = test_soil_raster - optimal_value
squared_difference = difference ** 2
scaled_difference = squared_difference / (2 * tolerance_range**2)
negative_scaled = -scaled_difference
test_soil_suitability = np.exp(negative_scaled)


# %%
# Plot it
test_soil_suitability.plot()

# %% [markdown]
# #### Guassian Standard Fuzzy Logic - Soil pH, elevation, slope, and precipitation

# %%
# Create gaussian math function
def gaussian_fuzzy(raster, optimal_value, tolerance_range):
    """
    Convert a raster to fuzzy suitability (0-1) using a Gaussian math

    Args
        raster (xarray.DataArray): Input raster layer
        optimal_value (float): Ecologically optimal value for the variable
        tolerance_range (float): width of tolerance around the optimal value

    Returns
        xarray.DataArray: fuzzy suitability raster with values from 0 to 1
    """
    # Guassian fuzzy logic math
    # Compute how far each raster cell is from optimal value
    difference = raster - optimal_value

    # Square the difference so all values become positive 
    # And large deviations strongly penalized
    squared_difference = difference ** 2

    # Scale the squared difference by the Gaussian denominator:
    # 2 * tolerance range square (how quickly suitability drops as you move from optimal)
    scaled_difference = squared_difference / (2 * tolerance_range ** 2)

    # Apply the negative sign required by gaussian formula
    negative_scaled = -scaled_difference

    # Apply the exponential functino to convert scaled distances into
    # Fuzzy suitabilty values between 0 and 1
    suitability = np.exp(negative_scaled)

    return suitability

# %%
### Funciton to fuzzify gausian layers 
# Function developed with help of ChatGPT & adapted by author
def fuzzify_gaussian_layers(raster_list):
    """
    Apply Gaussian fuzzy membership function to a list of raster layers

    This function loops through a list of rasters and applies
    a Gaussian fuzzy suitability transformation to variables that are modeled
    with symmetric ecological tolerances (e.g., soil pH, slope, elevation,
    and precipitation). The appropriate optimal value and tolerance range
    are assigned based on the raster name.

    Each raster is converted to a fuzzy suitability surface with values
    between 0 and 1, where:
        1 = optimal environmental conditions
        0 = highly unsuitable conditions

    Args
        raster_list (list of xarray.DataArray): 
            list of harmonized raster layers to be fuzzified

    Returns
        list of xarray.DataArray
        a list of fuzzified raster layers with values between 0 and 1.
        Each output raster is renamed with the suffix "_fuzzy"
    """
    # Create empty list for fuzzy layers
    fuzzy_layers = []

    # For each raster loop
    for raster in raster_list:
        name = raster.name.lower()

        # For each type apply optimal and tolerance
        if "soil" in name or "ph" in name:
            optimal = 5.5
            tolerance = 0.5

        elif "slope" in name:
            optimal = 20
            tolerance = 25

        elif "elev" in name or "srtm" in name:
            optimal = 760
            tolerance = 700

        elif "pr" in name or "precip" in name:
            optimal = 140
            tolerance = 60

        else:
            continue
        
        # Apply fuzzy logic gausian math
        fuzzy = gaussian_fuzzy(raster, optimal, tolerance)

        # Give it a new name with fuzzy
        fuzzy.name = f"{raster.name}_fuzzy"

        # Append the list
        fuzzy_layers.append(fuzzy)

    # Return the list
    return fuzzy_layers

# %%
# Apply fuzzification to both layers
willamette_gaussian_fuzzy = fuzzify_gaussian_layers(willamette_gaussian_layers)

olympic_gaussian_fuzzy = fuzzify_gaussian_layers(olympic_gaussian_layers)

# %%
# Check to see that it worked 
[r.name for r in willamette_gaussian_fuzzy]

# %%
for r in willamette_gaussian_layers:
    if "canesm2" in r.name.lower() and "pr" in r.name.lower() and "historical" in r.name.lower():
        print(r.name, float(r.min()), float(r.max()), float(r.mean()))

# %%
# Plot one raster to double check
willamette_gaussian_fuzzy[0].plot()

# %% [markdown]
# #### Aspect Fuzzy Logic

# %%
# Create unction to calculate aspect with gaussian distribution
# Function developed with help of ChatGPT & adapted by author**
def aspect_fuzzy(raster, optimal_aspect, tolerance):
    """
    Convert an aspect raster to fuzzy suitability (0-1) using circular
    angular distance and a Gaussian distribution

    Because aspect is circular (0° = 360°), suitability is calculated using
    the shortest angular distance from the optimal direction and then
    converted to fuzzy suitability using a Gaussian function.

    Args
        raster (xarray.DataArray): input aspect raster in degrees (0-360)
        optimal_aspect (float): cologically optimal aspect in degrees
        tolerance (float): angular tolerance around the optimum in degrees

    Returns
        xarray.DataArray: Fuzzy suitability raster with values from 0 to 1
    """
    # Difference between raster aspect and optimal aspect
    angular_difference = abs(raster - optimal_aspect)

    # Account for circular nature of aspect (shortest angular distance)
    circular_distance = np.minimum(angular_difference, 360 - angular_difference)

    # Gaussian fuzzy membership function
    scaled_difference = (circular_distance ** 2) / (2 * tolerance ** 2)
    suitability = np.exp(-scaled_difference)

    return suitability

# %%
# Create function for fuzzy logic for aspect layer for both sites (different preferences)
# Function developed with help of ChatGPT & adapted by author
def fuzzify_aspect_layers(raster_list, site_name):
    """
    Apply fuzzy suitability transformation to aspect raster(s) for a site.

    Olympic Peninsula (cooler, wetter):
    Douglas-fir is assumed to prefer warmer slopes with an optimal aspect of
    200° (south-southwest) and a tolerance of 90°.

    Willamette Valley (warmer, drier):
    Douglas-fir is assumed to prefer cooler slopes with an optimal aspect of
    340° (north-northwest) and a tolerance of 90°.

    Args
        raster_list (list of xarray.DataArray): list containing aspect raster(s)
        site_name (str): study site name ("Olympic" or "Willamette")

    Returns
        list of xarray.DataArray: list of fuzzified aspect raster(s)
    """
    fuzzy_layers = []

    # Apply optimal aspect for olympic
    if site_name.lower() == "olympic": 
        optimal_aspect = 200 # SSW (South - Southwest)
        tolerance = 90

    # Otherwise apply optimal aspect for willamette
    elif site_name.lower() == "willamette":
        optimal_aspect = 340 # NNW (North - Northwest)
        tolerance = 90

    # Error if doesn't work
    else:
        raise ValueError("site_name must be 'Olympic' or 'Willamette'")

    # For each raster do the fuzzy logic
    for raster in raster_list:

        # Apply fuzzy logic math
        fuzzy = aspect_fuzzy(raster, optimal_aspect, tolerance)

        # Add fuzzy logic name
        fuzzy.name = f"{raster.name}_fuzzy"
        
        # Append to empty list
        fuzzy_layers.append(fuzzy)

    # Return
    return fuzzy_layers

# %%
# Fuzzify aspect layers for both sites
# Apply fuzzification to both layers
willamette_aspect_fuzzy = fuzzify_aspect_layers(willamette_aspect_layers, "Willamette")

olympic_aspect_fuzzy = fuzzify_aspect_layers(olympic_aspect_layers, "Olympic")


# %%
# Check that it worked
print(len(willamette_aspect_fuzzy))
print(len(olympic_aspect_fuzzy))

# %%
# Plot one raster to double check
willamette_aspect_fuzzy[0].plot()

# %% [markdown]
# #### Temperature Fuzzy Logic

# %%
# Create function to calculate temperature optimal for fuzzy
# Function developed with help of ChatGPT & adapted by author
def temp_fuzzy(raster, optimal_temp=68, lower_limit=10, upper_limit=80):
    """
    Apply an asymmetric fuzzy suitability function for Douglas-fir temperature
    preferences using exponential decay from the optimal temperature

    Suitability peaks at 68°F, declines gradually toward colder temperatures
    down to 10°F, and declines more sharply toward warmer temperatures up to
    80°F. Values outside this range are set to 0.

    Args
        raster (xarray.DataArray): Temperature raster in degrees Fahrenheit
        optimal_temp (float): Optimal temperature
        lower_limit (float): lower tolerance limit
        upper_limit (float): upper tolerance limit

    Returns
        xarray.DataArray: Fuzzy suitability raster with values from 0 to 1
    """

    # Distance from optimal value
    distance = raster - optimal_temp

    # Separate warm vs cold sides
    cold_side = xr.where(distance < 0, np.exp(-(distance**2) / (2 * 30**2)), 1)
    warm_side = xr.where(distance > 0, np.exp(-(distance**2) / (2 * 10**2)), 1)

    # Create one asymetric suitability curve
    suitability = cold_side * warm_side

    # Set tolerance limits
    suitability = xr.where(
        (raster < lower_limit) | (raster > upper_limit),
        0,
        suitability
    )

    return suitability

# %%
# Create functin to apply fuzzy logic to temperature layers in loop
# Function developed with help of ChatGPT & adapted by author
def fuzzify_temp_layers(raster_list):
    """
    Apply the temperature fuzzy suitability function to a list of
    temperature rasters.

    Args:
        raster_list (list of xarray.DataArray): list of harmonized temperature rasters

    Returns
        list of xarray.DataArray: list of fuzzified temperature rasters with values from 0 to 1
    """
    fuzzy_layers = []

    for raster in raster_list:
        fuzzy = temp_fuzzy(raster, optimal_temp=68, lower_limit=10, upper_limit=80)
        fuzzy.name = f"{raster.name}_fuzzy"
        fuzzy_layers.append(fuzzy)

    return fuzzy_layers

# %%
# Apply fuzzy logic to both sites temp
willamette_temp_fuzzy = fuzzify_temp_layers(willamette_temp_layers)
olympic_temp_fuzzy = fuzzify_temp_layers(olympic_temp_layers)

# %%
# Check that it worked
[r.name for r in willamette_temp_fuzzy]

# %%
# Plot one to double check
willamette_temp_fuzzy[0].plot()

# %% [markdown]
# #### Combine fuzzy rasters for each scenario

# %% [markdown]
# ##### Test on one scenario - Willamette, CanESM2, Historical

# %%
# Create static fuzzy list with soil ph, elevation, slope, aspect
willamette_static_fuzzy = willamette_gaussian_fuzzy[:3] + willamette_aspect_fuzzy

# %%
# Filter historical precip layers for one model canesm2
willamette_hist_pr_canesm2 = [
    r for r in willamette_gaussian_fuzzy
    if "canesm2" in r.name.lower() and "pr" in r.name.lower() and "historical" in r.name.lower()
][0]

# Filter historical temp layers for one model canesm2
willamette_hist_temp_canesm2 = [
    r for r in willamette_temp_fuzzy
    if "canesm2" in r.name.lower() and "historical" in r.name.lower()
][0]

# %%
# Combine all layers for historical and one climate model 
willamette_hist_canesm2_layers = (
    willamette_static_fuzzy +
    [willamette_hist_pr_canesm2, willamette_hist_temp_canesm2]
)

# %%
# Create function to combine suitability layers
def combine_fuzzy_layers(suitability_layers):
    """
    Combine multiple fuzzy suitability rasters using multiplication.

    Args
        suitability_layers (list of xarray.DataArray): 
        list of fuzzy suitability rasters scaled from 0 to 1

    Returns
        xarray.DataArray: Combined habitat suitability raster
    """
    # Combine all suitability layers
    combined_suitability = suitability_layers[0]

    # Loop over all layers and multiply
    for layer in suitability_layers[1:]:
        combined_suitability = combined_suitability * layer

    # Return one 
    return combined_suitability

# %%
# Combine all layers for one suitability map
willamette_hist_canesm2_suitability = combine_fuzzy_layers(willamette_hist_canesm2_layers)

# %%
# Plot it to see if it worked
willamette_hist_canesm2_suitability.plot(
    cmap="viridis")

# %%
for layer in willamette_hist_canesm2_layers:
    print(layer.name, float(layer.min()), float(layer.max()))

# %% [markdown]
# #### Function to combine all scenarios and sites

# %%
# Make folder for suitability rasters
suitability_dir = os.path.join(data_dir, "suitability_maps")
os.makedirs(suitability_dir, exist_ok=True)

# %%
# Create a function to build all site suitability maps
def build_site_suitability_maps(site_name, gaussian_fuzzy, aspect_fuzzy, temp_fuzzy):
    """
    Function to build all site suitability maps for each site and all scenarios

    Args
        site_name (str): name of site
        gaussian_fuzzy (list of xarray.DataArray): 
            fuzzified rasters with simple gaussian - soil, elevation, slope, precip
        aspect_fuzzy (list of xarray.DataArray): fuzzified aspect raster for each site
        temp_fuzzy (list of xarray.DataArray): 
            list of fuzzified rasters with asymetric distribution for temperature

    Return 
        dict: 
            dictionary of output file paths for all combined habitat suitability rasters, 
            for all models, time periods, and sites
    """
    # Filter only the static gaussian rasters: soil, elevation, slope
    static_gaussian = [
        r for r in gaussian_fuzzy
        if ("soil" in r.name.lower() or "ph" in r.name.lower()
            or "elev" in r.name.lower() or "srtm" in r.name.lower()
            or "slope" in r.name.lower())
    ]

    # Add static gaussian + aspect together
    static_fuzzy = static_gaussian + aspect_fuzzy 

    # Define climate models and time periods
    models = ["CanESM2", "HadGEM2-CC365", "inmcm4", "IPSL-CM5A-LR"]
    time_periods = ["historical", "future"]

    # Create an empty dictionary for outputs
    suitability_outputs = {}

    # Loop through each climate model 
    for model in models:
        
        # Loop through each time period
        for time_period in time_periods:
        
            # Find matching precipitation layer
            pr_layer = [
                r for r in gaussian_fuzzy
                if model.lower() in r.name.lower()
                and "pr" in r.name.lower()
                and time_period in r.name.lower()
            ][0]

            # Find matching temperature layer
            temp_layer = [
                r for r in temp_fuzzy
                if model.lower() in r.name.lower()
                and time_period in r.name.lower()
            ][0]

            # Combine the static layers with matching climate layers
            layers_to_combine = static_fuzzy + [pr_layer, temp_layer]

            # Create combined suitability raster
            combined = combine_fuzzy_layers(layers_to_combine)

            # Name the output raster
            combined.name = f"{site_name}_{model}_{time_period}_suitability"

            # Create output file path
            output_file = os.path.join(suitability_dir, f"{combined.name}.tif")

            # Save raster
            combined.rio.to_raster(output_file)

            # Add output file path to output dictionary
            suitability_outputs[combined.name] = output_file

            # Delete combined raster from memory (to preserve memory)
            del combined

    return suitability_outputs

# %%
# Run build site suitability maps function for willamette
willamette_suitability_maps = build_site_suitability_maps(
    "Willamette",
    willamette_gaussian_fuzzy,
    willamette_aspect_fuzzy,
    willamette_temp_fuzzy
)

# %%
# Check that its the right amount
len(willamette_suitability_maps)

# %%
# Run build site suitability maps function for willamette
olympic_suitability_maps = build_site_suitability_maps(
    "Olympic",
    olympic_gaussian_fuzzy,
    olympic_aspect_fuzzy,
    olympic_temp_fuzzy
)

# %%
# Check that its the right amount
len(olympic_suitability_maps)

# %% [markdown]
# ## STEP 5: Present your results
# Generate some plots that show your key findings of habitat suitability in your study sites across the different time periods and climate models. Don’t forget to interpret your plots!

# %% [markdown]
# #### Test plotting on one site/one model/one time period

# %%
# Define test path
test_path = willamette_suitability_maps["Willamette_CanESM2_historical_suitability"]

# %%
# Open raster and squeeze
test_raster = rxr.open_rasterio(test_path, masked=True).squeeze()

# %%
# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

# Set parameters
test_raster.plot(
    ax=ax,
    cmap="viridis", # Color
    cbar_kwargs={"label": "Suitability"} # Label side bar
)

# Set titles
ax.set_title("Willamette - CanESM2 Historical Suitability")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Disply
plt.show()

# %%
### Display historical vs future maps for one site/one model
hist_path = willamette_suitability_maps["Willamette_CanESM2_historical_suitability"]
fut_path = willamette_suitability_maps["Willamette_CanESM2_future_suitability"]

# Open rasters
hist_raster = rxr.open_rasterio(hist_path, masked=True).squeeze()
fut_raster = rxr.open_rasterio(fut_path, masked=True).squeeze()

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Set plot parameters
hist_plot = hist_raster.plot(
    ax=axes[0],
    cmap="viridis",
    vmin=0,
    vmax=1,
    add_colorbar=False
)

# Set plot parameters
fut_plot = fut_raster.plot(
    ax=axes[1],
    cmap="viridis",
    vmin=0,
    vmax=1,
    add_colorbar=False
)

# Add titles
axes[0].set_title("Historical Suitability")
axes[1].set_title("Future Suitability")

# Set axises titles
for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

# Leave space on right for colorbar
fig.subplots_adjust(right=0.88)

# Shared colorbar for both plots
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(hist_plot, cax=cbar_ax, label="Suitability")

# Set overall title
plt.suptitle("Willamette - CanESM2")
plt.show()

# %%
# Calculate the difference in future vs historical
diff_raster = fut_raster - hist_raster

# Symmetric color scale around 0
abs_max = float(np.nanmax(np.abs(diff_raster.values)))

# Create plots
fig, ax = plt.subplots(figsize=(8, 6))

# plot the difference for plots
diff_plot = diff_raster.plot(
    ax=ax,
    cmap="RdBu_r",
    vmin=-abs_max,
    vmax=abs_max,
    cbar_kwargs={"label": "Suitability Change"}
)

# Set titles and labels
ax.set_title("Willamette - CanESM2\nFuture - Historical")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Display
plt.show()

# %%
# Double check what is in there
list(willamette_suitability_maps.keys())

# %%
# Folder containing the notebook or script
repo_root = Path.cwd()

# %%
# Create directory from the repo root
suitability_dir = (
    repo_root.parent / "spring-2026-data" / "douglas-fir-habitat-suitability" / "suitability_maps"
)
# Make sure it exists
suitability_dir.mkdir(parents=True, exist_ok=True)


# %%
# Grab all rasters into dictionary
suitability_maps = {}

# Grab all of them
for f in suitability_dir.glob("*_suitability.tif"):
    key = f.stem
    suitability_maps[key] = f

# %%
# recreate site-specific dictionaries
willamette_suitability_maps = {
    k: v for k, v in suitability_maps.items() if "Willamette" in k
}

olympic_suitability_maps = {
    k: v for k, v in suitability_maps.items() if "Olympic" in k
}

# %%
# quick check
print(len(willamette_suitability_maps))
print(len(olympic_suitability_maps))

# %%
# Create a 3x4 map grid to look at historical, future, and change in habitat suitability
site = "Willamette"

# Define the models
models = [
    "CanESM2",
    "HadGEM2-CC365",
    "inmcm4",
    "IPSL-CM5A-LR"
]

# Create plots
fig, axes = plt.subplots(len(models), 3, figsize=(15, 16))

# Loop over all models
for i, model in enumerate(models):

    hist_key = f"{site}_{model}_historical_suitability"
    fut_key = f"{site}_{model}_future_suitability"

    # Open raster
    hist = rxr.open_rasterio(
        willamette_suitability_maps[hist_key],
        masked=True
    ).squeeze()

    # Open raster
    fut = rxr.open_rasterio(
        willamette_suitability_maps[fut_key],
        masked=True
    ).squeeze()

    # Calcualte the difference
    diff = fut - hist

    # Historical plot and parameters
    hist_plot = hist.plot(
        ax=axes[i,0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        add_colorbar=False # dont add colorbar on the left plot
    )

    # Future  plot and parameters
    fut.plot(
        ax=axes[i,1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        add_colorbar=False
    )

    # Change
    abs_max = float(np.nanmax(np.abs(diff.values)))

    # Create the change plot
    diff_plot = diff.plot(
        ax=axes[i,2],
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        add_colorbar=False
    )

    axes[i,0].set_title(f"{model} Historical")
    axes[i,1].set_title(f"{model} Future")
    axes[i,2].set_title(f"{model} Change")

    for j in range(3):
        axes[i,j].set_xlabel("")
        axes[i,j].set_ylabel("")

    # Free memory each loop
    del hist, fut, diff

# spacing
fig.subplots_adjust(right=0.88, hspace=0.25)

# colorbars
cbar1 = fig.add_axes([0.90, 0.55, 0.015, 0.3])
fig.colorbar(hist_plot, cax=cbar1, label="Suitability")

cbar2 = fig.add_axes([0.90, 0.15, 0.015, 0.3])
fig.colorbar(diff_plot, cax=cbar2, label="Suitability Change")

plt.suptitle(
    "Willamette Douglas-Fir Habitat Suitability Across Climate Models",
    fontsize=16
)

plt.show()

# %%
### Write function for plotting maps together for both sites
# Function developed with help of ChatGPT & adapted by author
def plot_site_suitability_comparison(
    site,
    maps,
    models,
    coarsen_factor=4,
    boundary_gdf=None,
    gbif_gdf=None,
    point_size=8,
    point_alpha=0.6
):
    """
    Plot historical, future, and change suitability maps for all climate models
    for a given site, with optional site boundary and GBIF occurrence overlays

    Args
        site (str): Site name, e.g. "Willamette" or "Olympic"
        maps (dict): Dictionary of raster file paths keyed by raster name
        models (list): List of climate model names
        coarsen_factor (int): default=4, Factor to downsample rasters for plotting only
        boundary_gdf (GeoDataFrame): optional, site boundary to overlay
        gbif_gdf (GeoDataFrame) optional: GBIF occurrence points to overlay
        point_size (int) default=8, GBIF point size
        point_alpha (float): default=0.6, GBIF point transparency
    """

    # Create folder for saved plots
    output_dir = Path("final_suitability_plots")
    output_dir.mkdir(exist_ok=True)

    # Keep track of saved files
    saved_files = []

    # First pass: find the largest change value across models
    # This allows us to use the same color scale for all change maps
    max_abs_change = 0

    for model in models:

        # Build raster names
        hist_key = f"{site}_{model}_historical_suitability"
        fut_key = f"{site}_{model}_future_suitability"

        # Open rasters
        hist = rxr.open_rasterio(maps[hist_key], masked=True).squeeze()
        fut = rxr.open_rasterio(maps[fut_key], masked=True).squeeze()

        # Downsample rasters to reduce memory when plotting
        hist_plot_r = hist.coarsen(
            x=coarsen_factor,
            y=coarsen_factor,
            boundary="trim"
        ).mean()

        fut_plot_r = fut.coarsen(
            x=coarsen_factor,
            y=coarsen_factor,
            boundary="trim"
        ).mean()

        # Calculate change raster
        diff_plot_r = fut_plot_r - hist_plot_r

        # Find maximum absolute change
        this_max = float(np.nanmax(np.abs(diff_plot_r.values)))

        # Keep the largest value across all models
        max_abs_change = max(max_abs_change, this_max)

        # Free memory
        del hist, fut, hist_plot_r, fut_plot_r, diff_plot_r

    # Create figure layout
    # rows = models, columns = historical / future / change
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)))

    # Handle case of only one model
    if len(models) == 1:
        axes = np.array([axes])

    # Second pass: create plots
    for i, model in enumerate(models):

        # Build raster names
        hist_key = f"{site}_{model}_historical_suitability"
        fut_key = f"{site}_{model}_future_suitability"

        # Open rasters
        hist = rxr.open_rasterio(maps[hist_key], masked=True).squeeze()
        fut = rxr.open_rasterio(maps[fut_key], masked=True).squeeze()

        # Downsample rasters for plotting
        hist_plot_r = hist.coarsen(
            x=coarsen_factor,
            y=coarsen_factor,
            boundary="trim"
        ).mean()

        fut_plot_r = fut.coarsen(
            x=coarsen_factor,
            y=coarsen_factor,
            boundary="trim"
        ).mean()

        # Calculate change raster
        diff_plot_r = fut_plot_r - hist_plot_r

        # Reproject overlays so they match raster CRS
        boundary_proj = None
        gbif_proj = None

        if boundary_gdf is not None:
            boundary_proj = boundary_gdf.to_crs(hist_plot_r.rio.crs)

        if gbif_gdf is not None:
            gbif_proj = gbif_gdf.to_crs(hist_plot_r.rio.crs)

        # Historical map plot and parameters
        hist_plot = hist_plot_r.plot(
            ax=axes[i, 0],
            cmap="viridis",
            vmin=0,
            vmax=1,
            add_colorbar=False
        )

        # Add site boundary if provided
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=axes[i, 0],
                color="white",
                linewidth=1.0
            )

        # Add GBIF points if provided
        if gbif_proj is not None:
            gbif_proj.plot(
                ax=axes[i, 0],
                color="white",
                markersize=point_size,
                alpha=point_alpha
            )

        # Future map
        fut_plot_r.plot(
            ax=axes[i, 1],
            cmap="viridis",
            vmin=0,
            vmax=1,
            add_colorbar=False
        )

        # Add boundary and gbif if wanted
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=axes[i, 1],
                color="white",
                linewidth=1.0
            )

        if gbif_proj is not None:
            gbif_proj.plot(
                ax=axes[i, 1],
                color="white",
                markersize=point_size,
                alpha=point_alpha
            )

        # Change map (future - historical)
        diff_plot = diff_plot_r.plot(
            ax=axes[i, 2],
            cmap="RdBu_r",
            vmin=-max_abs_change,
            vmax=max_abs_change,
            add_colorbar=False
        )

        # Add boundary and gbif if wanted
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=axes[i, 2],
                color="black",
                linewidth=1.0
            )

        if gbif_proj is not None:
            gbif_proj.plot(
                ax=axes[i, 2],
                color="black",
                markersize=point_size,
                alpha=point_alpha
            )

        # Titles and labels
        axes[i, 0].set_title(f"{model} Historical")
        axes[i, 1].set_title(f"{model} Future")
        axes[i, 2].set_title(f"{model} Change")

        # Remove axis labels for cleaner maps
        for j in range(3):
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")

        # Also save each model as its own figure
        model_fig, model_axes = plt.subplots(1, 3, figsize=(15, 4))

        # Historical map plot and parameters
        model_hist_plot = hist_plot_r.plot(
            ax=model_axes[0],
            cmap="viridis",
            vmin=0,
            vmax=1,
            add_colorbar=False
        )

        # Add site boundary if provided
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=model_axes[0],
                color="white",
                linewidth=1.0
            )

        # Add GBIF points if provided
        if gbif_proj is not None:
            gbif_proj.plot(
                ax=model_axes[0],
                color="white",
                markersize=point_size,
                alpha=point_alpha
            )

        # Future map
        fut_plot_r.plot(
            ax=model_axes[1],
            cmap="viridis",
            vmin=0,
            vmax=1,
            add_colorbar=False
        )

        # Add boundary and gbif if wanted
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=model_axes[1],
                color="white",
                linewidth=1.0
            )

        if gbif_proj is not None:
            gbif_proj.plot(
                ax=model_axes[1],
                color="white",
                markersize=point_size,
                alpha=point_alpha
            )

        # Change map (future - historical)
        model_diff_plot = diff_plot_r.plot(
            ax=model_axes[2],
            cmap="RdBu_r",
            vmin=-max_abs_change,
            vmax=max_abs_change,
            add_colorbar=False
        )

        # Add boundary and gbif if wanted
        if boundary_proj is not None:
            boundary_proj.boundary.plot(
                ax=model_axes[2],
                color="black",
                linewidth=1.0
            )

        if gbif_proj is not None:
            gbif_proj.plot(
                ax=model_axes[2],
                color="black",
                markersize=point_size,
                alpha=point_alpha
            )

        # Titles and labels
        model_axes[0].set_title(f"{model} Historical")
        model_axes[1].set_title(f"{model} Future")
        model_axes[2].set_title(f"{model} Change")

        # Remove axis labels for cleaner maps
        for j in range(3):
            model_axes[j].set_xlabel("")
            model_axes[j].set_ylabel("")

        # Adjust layout and add colorbars
        model_fig.subplots_adjust(right=0.88, wspace=0.20)

        # Suitability colorbar
        cbar1 = model_fig.add_axes([0.90, 0.55, 0.015, 0.25])
        model_fig.colorbar(model_hist_plot, cax=cbar1, label="Suitability")

        # Change colorbar
        cbar2 = model_fig.add_axes([0.90, 0.15, 0.015, 0.25])
        model_fig.colorbar(model_diff_plot, cax=cbar2, label="Suitability Change")

        # Figure title
        plt.suptitle(
            f"{site} Douglas-Fir Habitat Suitability - {model}",
            fontsize=16,
            y=0.98
        )

        # Save model plot
        model_save_path = output_dir / f"{site}_{model}_suitability_comparison.png"
        model_fig.savefig(model_save_path, dpi=300, bbox_inches="tight")
        saved_files.append(model_save_path)

        # Close model figure to free memory
        plt.close(model_fig)

        # Free memory
        del hist, fut, hist_plot_r, fut_plot_r, diff_plot_r, model_fig, model_axes

    # Adjust layout and add colorbars
    fig.subplots_adjust(right=0.88, hspace=0.30)

    # Suitability colorbar
    cbar1 = fig.add_axes([0.90, 0.55, 0.015, 0.25])
    fig.colorbar(hist_plot, cax=cbar1, label="Suitability")

    # Change colorbar
    cbar2 = fig.add_axes([0.90, 0.15, 0.015, 0.25])
    fig.colorbar(diff_plot, cax=cbar2, label="Suitability Change")

    # Figure title
    plt.suptitle(
        f"{site} Douglas-Fir Habitat Suitability Across Climate Models",
        fontsize=16,
        y=0.92
    )

    # Save big comparison plot
    comparison_save_path = output_dir / f"{site}_model_comparison.png"
    fig.savefig(comparison_save_path, dpi=300, bbox_inches="tight")
    saved_files.append(comparison_save_path)

    # Display plot
    plt.show()

    # Close figure to free memory
    plt.close(fig)

    return saved_files

# %%
# Run plots for olympic
olympic_final_plot_set = plot_site_suitability_comparison(
    site="Olympic",
    maps=olympic_suitability_maps,
    models=models,
    coarsen_factor=4,
    boundary_gdf=olympic_gdf,
    gbif_gdf=None
)

# %%
# Run plots for olympic
willamette_final_plot_set = plot_site_suitability_comparison(
    site="Willamette",
    maps=willamette_suitability_maps,
    models=models,
    coarsen_factor=4,
    boundary_gdf=willamette_gdf,
    gbif_gdf=None
)

# %% [markdown]
# ### Projected Changes in Douglas-Fir Habitat Suitability Across Distinct Ecological Regions in the Pacific Northwest: Willamette and Olympic National Forests
# 
# 

# %% [markdown]
# There are changes in habitat suitability in both Willamette and Olympic National Forests for Douglas Fir under most of the climate models. In the Olympic National Forest, it appears that the models are more stable under all predictions, as the increases and decreases across the different models are similar. When compared to the suitability maps, some preferred locations across climate models increase. In some of the areas with lower habitat suitability, portions of the landscape appear to decrease in habitat suitability.
# 
# In Willamette National Forest, we do see larger variations between the different climate models in the changes. This would suggest that the outcome of changes in habitat suitability is dependent on the changing conditions under each model, heavily influenced by temperature and precipitation changes. We see increases in habitat suitability for Douglas Fir in the northern sections of Willamette under the climate model representing a cold and wet predicted scenario (IPSL-CM5A-LR). Whereas under the warm and wet scenario (CanESM2 model), we see a decrease in habitat suitability in the northern region. Douglas Fir is more sensitive to increases in temperature and moisture stress and is more tolerant to colder conditions, which fits with the habitat preferences of the species.
# 
# Further discussion in the portfolio assignment

# %% [markdown]
# <u>Citations</u>
# 
# - Klamerus-Iwan, A., Behan, P., Słowik-Opoka, E., Delgado-Moreira, M. I., & Reyna-Bowen, L. (2025). Soil hydrological properties and organic matter content in Douglas-fir and spruce stands: Implications for forest resilience to climate change. Forests, 16(8), 1217. https://doi.org/10.3390/f16081217
# 
# - National Forest Foundation. (n.d.). Olympic National Forest. Retrieved March 16, 2026, from https://www.nationalforests.org/our-forests/find-a-forest/olympic-national-forest
# 
# - National Forest Foundation. (n.d.). Willamette National Forest. Retrieved March 16, 2026, from https://www.nationalforests.org/our-forests/find-a-forest/willamette-national-forest
# 
# - U.S. Department of Agriculture, Natural Resources Conservation Service. (2002). Douglas-fir (Pseudotsuga menziesii) fact sheet. https://plants.sc.egov.usda.gov/DocumentLibrary/factsheet/pdf/fs_psme.pdf
# 
# - U.S. Forest Service. (1990, December). Douglas-fir | Silvics of North America. https://research.fs.usda.gov/silvics/douglas-fir
# 
# - U.S. National Park Service. (2025). Coast Douglas-fir. https://www.nps.gov/articles/000/douglas-fir.htm
# 
# - U.S. National Park Service. (2025, April 11). Temperate rain forests. https://www.nps.gov/olym/learn/nature/temperate-rain-forests.htm


