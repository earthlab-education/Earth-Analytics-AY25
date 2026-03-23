#!/usr/bin/env python
# coding: utf-8

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

# My species is Populus tremuloides (quaking aspen), a widespread tree species found across much of North America. It is  found in subalpine environments, especially in the western United States, where it grows in areas with moderate moisture, cooler temperatures, and well-drained soils. The geographic range of the quaking aspen spans from Alaska, through the Rocky Mountains and into parts of the southwestern United States. It's distribution is strongly influenced by elevation and local climate, like Colorado, where it is often found at mid to high elevations, where conditions are cooler and moisture is more available compared to surrounding high elevation plains. 
# 
# One conservation concern for the quaking aspen is its sensitivity to climate change, especially increasing temperatures and drought stress. In some regions, aspen decline has already been observed and often linked reduced snowpack, and changes in soil moisture. Because of this, understanding how habitat suitability may shift under future climate scenarios is important for predicting where aspen may persist or decline.
# 
# To model habitat suitability, I selected variables that represent on aspen growth and are senstive to enviromental change, including soil organic matter and soil moisture. Soil organic matter is closely tied to soil carbon, and together these reflect nutrient availability, water-holding capacity, and soil health. I also included slope and aspect for understanding drainage and solar radiation, and climate variables such as precipitation and maximum temperature, which are predicted to change based off the climate models I chose to look at.
# 
# My main scientific question guiding this study is to understand how will habitat suitability for Populus tremuloides change under future climate scenarios, and will suitable habitat shift in elevation or spatial distribution across two different landscapes?
# 
# Some Sources:
# USDA NRCS. Plant Guide for Quaking Aspen (Populus tremuloides)
# https://plants.usda.gov/DocumentLibrary/plantguide/pdf/pg_potr5.pdf
# 
# Worrall, J. J., et al. (2013). Recent declines of Populus tremuloides in North America linked to climate. Forest Ecology and Management, 299, 35–51.
# 
# Anderegg, W. R. L., et al. (2012). The roles of hydraulic and carbon stress in a widespread climate-induced forest die-off. PNAS, 109(1), 233–237.
# 

# ### Step 1b: Select study sites
# Based on your research and/or range maps you find online, select at least 2 sites where your species occurs. These could be national parks, national forests, national grasslands or other protected areas, or some other area you're interested in. You can access protected area polygons from the [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview), [national grassland units](https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip), etc.
# 
# When selecting your sites, you might want to look for places that are marginally habitable for this species, since those locations will be most likely to show changes due to climate.
# 
# Generate a site map for each location.

# **Reflect and Respond**: 
# Write a site description for each of your sites, or for all of your sites as a group if you have chosen a large number of linked sites. What
# differences or trends in habitat suitability over time do you expect to see among your sites?

# I chose my sites from the USGS Protected Areas Database (PAD-US v4.1), then filtered specifically for National Parks and selected the two sites with the highest occurrences of Populus tremuloides. I focused on National Parks because they tend to be relatively untouched in terms of land use (no timber harvesting, etc.), and they also offer comparably sized areas for analysis.
# 
# Rocky Mountain National Park was my first site chosen. The elevation has a decent range, with over 6000 feet of difference in some areas, which creates a variety of microclimates that are important for aspen growth. Nearly directly south is my second site, Great Sand Dunes National Park. This site is more between the Sangre de Cristo Mountains and the  San Luis Valley, it represents a transition between alpine and airid desert environments.
# 
# For Rocky mountaain national park, I expect Populus tremuloides to shift upslope as temperatures warm, while in Great Sand Dunes National Park, I expect an overall decline in sustainability as areas become more warm and dry
# 
# 

# In[1]:


### load libraries

### reproducible file paths
import os
from glob import glob
import pathlib
from pathlib import Path

### gbif packages
import pygbif.occurrences as occ
import pygbif.species as species
from getpass import getpass

### unzipping
import zipfile
import time

### spatial data
import geopandas as gpd
import xrspatial

### other data types
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrm

### invalid geometries
from shapely.geometry import MultiPolygon, Polygon

### visualization
import holoviews as hv
import hvplot.pandas
import hvplot.xarray

## for api
import requests


# In[2]:


# --------------------------------------------------------
# CHANGE THIS: set your species of interest
# --------------------------------------------------------
species_name = "Populus tremuloides"


# In[3]:


### file paths

## base data directory
data_dir = os.path.join(
    pathlib.Path.home(), 'earth-analytics', 'data',
    'spring-03-habitat-suitability-climate-change-hellafolk', 'hab_suit'
)
os.makedirs(data_dir, exist_ok=True)

## auto-name gbif folder after species (e.g. "gbif_populus_tremuloides")
species_folder = "gbif_" + species_name.lower().replace(" ", "_")
gbif_dir = os.path.join(data_dir, species_folder)
os.makedirs(gbif_dir, exist_ok=True)

print(f"GBIF directory: {gbif_dir}")


# In[4]:


### GBIF credentials

reset_credentials = False

credentials = dict(
    GBIF_USER=(input, 'GBIF username:'),
    GBIF_PWD=(getpass, 'GBIF password'),
    GBIF_EMAIL=(input, 'GBIF email'),
)

for env_variable, (prompt_func, prompt_text) in credentials.items():
    if reset_credentials and (env_variable in os.environ):
        os.environ.pop(env_variable)
    if not env_variable in os.environ:
        os.environ[env_variable] = prompt_func(prompt_text)


# In[5]:


### look up species key from GBIF

species_info = species.name_lookup(species_name, rank='SPECIES')
first_result = species_info['results'][0]
species_key = first_result['nubKey']

print(f"Species:     {first_result['species']}")
print(f"Species key: {species_key}")


# In[6]:


### download GBIF occurrence data (skip if already exists)

## define expected output path
gbif_path = os.path.join(gbif_dir, f"{species_folder}.csv")

if os.path.exists(gbif_path):
    print(f"File already exists, skipping download:\n  {gbif_path}")

else:
    print(f"Starting GBIF download for: {species_name}")

    ## request download
    download_key = occ.download([
        f"speciesKey = {species_key}",
        "hasCoordinate = TRUE",
        "hasGeospatialIssue = FALSE"
    ])

    ## wait for download to complete
    print("Waiting for GBIF to prepare download", end="")
    while True:
        status = occ.download_meta(download_key[0])['status']
        if status == 'SUCCEEDED':
            print(" done!")
            break
        elif status == 'FAILED':
            raise RuntimeError("GBIF download failed. Check your query.")
        print(".", end="", flush=True)
        time.sleep(5)

    ## download and unzip
    occ.download_get(download_key[0], path=gbif_dir)

    zip_path = glob(os.path.join(gbif_dir, "*.zip"))[0]
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(gbif_dir)

    ## rename extracted csv to something readable
    extracted_csv = glob(os.path.join(gbif_dir, "*.csv"))[0]
    os.rename(extracted_csv, gbif_path)
    print(f"Saved to: {gbif_path}")


# In[7]:


### load gbif data

gbif_df = pd.read_csv(gbif_path, delimiter='\t')

## quick sanity check
print(f"Rows: {len(gbif_df):,}")
print(f"Columns: {gbif_df.shape[1]}")
gbif_df.head()


# In[8]:


 ### site directory (auto-named after species like gbif folder)
site_folder = "site_" + species_name.lower().replace(" ", "_") + "_co"
site_dir = Path(data_dir) / site_folder
site_dir.mkdir(parents=True, exist_ok=True)

print(f"Site directory: {site_dir}")


# In[9]:


### download Colorado protected areas from ScienceBase

ITEM_ID = "6759abcfd34edfeb8710a004"
FILENAME = "PADUS4_1_State_CO_GDB_KMZ.zip"

url = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}?name={FILENAME}"
output_path = site_dir / FILENAME

## skip if already downloaded
if output_path.exists():
    print(f"File already exists, skipping download:\n  {output_path}")
else:
    print("Downloading Colorado protected areas...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download complete.")


# In[10]:


### unzip

zip_path = Path(output_path)
extract_folder = zip_path.parent
extract_folder.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


# In[11]:


### load and clean protected areas layer

import fiona

pa_path = extract_folder / "PADUS4_1_StateCO.gdb"

pa_shp = gpd.read_file(pa_path, layer="PADUS4_1Fee_State_CO")
pa_shp = pa_shp.to_crs(epsg=4326)

## fix invalid geometries
pa_shp['geometry'] = pa_shp['geometry'].apply(
    lambda geom: geom.make_valid() if not isinstance(geom, MultiPolygon) and not geom.is_valid else geom
)
pa_shp = pa_shp[pa_shp.geometry.is_valid]
pa_shp = pa_shp.dropna(subset=['geometry'])

print(f"Protected areas loaded: {len(pa_shp):,} features")


# In[12]:


### convert gbif dataframe to geodataframe

gbif_gdf = gpd.GeoDataFrame(
    gbif_df,
    geometry=gpd.points_from_xy(gbif_df['decimalLongitude'], gbif_df['decimalLatitude']),
    crs='epsg:4326'
)

## drop rows missing coordinates
gbif_gdf = gbif_gdf.dropna(subset=['decimalLongitude', 'decimalLatitude'])

## filter to colorado only (roughly)
gbif_gdf = gbif_gdf[
    (gbif_gdf['decimalLongitude'].between(-109.1, -102.0)) &
    (gbif_gdf['decimalLatitude'].between(36.9, 41.1))
]

print(f"GBIF points in Colorado: {len(gbif_gdf):,}")
gbif_gdf.head()


# In[13]:


### find all national parks in colorado from PAD-US

## filter to NPS units only
co_nps = pa_shp[
    pa_shp['Mang_Name'].str.contains('NPS', na=False) |
    pa_shp['Mang_Type'].str.contains('NPS', na=False)
].copy()

## intersect with GBIF occurrences to get overlap counts
species_co = gpd.overlay(gbif_gdf, pa_shp, how='intersection')
value_counts = species_co['Loc_Nm'].value_counts()

## filter to NPS sites that have species occurrences
nps_loc_names = co_nps['Loc_Nm'].unique()
nps_counts = value_counts[value_counts.index.isin(nps_loc_names)]

## print the list so you can choose
print(f"NPS sites in Colorado with {species_name} occurrences:\n")
for i, (site, count) in enumerate(nps_counts.items()):
    print(f"  [{i}] {site}  ({count:,} occurrences)")


# In[14]:


### --------------------------------------------------------
### CHANGE THESE: pick your two sites by number from the list below
### --------------------------------------------------------
SITE_1_IDX = 0
SITE_2_IDX = 1


# In[15]:


### set the two sites based on your index choices above

top_sites = [
    nps_counts.index[SITE_1_IDX],
    nps_counts.index[SITE_2_IDX]
]

## build site gdfs
site_gdfs = {
    site: pa_shp[pa_shp['Loc_Nm'] == site]
    for site in top_sites
}

print(f"\nSelected sites:")
for site in top_sites:
    print(f"  - {site}")


# In[16]:


### subset and plot each site individually

for site in top_sites:
    site_gdf = pa_shp[pa_shp['Loc_Nm'] == site]
    display(site_gdf.hvplot(
        geo=True,
        tiles='EsriImagery',
        title=site,
        fill_color=None,
        line_color="white",
        frame_width=600
    ))


# In[17]:


### combine top 2 sites and plot together

sites_gdf = gpd.GeoDataFrame(
    pd.concat(
        [pa_shp[pa_shp['Loc_Nm'] == site] for site in top_sites],
        ignore_index=True
    )
)

combined_title = " and ".join(top_sites)

sites_gdf.hvplot(
    geo=True,
    tiles='EsriImagery',
    title=combined_title,
    fill_color=None,
    line_color="white",
    frame_width=600
)


# ### Step 1c: Select time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, you will need to access climate data from 2035-2065.
# 
# **Reflect and Respond**: Select at least two 30-year time periods to compare, such as historical and 30 years into the future. These time periods should help you to answer your scientific question.

# The two times periods I am selecting are 2011-2040 and 2041-2070. These time periods are significant because I belive they are great representive of pre- and post- ai boom and datacenters, which I belive will radically change our climate in a human driven way. Also, each one of these time periods kind of feel like two halves of my life. The period I grow up in and the period I grow old in. 

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

# I selected four climate models using the Climate Futures Toolbox Future Climate Scatter tool to capture a range of possible future conditions at my sites. I chose a subset that represents the spread of temperature and precipitation outcomes by looking at the extremes (each corner) of the scatter plot. The models I selected were CNRM-CM5 (cool and wet), MIROC5 (cool and dry), IPSL-CM5A-LR (hot and dry), and HadGEM2-ES365 (hot and wet). This approach allows me to sample across the range of uncertainty in future climate conditions.
# 
# Climate Futures Toolbox (https://climatetoolbox.org)

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

# In[18]:


### --------------------------------------------------------
### CHANGE THESE: soil variables of interest
### --------------------------------------------------------

SOIL_VARS = {
    "om": {
        "depth": "15_30",       # 15-30cm captures active root zone organic matter
        "stat": "mean",
        "label": "Organic Matter (kg/kg)",
        "cmap": "YlOrBr",
        "title_name": "Soil Organic Matter (15-30cm)"
    },
    "theta_s": {
        "depth": "0_5",         # surface moisture most sensitive to drought
        "stat": "mean",
        "label": "Saturated Water Content (m³/m³)",
        "cmap": "Blues",
        "title_name": "Soil Moisture - Saturated Water Content (0-5cm)"
    }
}


# In[19]:


### soil data directory

soil_dir = Path(data_dir) / "soil"
soil_dir.mkdir(parents=True, exist_ok=True)


# In[20]:


from math import floor, ceil
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt


def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):
    """
    Generate list of POLARIS urls using site boundary.

    Args:
        soil_prop (str): soil property (soc, theta_s, ph, etc)
        stat (str): summary statistic (mean, p5, etc)
        soil_depth (str): soil depth in cm
        gdf_bounds: array of site boundaries

    Returns:
        list: a list of POLARIS urls
    """

    min_lon, min_lat, max_lon, max_lat = gdf_bounds

    site_min_lon = floor(min_lon)
    site_min_lat = floor(min_lat)
    site_max_lon = ceil(max_lon)
    site_max_lat = ceil(max_lat)

    all_soil_urls = []

    for lon in range(site_min_lon, site_max_lon):
        for lat in range(site_min_lat, site_max_lat):
            current_max_lon = lon + 1
            current_max_lat = lat + 1

            soil_url = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
                f"{soil_prop}/{stat}/{soil_depth}/"
                f"lat{lat}{current_max_lat}_lon{lon}{current_max_lon}.tif"
            )
            all_soil_urls.append(soil_url)

    return all_soil_urls


def build_da(urls, bounds):
    """
    Build a DataArray from list of POLARIS urls, skipping missing tiles.

    Args:
        urls (list): list of tile urls
        bounds (tuple): site boundaries

    Returns:
        xarray.DataArray: merged DataArray clipped to site
    """

    all_das = []
    buffer = 0.025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    for url in urls:
        try:
            tile_da = rxr.open_rasterio(url, mask_and_scale=True).squeeze()
            cropped_da = tile_da.rio.clip_box(*bounds_buffer)
            all_das.append(cropped_da)
        except Exception as e:
            print(f"  skipping tile (not available): {url.split('/')[-1]}")

    if not all_das:
        raise RuntimeError(
            f"No tiles loaded successfully. Check your soil_prop/depth combination."
        )

    return merge_arrays(all_das)

def export_raster(da, raster_path):
    """
    Export xarray.DataArray as a raster file.

    Args:
        da (xarray.DataArray): input raster
        raster_path (str): full output path

    Returns:
        None
    """

    os.makedirs(os.path.dirname(raster_path), exist_ok=True)
    da_out = da.copy()
    da_out.attrs.pop("_FillValue", None)
    if hasattr(da_out, "encoding"):
        da_out.encoding.pop("_FillValue", None)
    da_out.rio.to_raster(raster_path)


def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title,
              bar_label, plot_cmap, boundary_clr="white", tif_file=False):
    """
    Create and save a site raster plot with boundary overlay.

    Args:
        site_da (xarray.DataArray): input site raster
        site_gdf (geopandas.GeoDataFrame): site boundary
        plots_dir (str): directory to save plot
        site_fig_name (str): output filename (no extension)
        plot_title (str): plot title
        bar_label (str): colorbar label
        plot_cmap (str): colormap
        boundary_clr (str): boundary line color
        tif_file (bool): if True, open site_da as a file path first

    Returns:
        matplotlib plot
    """

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()

    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked=True)

    site_plot = site_da.plot(
        cmap=plot_cmap,
        cbar_kwargs={'label': bar_label}
    )

    site_gdf.boundary.plot(ax=plt.gca(), color=boundary_clr)
    plt.title(plot_title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(f"{plots_dir}/{site_fig_name}.png")

    return site_plot


def download_polaris(site_name, site_gdf, soil_prop, stat, soil_depth,
                     bar_label, plot_title, plot_cmap, data_dir, plots_dir):
    """
    Full POLARIS pipeline: download, merge, export raster, and save plot.

    Args:
        site_name (str): used in output filenames
        site_gdf (geopandas.GeoDataFrame): site boundary
        soil_prop (str): soil property of interest
        stat (str): summary statistic
        soil_depth (str): depth string like "15_30"
        bar_label (str): colorbar label for plot
        plot_title (str): plot title
        plot_cmap (str): colormap
        data_dir (str): folder for raster output
        plots_dir (str): folder for plot output

    Returns:
        xarray.DataArray: merged soil DataArray
    """

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    urls = create_polaris_urls(soil_prop, stat, soil_depth, site_gdf.total_bounds)
    site_soil_da = build_da(urls=urls, bounds=tuple(site_gdf.total_bounds))

    raster_path = os.path.join(data_dir, f"{site_name}_{soil_prop}_{soil_depth}.tif")
    export_raster(da=site_soil_da, raster_path=raster_path)

    plot_site(
        site_da=site_soil_da,
        site_gdf=site_gdf,
        plots_dir=plots_dir,
        site_fig_name=f"{site_name}_{soil_prop}_{soil_depth}",
        plot_title=plot_title,
        bar_label=bar_label,
        plot_cmap=plot_cmap
    )

    return site_soil_da


# In[21]:


### run POLARIS downloads for all variables and both sites
### results stored as soil_results[site_name][soil_prop]

## map site names to their gdfs (uses your auto-detected top_sites)
site_gdfs = {
    top_sites[0]: pa_shp[pa_shp['Loc_Nm'] == top_sites[0]],
    top_sites[1]: pa_shp[pa_shp['Loc_Nm'] == top_sites[1]]
}

## initialize results dictionary
soil_results = {}

for site_name, site_gdf in site_gdfs.items():

    soil_results[site_name] = {}
    site_slug = site_name.lower().replace(" ", "_")

    for soil_prop, cfg in SOIL_VARS.items():

        print(f"\nDownloading {soil_prop} for {site_name}...")

        raster_dir = soil_dir / site_slug / soil_prop / "rasters"
        plots_dir  = soil_dir / site_slug / soil_prop / "plots"

        soil_results[site_name][soil_prop] = download_polaris(
            site_name   = site_slug,
            site_gdf    = site_gdf,
            soil_prop   = soil_prop,
            stat        = cfg["stat"],
            soil_depth  = cfg["depth"],
            bar_label   = cfg["label"],
            plot_title  = f"{cfg['title_name']} — {site_name}",
            plot_cmap   = cfg["cmap"],
            data_dir    = str(raster_dir),
            plots_dir   = str(plots_dir)
        )

        print(f"  saved raster -> {raster_dir}")
        print(f"  saved plot   -> {plots_dir}")

print("\nAll done!")


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

# In[22]:


### --------------------------------------------------------
### CHANGE THESE: topographic variables of interest
### --------------------------------------------------------

TOPO_VARS = {
    "aspect": {
        "func": xrspatial.aspect,
        "cmap": "twilight",
        "label": "Aspect (degrees)",
        "title_name": "Aspect"
    },
    "slope": {
        "func": xrspatial.slope,
        "cmap": "terrain",
        "label": "Slope (degrees)",
        "title_name": "Slope Angle"
    }
}


# In[23]:


### topography directory

topo_dir = Path(data_dir) / "topography"
topo_dir.mkdir(parents=True, exist_ok=True)


# In[24]:


### log in to earthaccess

import earthaccess
earthaccess.login()


# In[25]:


### helper functions for topography

def download_srtm(site_name, site_gdf, topo_dir, buffer=0.025):
    """
    Download SRTM tiles for a site, skipping if already downloaded.

    Args:
        site_name (str): used for folder naming
        site_gdf (geopandas.GeoDataFrame): site boundary
        topo_dir (Path): base topography directory
        buffer (float): bounding box buffer in degrees

    Returns:
        str: glob pattern to find downloaded .hgt.zip files
    """

    ## make site subdir
    site_topo_dir = topo_dir / site_name
    site_topo_dir.mkdir(parents=True, exist_ok=True)

    ## build buffered bounding box
    xmin, ymin, xmax, ymax = site_gdf.total_bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    ## file pattern to check for existing downloads
    srtm_pattern = str(site_topo_dir / "*.hgt.zip")

    if not glob(srtm_pattern):
        print(f"Downloading SRTM for {site_name}...")
        results = earthaccess.search_data(
            short_name='SRTMGL3',
            bounding_box=bounds_buffer
        )
        earthaccess.download(results, str(site_topo_dir))
        print(f"  download complete")
    else:
        print(f"SRTM already downloaded for {site_name}, skipping")

    return srtm_pattern, bounds_buffer


def build_srtm_da(srtm_pattern, bounds_buffer):
    """
    Load, merge, and clip SRTM tiles into a single DataArray.

    Args:
        srtm_pattern (str): glob pattern for .hgt.zip files
        bounds_buffer (tuple): buffered bounding box

    Returns:
        xarray.DataArray: merged elevation DataArray
    """

    da_list = []

    for srtm_path in glob(srtm_pattern):
        tile_da = rxr.open_rasterio(
            srtm_path,
            mask_and_scale=True
        ).squeeze()

        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        da_list.append(cropped_da)

    if not da_list:
        raise RuntimeError(f"No SRTM tiles found at: {srtm_pattern}")

    return merge_arrays(da_list)


def compute_topo_var(elev_da, topo_func):
    """
    Compute a topographic variable from elevation, reprojects
    to EPSG:5070 for accurate calculation then back to EPSG:4326.

    Args:
        elev_da (xarray.DataArray): elevation DataArray
        topo_func (callable): xrspatial function (slope, aspect, etc)

    Returns:
        xarray.DataArray: topo variable in EPSG:4326
    """

    ## reproject to equal-area CRS for accurate calculation
    elev_rpj = elev_da.rio.reproject("EPSG:5070")

    ## compute the variable
    topo_da = topo_func(elev_rpj)

    ## reproject back to geographic CRS
    return topo_da.rio.reproject("EPSG:4326")


def download_topo(site_name, site_gdf, topo_vars, topo_dir):
    """
    Full topo pipeline: download SRTM, compute variables, save rasters and plots.

    Args:
        site_name (str): used in folder and file naming
        site_gdf (geopandas.GeoDataFrame): site boundary
        topo_vars (dict): TOPO_VARS config dict
        topo_dir (Path): base topography directory

    Returns:
        dict: {variable_name: xarray.DataArray} for each topo variable
    """

    site_slug = site_name.lower().replace(" ", "_")
    results = {}

    ## download and merge elevation tiles
    srtm_pattern, bounds_buffer = download_srtm(site_slug, site_gdf, topo_dir)
    elev_da = build_srtm_da(srtm_pattern, bounds_buffer)

    ## save raw elevation raster
    elev_raster_dir = topo_dir / site_slug / "elevation" / "rasters"
    elev_raster_dir.mkdir(parents=True, exist_ok=True)
    export_raster(elev_da, str(elev_raster_dir / f"{site_slug}_elevation.tif"))

    ## plot elevation
    elev_plots_dir = topo_dir / site_slug / "elevation" / "plots"
    elev_plots_dir.mkdir(parents=True, exist_ok=True)
    plot_site(
        site_da=elev_da,
        site_gdf=site_gdf,
        plots_dir=str(elev_plots_dir),
        site_fig_name=f"{site_slug}_elevation",
        plot_title=f"Elevation — {site_name}",
        bar_label="Elevation (m)",
        plot_cmap="terrain",
        boundary_clr="black"
    )

    ## compute, save, and plot each topo variable
    for var_name, cfg in topo_vars.items():

        print(f"  computing {var_name} for {site_name}...")

        topo_da = compute_topo_var(elev_da, cfg["func"])

        ## save raster
        raster_dir = topo_dir / site_slug / var_name / "rasters"
        raster_dir.mkdir(parents=True, exist_ok=True)
        export_raster(topo_da, str(raster_dir / f"{site_slug}_{var_name}.tif"))

        ## save plot
        plots_dir = topo_dir / site_slug / var_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_site(
            site_da=topo_da,
            site_gdf=site_gdf,
            plots_dir=str(plots_dir),
            site_fig_name=f"{site_slug}_{var_name}",
            plot_title=f"{cfg['title_name']} — {site_name}",
            bar_label=cfg["label"],
            plot_cmap=cfg["cmap"],
            boundary_clr="black"
        )

        results[var_name] = topo_da
        print(f"    saved to {raster_dir}")

    return results


# In[26]:


### run topo downloads for both sites
### results stored as topo_results[site_name][var_name]

topo_results = {}

for site_name, site_gdf in site_gdfs.items():
    print(f"\nProcessing topography for: {site_name}")

    topo_results[site_name] = download_topo(
        site_name=site_name,
        site_gdf=site_gdf,
        topo_vars=TOPO_VARS,
        topo_dir=topo_dir
    )

print("\nAll done!")


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

# I used MACAv2 data, along with the Climate Futures Toolbox Future Climate Scatter tool, to select four climate models that represent a range of possible future conditions for my sites. CNRM-CM5 was used as a cool and wet scenario, MIROC5 as cool and dry, IPSL-CM5A-LR as hot and dry, and HadGEM2-ES365 as hot and wet. Together, these models capture a broad spread of temperature and precipitation outcomes.
# 
# I used a high-emissions scenario (RCP 8.5), based on current trajectories of anthropogenic greenhouse gas emissions reported by NOAA GML. This scenario provides a more extreme, but still realistic, view of how climate conditions could shift over time.
# 
# MACAv2-METDATA dataset (Abatzoglou & Brown, 2012), available via https://www.climatologylab.org/maca.html

# I used the MACAv2 data, along with the Climate Futures Toolbox Future Climate Scatter tool, to select four climate models that represent a range of possible future conditions for my sites. CNRM-CM5 was used as a cool and wet scenario, MIROC5 as cool and dry, IPSL-CM5A-LR as hot and dry, and HadGEM2-ES365 as hot and wet. Together, these models capture a broad spread of temperature and precipitation outcomes. I used a high-emissions scenario (RCP 8.5), based on current trajectories of anthropogenic greenhouse gas emissions reported by NOAA GML. This scenario provides a more extreme, but still realistic, view of how climate conditions could shift over time.

# In[27]:


### --------------------------------------------------------
### CHANGE THIS: which RCP to use for climate rasters
### --------------------------------------------------------
MACA_RCPS = ["rcp85"]   


# In[28]:


### --------------------------------------------------------
### CHANGE THESE: climate model parameters
### --------------------------------------------------------

MACA_VARS = {
    "pr": {
        "var_name": "precipitation",
        "label": "Precipitation (mm)",
        "cmap": "Blues",
        "title_name": "Precipitation"
    },
    "tasmax": {
        "var_name": "air_temperature",
        "label": "Max Temperature (°F)",
        "cmap": "RdYlBu_r",
        "title_name": "Max Temperature"
    }
}

MACA_MODELS = [
    "MIROC5",        # warm & dry
    "CNRM-CM5",       # warm & wet
    "HadGEM2-CC365",  # hot & dry
    "IPSL-CM5A-LR"    # moderate
]



### two 30-year periods as (start_year, end_year)
MACA_PERIODS = [
    (2011, 2040),
    (2041, 2070)
]


# In[29]:


### climate data directory

maca_dir = Path(data_dir) / "maca"
maca_dir.mkdir(parents=True, exist_ok=True)


# In[30]:


### helper functions for MACA

import xarray as xr


def convert_temperature(temp):
    """Convert Kelvin to Fahrenheit."""
    return temp * 1.8 - 459.67


def convert_longitude(longitude):
    """Convert 0-360 longitude to -180-180."""
    return (longitude - 360) if longitude > 180 else longitude


def make_5yr_chunks(start_year, end_year):
    """
    Split a year range into 5-year chunk strings for MACA URLs.

    Args:
        start_year (int): first year of period
        end_year (int): last year of period

    Returns:
        list of str: e.g. ['2011_2015', '2016_2020', ...]
    """
    chunks = []
    for yr in range(start_year, end_year + 1, 5):
        chunks.append(f"{yr}_{yr + 4}")
    return chunks


def build_maca_url(model, maca_var, rcp, date_range):
    """
    Build a MACA thredds URL for one variable/model/rcp/date chunk.

    Args:
        model (str): GCM model name
        maca_var (str): MACA variable code (pr, tasmax, etc)
        rcp (str): emissions scenario (rcp45, rcp85)
        date_range (str): date chunk string like '2041_2045'

    Returns:
        str: full MACA URL
    """
    return (
        "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
        f"/MACAV2/{model}"
        f"/macav2metdata_{maca_var}_{model}_r1i1p1"
        f"_{rcp}_{date_range}_CONUS_monthly.nc"
    )


def download_maca_chunk(model, maca_var, var_name, rcp,
                        date_range, site_name, maca_dir, site_gdf, buffer=0.025):
    """
    Download one MACA chunk, clip to site bounds, and save locally.
    Skip if exists and valid.

    Args:
        model (str): GCM model name
        maca_var (str): MACA variable code
        var_name (str): variable name inside the netCDF file
        rcp (str): emissions scenario
        date_range (str): 5-year chunk string
        site_name (str): used in filename
        maca_dir (Path): directory to save files
        site_gdf (geopandas.GeoDataFrame): site boundary for clipping
        buffer (float): bounding box buffer in degrees

    Returns:
        Path: local file path
    """
    filename = f"maca_{model}_{site_name}_{maca_var}_{rcp}_{date_range}.nc"
    local_path = maca_dir / filename
    tmp_path = maca_dir / f"{filename}.tmp"

    ## get site bounds with buffer
    xmin, ymin, xmax, ymax = site_gdf.total_bounds
    bounds_buffer = (xmin - buffer, ymin - buffer,
                     xmax + buffer, ymax + buffer)

    ## check if file exists AND is valid
    file_is_valid = False
    if local_path.exists():
        try:
            with xr.open_dataset(local_path) as ds:
                _ = ds[var_name].load()
            file_is_valid = True
        except Exception:
            print(f"  corrupted file detected, re-downloading: {filename}")
            local_path.unlink()

    if not file_is_valid:
        url = build_maca_url(model, maca_var, rcp, date_range)

        try:
            with xr.open_dataset(url) as ds:
                da = ds[var_name].squeeze()

                ## fix longitudes before clipping
                da = da.assign_coords(
                    lon=("lon", [convert_longitude(l) for l in da.lon.values])
                )
                da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                da = da.rio.write_crs("EPSG:4326")

                ## clip to site before loading into memory
                da = da.rio.clip_box(*bounds_buffer)

                ## now load clipped data into memory
                da = da.load()

        except Exception as e:
            raise RuntimeError(f"Failed opening/loading remote dataset: {url}\n{e}")

        try:
            da.to_netcdf(tmp_path)
            tmp_path.replace(local_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed writing local NetCDF: {local_path}\n{e}")

    return local_path


def process_maca_chunk(local_path, var_name, site_gdf, convert_temp=False):
    """
    Open, reproject, fix longitudes, and clip one MACA chunk to site.

    Args:
        local_path (Path): path to local .nc file
        var_name (str): variable name inside the netCDF
        site_gdf (geopandas.GeoDataFrame): site boundary
        convert_temp (bool): if True, convert Kelvin to Fahrenheit

    Returns:
        xarray.DataArray: cropped DataArray for the site
    """
    da = xr.open_dataset(local_path).squeeze()[var_name]

    ## fix longitudes
    da = da.assign_coords(
        lon=("lon", [convert_longitude(l) for l in da.lon.values])
    )

    ## set spatial dims
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    ## write CRS explicitly since MACA files don't include it
    da = da.rio.write_crs("EPSG:4326")

    ## reproject site boundary to match
    site_rpj = site_gdf.to_crs("EPSG:4326")
    bounds = site_rpj.total_bounds

    ## crop to site
    da_cropped = da.rio.clip_box(*bounds)

    ## convert temperature if needed
    if convert_temp:
        da_cropped = convert_temperature(da_cropped)

    return da_cropped


# In[31]:


### run MACA downloads and processing for all combinations
### results stored as:
### maca_results[site_name][model][rcp][period_str][maca_var]

## calculate total combinations for progress tracking
total = (len(site_gdfs) * len(MACA_MODELS) * len(MACA_RCPS) *
         len(MACA_PERIODS) * len(MACA_VARS))
total_chunks = total * 6  ## 6 five-year chunks per combination
current = 0

maca_results = {}

for site_name, site_gdf in site_gdfs.items():

    site_slug = site_name.lower().replace(" ", "_")
    maca_results[site_name] = {}

    site_maca_dir = maca_dir / site_slug
    site_maca_dir.mkdir(parents=True, exist_ok=True)

    for model in MACA_MODELS:
        maca_results[site_name][model] = {}

        for rcp in MACA_RCPS:
            maca_results[site_name][model][rcp] = {}

            for start_yr, end_yr in MACA_PERIODS:

                period_str = f"{start_yr}_{end_yr}"
                chunks = make_5yr_chunks(start_yr, end_yr)
                maca_results[site_name][model][rcp][period_str] = {}

                for maca_var, cfg in MACA_VARS.items():

                    chunk_das = []

                    for date_range in chunks:
                        current += 1
                        print(
                            f"[{current}/{total_chunks}] "
                            f"{site_name} | {model} | {rcp} | "
                            f"{date_range} | {maca_var}",
                            end="... "
                        )

                        local_path = download_maca_chunk(
                            model=model,
                            maca_var=maca_var,
                            var_name=cfg["var_name"],
                            rcp=rcp,
                            date_range=date_range,
                            site_name=site_slug,
                            maca_dir=site_maca_dir,
                            site_gdf=site_gdf
                        )

                        da_chunk = process_maca_chunk(
                            local_path=local_path,
                            var_name=cfg["var_name"],
                            site_gdf=site_gdf,
                            convert_temp=(maca_var == "tasmax")
                        )
                        chunk_das.append(da_chunk)
                        print("✓")

                    ## concatenate all chunks into full 30-year DataArray
                    full_da = xr.concat(chunk_das, dim="time")
                    maca_results[site_name][model][rcp][period_str][maca_var] = full_da

print("\nAll MACA downloads complete!")


# In[32]:


import shutil
total, used, free = shutil.disk_usage(data_dir)
print(f"Free: {free / (1024**3):.1f} GB")
print(f"Used: {used / (1024**3):.1f} GB")
print(f"Total: {total / (1024**3):.1f} GB")


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

# In[33]:


### --------------------------------------------------------
### CHANGE THIS: which RCP to use for climate rasters
### --------------------------------------------------------
MACA_RCP = "rcp85"


# In[34]:


### helper functions for harmonization

def get_template_da(site_name, topo_results):
    """
    Get the highest resolution layer to use as reprojection template.

    Args:
        site_name (str): site name
        topo_results (dict): topo_results dictionary

    Returns:
        xarray.DataArray: template DataArray
    """
    site_slug = site_name.lower().replace(" ", "_")
    template_path = (
        topo_dir / site_slug / "elevation" / "rasters"
        / f"{site_slug}_elevation.tif"
    )
    return rxr.open_rasterio(template_path, masked=True).squeeze()


def average_maca_period_by_model(maca_results, site_name, model, rcp, period_str):
    """
    Average one model/RCP combo over time for a given period.

    Args:
        maca_results (dict): full maca_results dictionary
        site_name (str): site name
        model (str): GCM model name
        rcp (str): emissions scenario
        period_str (str): period string like '2011_2040'

    Returns:
        dict: {maca_var: time-averaged DataArray}
    """
    result = {}

    for maca_var in MACA_VARS:
        da = maca_results[site_name][model][rcp][period_str][maca_var]

        ## average over time
        mean_da = da.mean(dim="time")

        ## write CRS back in since mean() strips it
        mean_da = mean_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        mean_da = mean_da.rio.write_crs("EPSG:4326")

        result[maca_var] = mean_da

    return result


def harmonize_site(site_name, site_gdf, soil_results, topo_results,
                   maca_results, template_da, rcp, buffer=0.025):
    """
    Harmonize all data layers for one site to a common grid.
    Keeps models separate — returns harmonized layers for each
    model and 30-year period.

    Args:
        site_name (str): site name
        site_gdf (geopandas.GeoDataFrame): site boundary
        soil_results (dict): soil_results dictionary
        topo_results (dict): topo_results dictionary
        maca_results (dict): maca_results dictionary
        template_da (xarray.DataArray): reprojection template
        rcp (str): emissions scenario to use
        buffer (float): bounding box buffer in degrees

    Returns:
        dict: {model: {period_str: {layer_name: harmonized DataArray}}}
    """

    ## build buffered bounding box
    xmin, ymin, xmax, ymax = site_gdf.total_bounds
    bounds_buffer = (xmin - buffer, ymin - buffer,
                     xmax + buffer, ymax + buffer)

    ## harmonize static layers once (same for all models/periods)
    print(f"  harmonizing static layers for {site_name}...")
    harmonized_static = {}

    for soil_var in soil_results[site_name]:
        da = soil_results[site_name][soil_var]
        cropped = da.rio.clip_box(*bounds_buffer)
        reproj = cropped.rio.reproject_match(template_da)
        if reproj.ndim == 3:
            reproj = reproj.squeeze()
        harmonized_static[soil_var] = reproj
        print(f"    ✓ {soil_var}")

    for topo_var in topo_results[site_name]:
        da = topo_results[site_name][topo_var]
        cropped = da.rio.clip_box(*bounds_buffer)
        reproj = cropped.rio.reproject_match(template_da)
        if reproj.ndim == 3:
            reproj = reproj.squeeze()
        harmonized_static[topo_var] = reproj
        print(f"    ✓ {topo_var}")

    ## harmonize MACA per model per period
    model_results = {}

    for model in MACA_MODELS:
        model_results[model] = {}

        for start_yr, end_yr in MACA_PERIODS:
            period_str = f"{start_yr}_{end_yr}"
            print(f"  harmonizing MACA | {model} | {period_str}...")

            maca_means = average_maca_period_by_model(
                maca_results, site_name, model, rcp, period_str
            )

            harmonized_maca = {}
            for maca_var, da in maca_means.items():
                cropped = da.rio.clip_box(*bounds_buffer)
                reproj = cropped.rio.reproject_match(template_da)
                if reproj.ndim == 3:
                    reproj = reproj.squeeze()
                harmonized_maca[maca_var] = reproj
                print(f"    ✓ {maca_var}")

            ## combine static + climate for this model/period
            model_results[model][period_str] = {
                **harmonized_static,
                **harmonized_maca
            }

    return model_results


# In[ ]:





# In[35]:


### run harmonization for both sites
### results stored as harmonized[site_name][model][period_str][layer_name]

harmonized = {}

for site_name, site_gdf in site_gdfs.items():
    print(f"\nHarmonizing: {site_name}")

    template_da = get_template_da(site_name, topo_results)

    harmonized[site_name] = harmonize_site(
        site_name=site_name,
        site_gdf=site_gdf,
        soil_results=soil_results,
        topo_results=topo_results,
        maca_results=maca_results,
        template_da=template_da,
        rcp=MACA_RCP       
    )

print("\nHarmonization complete!")


# In[36]:


### colormap config for each layer
LAYER_CMAPS = {
    "om":      {"cmap": "YlOrBr", "label": "Organic Matter (kg/kg)"},
    "theta_s": {"cmap": "Blues",  "label": "Soil Moisture (m³/m³)"},
    "slope":   {"cmap": "terrain","label": "Slope (degrees)"},
    "aspect":  {"cmap": "twilight","label": "Aspect (degrees)"},
    "pr":      {"cmap": "Blues",  "label": "Precipitation (mm)"},
    "tasmax":  {"cmap": "RdYlBu_r","label": "Max Temp (°F)"},
}


# In[37]:


### plot all harmonized layers for both sites, all models, both periods

for site_name, site_gdf in site_gdfs.items():
    for model, periods in harmonized[site_name].items():
        for period_str, layers in periods.items():

            n_layers = len(layers)
            fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))

            if n_layers == 1:
                axes = [axes]

            fig.suptitle(
                f"{site_name} — {model} — {period_str}",
                fontsize=14, y=1.02
            )

            for ax, (layer_name, da) in zip(axes, layers.items()):
                if da.ndim == 3:
                    da = da.squeeze()

                cfg = LAYER_CMAPS.get(
                    layer_name,
                    {"cmap": "viridis", "label": layer_name}
                )

                da.plot(
                    ax=ax,
                    cmap=cfg["cmap"],
                    add_colorbar=True,
                    cbar_kwargs={"label": cfg["label"], "shrink": 0.8}
                )

                site_gdf.boundary.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=1
                )

                ax.set_title(layer_name)
                ax.set_aspect('equal')
                ax.set_axis_off()

            plt.tight_layout()
            plt.show()


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

# In[39]:


FUZZY_PARAMS = {
    "om": {
        "optimal": 0.5,      
        "tolerance": 1.0,
    },
    "theta_s": {
        "optimal": 0.55,     
        "tolerance": 0.10,
    },
    "slope": {
        "optimal": 15.0,     
        "tolerance": 12.0,
    },
    "aspect": {
        "optimal": 0.0,      ## north facing
        "tolerance": 120.0,  
    },
    "pr": {
        "optimal": 50.0,     ## monthly mm (600mm/yr ÷ 12)
        "tolerance": 20.0,   ## ±20mm/month
    },
    "tasmax": {
        "optimal": 50.0,     ## adjusted to match actual ROMO temps
        "tolerance": 10.0,   ## wider tolerance
    }
}


# In[41]:


### fuzzy logic functions

def gaussian_suitability(da, optimal, tolerance):
    """
    Apply Gaussian fuzzy membership function to a DataArray.
    Returns suitability score between 0 and 1.

    Args:
        da (xarray.DataArray): input raster layer
        optimal (float): optimal value for the species
        tolerance (float): tolerance range — controls bell curve width

    Returns:
        xarray.DataArray: suitability scores 0-1
    """
    difference = da - optimal
    squared = difference ** 2
    scaled = squared / (2 * tolerance ** 2)
    return np.exp(-scaled)


def compute_habitat_suitability(layers, fuzzy_params):
    """
    Compute combined habitat suitability from all layers
    using Gaussian fuzzy logic. Layers are averaged together
    so the result reflects overall suitability across all conditions.

    Args:
        layers (dict): {layer_name: xarray.DataArray}
        fuzzy_params (dict): FUZZY_PARAMS config dict

    Returns:
        xarray.DataArray: combined suitability map (0-1)
    """
    suitability_layers = []

    for layer_name, da in layers.items():
        if layer_name not in fuzzy_params:
            print(f"  skipping {layer_name} — not in FUZZY_PARAMS")
            continue

        params = fuzzy_params[layer_name]
        suit = gaussian_suitability(
            da,
            optimal=params["optimal"],
            tolerance=params["tolerance"]
        )
        suit.name = layer_name
        suitability_layers.append(suit)
        print(f"  ✓ {layer_name}")

    ## average all layers instead of multiplying
    combined = sum(suitability_layers) / len(suitability_layers)

    return combined


# In[42]:


### check mean suitability per layer at ROMO

test_layers = harmonized['ROMO']['CNRM-CM5']['2011_2040']

print(f"{'layer':10s} | {'data min':>10} | {'data mean':>10} | {'data max':>10} | {'suit mean':>10}")
print("-" * 60)

for layer_name, da in test_layers.items():
    if layer_name in FUZZY_PARAMS:
        params = FUZZY_PARAMS[layer_name]
        suit = gaussian_suitability(da, params["optimal"], params["tolerance"])
        print(
            f"{layer_name:10s} | "
            f"{float(da.min()):10.2f} | "
            f"{float(da.mean()):10.2f} | "
            f"{float(da.max()):10.2f} | "
            f"{float(suit.mean()):10.3f}"
        )


# In[43]:


### compute habitat suitability for all sites, models, and periods
### results stored as suitability_results[site_name][model][period_str]

suitability_dir = Path(data_dir) / "suitability"
suitability_dir.mkdir(parents=True, exist_ok=True)

suitability_results = {}

for site_name, site_gdf in site_gdfs.items():

    site_slug = site_name.lower().replace(" ", "_")
    suitability_results[site_name] = {}

    for model, periods in harmonized[site_name].items():

        suitability_results[site_name][model] = {}

        for period_str, layers in periods.items():

            print(f"\n{site_name} | {model} | {period_str}")

            ## compute suitability
            combined = compute_habitat_suitability(layers, FUZZY_PARAMS)

            ## save raster
            out_path = (
                suitability_dir
                / f"{site_slug}_{model}_{period_str}_suitability.tif"
            )
            combined.rio.to_raster(str(out_path))

            ## store result
            suitability_results[site_name][model][period_str] = combined
            print(f"  saved -> {out_path.name}")

print("\nAll suitability maps complete!")


# ## STEP 5: Present your results
# Generate some plots that show your key findings of habitat suitability in your study sites across the different time periods and climate models. Don’t forget to interpret your plots!

# 
# These plots show modeled habitat suitability for Populus tremuloides across my time periods using multiple climate models and a Gaussian fuzzy logic approach. Each environmental variable I selected to investigate (soil organic matter, soil moisture, slope, aspect, precipitation, and temperature) is converted into a continuous suitability score based on how close conditions are to an optimal value. The tolerance parameter controls how quickly suitability decreases from that optimal value. The final suitability map is computed by averaging across all variables, so that no single factor dominates the result.
# 
# Initially, it was difficult to interpret the differences in habitat suitability between my two time periods for each of my sites. Both sites have microclimates, with finely detailed areas that are very habitable, such as what appear to be drainages, and less habitable areas, such as where the slope angle is too steep. To better understand how habitat suitability is changing over time, I created a difference map by subtracting the earlier period (2011–2040) from the future period (2041–2070). These difference plots made it much clearer where suitability is increasing or decreasing. Positive values (blue) indicate areas becoming more suitable, which dominate my results.
# 
# Finding the right optimal and tolerance values took some tuning. At first, my precipitation data was in monthly units (mm/month) instead of annual totals, which caused the precipitation suitability scores to drop to almost zero across the entire study area. Once I adjusted the optimal value to match the monthly scale (around 50 mm/month), and widened the temperature tolerance to better reflect the cooler, high-elevation conditions in Colorado, the results started to look much more reasonable.
# 
# Habitability for my species shows mixed changes over time. In Rocky Mountain National Park, suitability increases in higher elevation regions, but decreases along the edges of the park, especially at lower elevations. This suggests that temperature is likely driving these patterns, with warming pushing conditions closer to optimal in cooler, high-elevation areas, while making already warmer, lower-elevation areas less suitable. In Great Sand Dunes National Park, habitability tends to decrease across most of the site, with only small areas showing slight increases. This suggests that this hability is more sensitive to drying or warming, and that conditions are moving further away from what is optimal for my species, especially in lower elevation and more exposed areas.
# 
# Across the models, CNRM-CM5 (cool and wet model) generally predicts the most favorable conditions, showing more areas of increased suitability, while HadGEM2-CC365 ( hot and dry model) shows the strongest decreases in suitability. 

# In[44]:


### plot suitability maps — one row per site, columns = models
### two figures total (one per period) for easy comparison

for start_yr, end_yr in MACA_PERIODS:
    period_str = f"{start_yr}_{end_yr}"

    n_sites = len(site_gdfs)
    n_models = len(MACA_MODELS)

    fig, axes = plt.subplots(
        n_sites, n_models,
        figsize=(5 * n_models, 5 * n_sites)
    )

    fig.suptitle(f"Habitat Suitability — {period_str}", fontsize=16, y=1.02)

    for row, (site_name, site_gdf) in enumerate(site_gdfs.items()):
        for col, model in enumerate(MACA_MODELS):

            ax = axes[row, col]
            da = suitability_results[site_name][model][period_str]

            if da.ndim == 3:
                da = da.squeeze()

            da.plot(
                ax=ax,
                cmap="RdYlGn",
                vmin=0, vmax=1,
                add_colorbar=True,
                cbar_kwargs={"label": "Suitability (0-1)", "shrink": 0.8}
            )

            site_gdf.boundary.plot(
                ax=ax,
                facecolor='none',
                edgecolor='black',
                linewidth=1
            )

            ax.set_title(f"{site_name}\n{model}")
            ax.set_aspect('equal')
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# In[45]:


### plot suitability difference between periods (2041_2070 minus 2011_2040)
### positive = gained suitability, negative = lost suitability

fig, axes = plt.subplots(
    n_sites, n_models,
    figsize=(5 * n_models, 5 * n_sites)
)

fig.suptitle(
    "Change in Habitat Suitability\n(2041–2070 minus 2011–2040)",
    fontsize=16, y=1.02
)

for row, (site_name, site_gdf) in enumerate(site_gdfs.items()):
    for col, model in enumerate(MACA_MODELS):

        ax = axes[row, col]

        early = suitability_results[site_name][model]['2011_2040']
        late  = suitability_results[site_name][model]['2041_2070']

        if early.ndim == 3:
            early = early.squeeze()
        if late.ndim == 3:
            late = late.squeeze()

        diff = late - early

        diff.plot(
            ax=ax,
            cmap="RdBu",        ## red = lost suitability, blue = gained
            vmin=-0.3, vmax=0.3,
            add_colorbar=True,
            cbar_kwargs={"label": "Suitability Change", "shrink": 0.8}
        )

        site_gdf.boundary.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=1
        )

        ax.set_title(f"{site_name}\n{model}")
        ax.set_aspect('equal')
        ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[46]:


### clip all existing MACA files to their site bounding box and re-save
### this will dramatically reduce file sizes for github upload

from tqdm.notebook import tqdm

buffer = 0.025

for site_name, site_gdf in site_gdfs.items():

    site_slug = site_name.lower().replace(" ", "_")
    site_maca_dir = maca_dir / site_slug

    ## get site bounds with buffer
    xmin, ymin, xmax, ymax = site_gdf.total_bounds
    bounds_buffer = (xmin - buffer, ymin - buffer,
                     xmax + buffer, ymax + buffer)

    nc_files = list(site_maca_dir.glob("*.nc"))
    print(f"\n{site_name}: {len(nc_files)} files to clip")

    for nc_path in tqdm(nc_files):
        try:
            ## open file
            ds = xr.open_dataset(nc_path)
            var_name = list(ds.data_vars)[0]
            da = ds[var_name]

            ## fix longitudes
            da = da.assign_coords(
                lon=("lon", [convert_longitude(l) for l in da.lon.values])
            )
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            da = da.rio.write_crs("EPSG:4326")

            ## clip to site bounds
            da_clipped = da.rio.clip_box(*bounds_buffer)

            ## save back to same path via temp file
            tmp_path = nc_path.with_suffix(".tmp.nc")
            da_clipped.to_netcdf(tmp_path)
            ds.close()
            tmp_path.replace(nc_path)

        except Exception as e:
            print(f"  skipping {nc_path.name}: {e}")

print("\nAll MACA files clipped!")

