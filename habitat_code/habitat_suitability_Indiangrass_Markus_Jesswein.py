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
# ## STEP 0: Packages

# %%
### reprodecable file paths
import os
from glob import glob
import pathlib
from pathlib import Path

### gbif packages
import pygbif.occurrences as occ
import pygbif.species as species 
from getpass import getpass 

### unzipping files
import zipfile
import time 

### spatial data handling
import geopandas as gpd
import xrspatial

### data types
import numpy as np
import pandas as pd 
import rioxarray as rxr 
import rioxarray.merge as rxrm

### imnvalid geometry handling
from shapely.geometry import MultiPolygon, Polygon
import hvplot.pandas
import hvplot.xarray

### for api use 
import requests

import fiona
from math import floor, ceil   
import matplotlib.pyplot as plt

import earthaccess

from tqdm.notebook import tqdm
import xarray as xr

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
# Your response here:
# 
# I have decided on the above suggestion and am focusing on Sorghastrum nutans which is also called Indiangrass or Yellow Indiangrass. 
# 
# Sorghastrum nutans is found from Quebec and Maine west to central Saskatchewan, south to Arizona and northern Mexico, and east to Florida. It is found in all but 5 of the 48 continental states.
# it commonly inhabits prairies, open woodlands, savannas, and roadsides. It thrives in well-drained soils but can tolerate a range of soil textures, from sandy to clay loams.
# 
# Indiangrass typically grows between 3 to 8 feet (0.9–2.4 meters) tall, with flowering stalks often reaching the upper end of that range.
# Its altitude (elevation) range in the United States extends from sea level up to approximately 7,000 feet (about 2,100 meters), depending on regional conditions. The vegetation most frequently occurs on south- to southwest-facing slopes but may occur on other aspects as well.
# 
# Sorghastrum nutans tolerates a soil pH range of approximately 5 to 7.8, preferring slightly acidic to neutral soils but showing adaptability to mildly alkaline conditions.
# In additon, it tolerates precipitation ranges from approcimately 12 to 40 inches of annual rainfall and its root depth (minimum) is 24 inches. 
# 
# Core scientic question: Indiangrass appears to be a very resilient plant in itself. However, the question arises as to how far its spread will be affected by changes in temperature precipitation due to climate change.
# I would like to compare two sites that may respond differently to climate change.
# 
# References: 
# 
# Walkup, Crystal J. 1991. Sorgastrum nutans. In: Fire Effects Information System, [Online]. 
# U.S. Department of Agriculture, Forest Service, Rocky Mountain Research Station, 
# Fire Sciences Laboratory (Producer). Available: 
# https://www.fs.usda.gov/database/feis/plants/graminoid/sornut/all.html [2026, March 3]
# 
# https://plants.usda.gov/plant-profile/SONU2/characteristics
# 
# https://extension.usu.edu/rangeplants/grasses-and-grasslikes/indiangrass
# 
# https://www1.usgs.gov/csas/nvcs/unitDetails/689138
# 
# 

# %% [markdown]
# ### File paths settings
# 
# We first have a look at the Global Biodiversity Information Facility site

# %%
### file paths 
data_dir = os.path.join(
    pathlib.Path.home(), 
    'earth-analytics',
    'data',
    'habitat_suit'                    
)
os.makedirs(data_dir, exist_ok=True)

# %%
### gbif data dir
gbif_dir = os.path.join(
    data_dir,
    'gbif_indiangrass'
)

# %%
### gbif login credentials
reset_credentials = False

### make dictionary for gbif credentials
credentials = dict(
    
    GBIF_USER = (input, 'GBIF username'),
    GBIF_PWD = (getpass, 'GBIF password'),
    GBIF_EMAIL = (input, 'GBIF email'),
)

### loop through credentials and get user input
for env_variable, (prompt_func, prompt_text) in credentials.items():
    if reset_credentials and (env_variable in os.environ):
        os.environ.pop(env_variable)
    if not env_variable in os.environ:
        os.environ[env_variable] = prompt_func(prompt_text)

# %%
### species name 
species_name = 'Sorghastrum nutans'

### species infos from gbif 
species_info_1 = species.name_backbone(name=species_name)
species_info_2 = species.name_lookup(name=species_name, rank='species')

# %%
species_info_1

# %% [markdown]
# For some reason, the command "species.name_lookup(name=species_name, rank='species')" dies not lead to the right species (with the same species name), but always to Caldisphaera lagunensis.
# I will rely on "species.name_backbone(name=species_name)"

# %%
species_info_2['results'][0]

# %%
species_key = species_info_1['speciesKey']
species_info_1['species'], species_key

# %%
### assigne species code 
species_key = 2704414

# %%
### make a filepath
gbif_pattern = os.path.join(
    gbif_dir,
    '*.csv'
)

### download it once
if not glob(gbif_pattern):

    ### sumit a download request to the gbif api
    gbif_query = occ.download([
            f"speciesKey = {species_key}",
            "hasCoordinate = True",
        ])

    ### only dowload once 
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
        download_key = os.environ['GBIF_DOWNLOAD_KEY']

        ### wait for the download to build 
        wait = occ.download_meta(download_key)['status']
        while not wait == 'SUCCEEDED':
            wait = occ.download_meta(download_key)['status']
            time.sleep(5)

    ### dowloand data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'], 
        path=data_dir
    )

    ### unizip the downloaded file
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path = gbif_dir)   


# %%
### find csv file in the gbif dir
gbif_path = glob(gbif_pattern)[0]
gbif_path

# %%
# read in the csv file as a pandas dataframe
gbif_df = pd.read_csv(gbif_path,
                      delimiter='\t'
                      )

gbif_df.head()

# %%
### make it geopandas dataframe
gbif_gdf = (
    gpd.GeoDataFrame(gbif_df, 
                            geometry=gpd.points_from_xy(
                                gbif_df.decimalLongitude, 
                                gbif_df.decimalLatitude),
                            crs='EPSG:4326'
                            )
                            )

gbif_gdf

# %%
### plit it 
gbif_gdf.hvplot(
    #x='decimalLongitude', 
    #y='decimalLatitude', 
    geo=True, 
    tiles='EsriImagery',
    title='Sorghastrum nutans occurrences in GBIF',
    line_color='yellow',
    frame_width=500,
    frame_height=500
)

# %% [markdown]
# The occurencence data in GBIF confirms the distribution given in the literature of Indiangrass.
# 

# %% [markdown]
# ### Step 1b: Select study sites
# Based on your research and/or range maps you find online, select at least 2 sites where your species occurs. These could be national parks, national forests, national grasslands or other protected areas, or some other area you're interested in. You can access protected area polygons from the [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview), [national grassland units](https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip), etc.
# 
# When selecting your sites, you might want to look for places that are marginally habitable for this species, since those locations will be most likely to show changes due to climate.
# 
# Generate a site map for each location.

# %%
class SciencebaseSite():
    """
    A class to handle the downloading, unzipping, and processing of government layers from ScienceBase for a specific site."""

    def __init__(self, site_dir_name ,filename):
        """
        Initialize the SciencebaseSite instance.
        site_dir_name: The name of the directory for the site (e.g., "indiangrass_CO").
        filename: The name of the file to download (e.g., "PADUS4_1_State_CO_GDB_KMZ.zip").
        """

        self.site_name = site_dir_name[-2:]  # Extract last two characters (e.g., "CO")
        self.item_id = "6759abcfd34edfeb8710a004"
        self.filename = filename

        # Create directory per instance (safer)
        self.site_dir = Path(data_dir) / site_dir_name # "indiangrass_sites"
        self.site_dir.mkdir(parents=True, exist_ok=True)

        self.url = (
            f"https://www.sciencebase.gov/catalog/file/get/"
            f"{self.item_id}?name={self.filename}"
        )
        # this will be the path to the downloaded file, set after download
        self.file_output_path = None

        ### thse methods are executed in the initialization, but you can call them separately if needed
        self.download()
        self.unzip()


    def download(self):
        """
        Download the file from ScienceBase and save it to the site directory.
        """
        response = requests.get(self.url)
        if response.status_code == 200:
            self.file_output_path = self.site_dir / self.filename
            with open(self.file_output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {self.filename} to {self.file_output_path}")

        else:
            print(f"Failed to download {self.filename}. Status code: {response.status_code}")
    

    def unzip(self):
        """
        Unzip the downloaded file if it's a zip archive.
        """
        ### place for unzipped file is the same as the zipped file 
        zip_path = Path(self.file_output_path)

        extract_folder = zip_path.parent
        extract_folder.mkdir(parents=True, exist_ok=True)

        ### unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Unzipped {self.filename} to {extract_folder}")


    def get_gov_layers(self):
        """
        Get the FEE layers
        """
        ### path to the gdb file
        pa_path = self.site_dir / f"PADUS4_1_State{self.site_name}.gdb"

        ### read the fee layer
        self.pa_shp = gpd.read_file(pa_path, layer= f"PADUS4_1Fee_State_{self.site_name}")

        ### change the crs to match the gbif data
        self.pa_shp = self.pa_shp.to_crs('EPSG:4326')

        ### fix invalid geometries
        self.pa_shp['geometry'] = self.pa_shp['geometry'].apply(
            lambda geom: geom.make_valid() if not isinstance(geom, MultiPolygon) and not geom.is_valid else geom
        )
        ### drop remaining invalid geometries
        self.pa_shp = self.pa_shp[self.pa_shp.geometry.is_valid]

        ### missing geometries
        self.pa_shp = self.pa_shp.dropna(subset=['geometry'])

        print(f"Loaded and processed government layers for {self.site_name}")


    def get_intersect_with_gbif(self, gbif_gdf):
        """
        Get the intersection of the government layers with the GBIF occurrences.
        """
        if hasattr(self, 'pa_shp'):
            self.gbif_intersection = gpd.overlay(gbif_gdf, self.pa_shp, how='intersection')
            self.gbif_intersection_count = self.gbif_intersection['Loc_Nm'].value_counts()
            print(f"Calculated intersection of government layers with GBIF occurrences for {self.site_name}")
        else:
            print("Government layers not loaded. Please run get_gov_layers() first.")


    def plot_gov_layers(self):
        """
        Plot the government layers using hvplot.
        """
        if hasattr(self, 'pa_shp'):
            self.pa_shp.hvplot(
                geo=True, 
                tiles='EsriImagery',
                title=f'Government Layers for {self.site_name}',
                line_color='blue',
                frame_width=500,
                frame_height=500
            )
        else:
            print("Government layers not loaded. Please run get_gov_layers() first.")

# %% [markdown]
# ### Fist site in Wisconsin

# %%
### create instances for both sites, starting with wisconsin
wisconsin_IG = SciencebaseSite(site_dir_name='indiangrass_WI', filename="PADUS4_1_State_WI_GDB_KMZ.zip")

# %%
### get government layers
wisconsin_IG.get_gov_layers()
### get intersection with gbif data
wisconsin_IG.get_intersect_with_gbif(gbif_gdf)

# %%
wisconsin_IG.gbif_intersection_count

# %%
### get gdf for richard bong recreation area
richard_bong_gdf = wisconsin_IG.pa_shp[wisconsin_IG.pa_shp['Loc_Nm'] == 'Richard Bong Recreation Area']

# %%
### plot the government layers for richard bong recreation area
richard_bong_gdf.hvplot(
                geo=True, 
                tiles='EsriImagery',
                title='Richard Bong Recreation Area',
                line_color='blue',
                frame_width=500,
                frame_height=500
            )

# %% [markdown]
# ### Second site in Texas

# %%
### create instance for texas
texas_IG = SciencebaseSite(site_dir_name='indiangrass_TX', filename="PADUS4_1_State_TX_GDB_KMZ.zip")

# %%
### get government layers
texas_IG.get_gov_layers()
### get intersection with gbif data
texas_IG.get_intersect_with_gbif(gbif_gdf)

# %%
texas_IG.gbif_intersection_count

# %%
### get gdf for indiangrass wildlife sanctuary
#i_grass_wildlife_gdf = texas_IG.pa_shp[texas_IG.pa_shp['Loc_Nm'] == 'Indiangrass Wildlife Sanctuary']
balcones_canyonlands_gdf = texas_IG.pa_shp[texas_IG.pa_shp['Loc_Nm'] == 'BALCONES CANYONLANDS NATIONAL WILDLIFE REFUGE']

# %%
### plot the government layers for indiangrass wildlife sanctuary
balcones_canyonlands_gdf.hvplot(
                geo=True, 
                tiles='EsriImagery',
                title='Balcones Canyonlands National Wildlife Refuge',
                line_color='blue',
                frame_width=500,
                frame_height=500
            )

# %%
### combined into sigle gdf 
#sites_gdf = gpd.GeoDataFrame(pd.concat([site_a_gdf, site_b_gdf], ignore_index=True))

# %% [markdown]
# **Reflect and Respond**: 
# Write a site description for each of your sites, or for all of your sites as a group if you have chosen a large number of linked sites. What
# differences or trends in habitat suitability over time do you expect to see among your sites?

# %% [markdown]
# #### Response:
# 
# Since the Indiangrass is native to much of the United States, I chose two sites that are quite far apart, assuming that changes in climate might affect these two sites differently. The first site is located in Wisconsin, at the northern end of the continental United States, and the second site is in Texas, at the southern end. Both sites are nature protection areas:
# 
# ##### Site selection for Wisconsin
# 
# The Richard Bong Recreation Area is home to a moderate population of Sorghastrum nutans.  It would therefore be interesting to know whether the suitability of the habitat will change as a result of climate change. 
# It covers an area of 4,515 acres and is a well-maintained prairie area.
# 
# ##### Site selection for Texas
# Balcones Canyonlands National Wildlife Refuge also shows a moderate to high occurrence, based on the data collected in Texas. 
# The total area is about 72 sq mi and is split in several smaller areas. 
# 
# I expect temperatures to rise at both sites, but I’m not sure if the increase will be the same. I can well imagine that the effects of climate change might be more pronounced at one site, which could lead to varying degrees of decline in suitability in the future—for example, conditions might be less favorable in Textas because it could become too warm there.

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
# Your response here:
# 
# I choose the time period of 1971-2000 for the "present" or "historical" time and the time period of 2070-2099. This provides both a solid time frame for climate data and a sufficient time span to detect any changes caused by climate change.

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

# %%
### import the .csv files with the future scenarios for both sites
### these are stored next to the notebook, so we can use relative file paths to access them.
notebook_path = Path().resolve()
notebook_path

notebook_dir = notebook_path if notebook_path.is_dir() else notebook_path.parent
file_path_FutureScatterTool_BalconesCanyonlands = notebook_dir / 'data_FutureScatterTool_BalconesCanyonlands.csv'
file_path_FutureScatterTool_RichardBongRecreationArea = notebook_dir / 'data_FutureScatterTool_RichardBongRecreationArea.csv'

BalconesCanyonlands_model_df = pd.read_csv(file_path_FutureScatterTool_BalconesCanyonlands, header=10)
RichardBongRecreationArea_model_df = pd.read_csv(file_path_FutureScatterTool_RichardBongRecreationArea, header=10)

# %%
### plot futre scenarios for both sites
fig, axes = plt.subplots(2, 1, figsize=(10, 14))

# First plot
axes[0].scatter(BalconesCanyonlands_model_df["X SCEN"], BalconesCanyonlands_model_df["Y SCEN"])
for x, y, label in zip(BalconesCanyonlands_model_df["X SCEN"], BalconesCanyonlands_model_df["Y SCEN"], BalconesCanyonlands_model_df["Model"]):
    axes[0].text(x, y, label)
axes[0].set_title("Proj for 2070-2099 (RCP 8.5) Balcones Canyonlands National Wildlife Refuge")
axes[0].set_xlabel("JJA max Temp. (°F)")
axes[0].set_ylabel("DJF precip. (inces)")

# Second plot
axes[1].scatter(RichardBongRecreationArea_model_df["X SCEN"], RichardBongRecreationArea_model_df["Y SCEN"])
for x, y, label in zip(RichardBongRecreationArea_model_df["X SCEN"], RichardBongRecreationArea_model_df["Y SCEN"], RichardBongRecreationArea_model_df["Model"]):
    axes[1].text(x, y, label)
axes[1].set_title("Proj for 2070-2099 (RCP 8.5) Richard Bong Recreation Area")
axes[1].set_xlabel("JJA max Temp. (°F)")
axes[1].set_ylabel("DJF precip. (inces)")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Reflect and respond**: Choose at least 4 climate models that cover the range of possible future climate variability at your sites. Which models did you choose, and how did you make that decision?
# 
# Your response here (don't forget to cite the Climate Toolbox): 
# 
# I used the [Climate Futures Toolbox Future Climate Scattertool](https://climatetoolbox.org/tool/Future-Climate-Scatter) to get future scenrarios for both of our sites. 
# These data can be downloaded from the website and are plotted above. 
# I selected the models for the four combinations based on visual assessment and have listed them below for both sites.
# 
# Balcones Canyonlands National Wildlife Refuge:
# 
# -   Warm and wet -> HadGEM2-CC365
# -   Warm and dry -> IPSL-CM5A-MR
# -   Cold and wet -> BNU-ESM
# -   Cold and dry -> inmcm4
# 
# Richard Bong Recreation Area
# 
# -   Warm and wet -> HadGEM2-CC365
# -   Warm and dry -> IPSL-CM5A-MR
# -   Cold and wet -> MRI-CGCM3
# -   Cold and dry -> inmcm4
# 
# 
# Citation: 
# 
# Hegewisch, K.C. and Abatzoglou, J.T..' Future Climate Scatter' web tool. Climate Toolbox (https://climatetoolbox.org/) accessed on [2026, March 3]

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
def getBuffer(site_gdf, buffer_size=0.025):
    """
    Buffer the site_gdf by a specified buffer size.
    site_gdf: GeoDataFrame containing the site geometry.
    buffer_size: The size of the buffer to apply (default is 0.025 degrees).
    
    Returns: A tuple representing the buffered bounds (xmin, ymin, xmax, ymax).
    """ 
    ### make a bounding box for the two sites and plot it on the map
    xmin, ymin, xmax, ymax = site_gdf.total_bounds

    ### initialize tiles to accumulate into 
    tiles = []

    ### buffer
    buffer = buffer_size
    bounds_buffered = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    return bounds_buffered

# %%
### Download and process soil data from the Polaris API for sites
def getPolarisSoilData(site_gdf, 
                       variable_name='ph',
                       depth_range=(100, 200)):
    """
    Get soil data from the Polaris API for a given site and variable.
    Merge downloaded tiles into a single raster and clip to the site boundary.
    site_gdf: A GeoDataFrame containing the geometry of the site.
    variable_name:  Default is 'ph', but can be set to any of the following variables:
                    silt - silt percentage, %
                    sand - sand percentage, %
                    clay - clay percentage, %
                    bd - bulk density, g/cm3
                    theta_s - saturated soil water content, m3/m3
                    theta_r - residual soil water content, m3/m3
                    ksat - saturated hydraulic conductivity, log10(cm/hr)
                    ph - soil pH in H2O, N/A
                    om - organic matter, log10(%)
                    lambda - pore size distribution index (brooks-corey), N/A
                    hb - bubbling pressure (brooks-corey), log10(kPa)
                    n - measure of the pore size distribution (van genuchten), N/A
                    alpha - scale parameter inversely proportional to mean pore diameter (van genuchten), log10(kPa-1)
    depth_range: Depth from surface, default is 100-200 cm, but can be set to any of the following ranges:
                 0-5 cm
                 5-15 cm
                 15-30 cm
                 30-60 cm
                 60-100 cm
                 100-200 cm
    """
    ### make a bounding box for the two sites and plot it on the map
    xmin, ymin, xmax, ymax = site_gdf.total_bounds

    ### initialize tiles to accumulate into 
    tiles = []

    # ### buffer
    # buffer = 0.025
    # bounds_buffered = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)
    bounds_buffered = getBuffer(site_gdf, buffer_size=0.025)

    ### loop 
    for lat_min in range(floor(ymin), ceil(ymax)):
        for lon_min in range(floor(xmin), ceil(xmax)):
            
            ### calculate max lt and lonf for tile 
            lat_max = lat_min + 1
            lon_max = lon_min + 1

            ### url for ph data 
            ph_url = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
                f"/{variable_name}/mean/{depth_range[0]}_{depth_range[1]}/"
                f"/lat{lat_min}{lat_max}_lon{lon_min}{lon_max}.tif"
            )

            ### open raster and append to list
            try:
                ph_tile = rxr.open_rasterio(ph_url, mask_and_scale=True).squeeze()
                #cropped_tile = ph_tile.rio.clip_box(*bounds_buffered)
                tiles.append(ph_tile)
                #tiles.append(cropped_tile)
            except Exception as e:
                print(f"Could not retrieve tile for lat {lat_min} to {lat_max} and lon {lon_min} to {lon_max}. Error: {e}")

    ### merge tiles into single raster
    if tiles:
        #merged_raster = rxrm.merge_arrays(tiles).rio.clip_box(*site_gdf.total_bounds)
        merged_raster = rxrm.merge_arrays(tiles).rio.clip_box(*bounds_buffered)
        #merged_raster = rxrm.merge_arrays(tiles)
        print("Successfully merged tiles into a single raster.")
        ### mask out nodata values
        #merged_raster = merged_raster.where(merged_raster != merged_raster.rio.nodata)
        return merged_raster
    else:    
        print("No tiles were retrieved, so merging was not performed.")

# %%
### get soil data for both sites
soil_ph_das = {
    # Richard Bong Recreation Area
    "RB": getPolarisSoilData(richard_bong_gdf, variable_name='ph', depth_range=(60, 100)), 
    # Balcones Canyonlands National Wildlife Refuge   
    "BC": getPolarisSoilData(balcones_canyonlands_gdf, variable_name='ph', depth_range=(60, 100)) 
}

# %%
# function to save an xarray.DataArray as a raster
def export_to_raster(data_array, raster_path, data_dir):
    """
    Export an xarray.DataArray to a raster file.
    data_array: The xarray.DataArray to export.
    raster_path: The original path of the raster (used to derive the output filename).
    data_dir: The directory where the output raster will be saved.

    returns: None
    """

    output_file = os.path.join(data_dir, os.path.basename(raster_path))
    data_array.rio.to_raster(output_file)

    print(f"Exported DataArray to {output_file}")

# %%
### export soil ph data for both sites to raster files
raster_richard_bong_path = os.path.join(wisconsin_IG.site_dir, 'ph_richard_bong.tif')
export_to_raster(soil_ph_das["RB"], raster_richard_bong_path, wisconsin_IG.site_dir)

raster_balcones_canyonlands_path = os.path.join(texas_IG.site_dir, 'ph_balcones_canyonlands.tif')
export_to_raster(soil_ph_das["BC"], raster_balcones_canyonlands_path, texas_IG.site_dir)

# %%
### function for customazible plots 
def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title, bar_label, plot_cmap, boundary_clr, tif_file =False, vmin=None, vmax=None):
    """
    Create custom site plot.

    site_da: xarray.DataArray containing input site raster.
    site_gdf: GeoDataFrame containing the boundary gdf.
    plots_dir: Directory where the plot will be saved.
    site_fig_name: A short name for the site.
    plot_title: Title for the plot.
    bar_label: Label for the colorbar.
    plot_cmap: Colormap to use for plotting the soil data.
    boundary_clr: Color to use for plotting the site boundaries.
    tif_file: indicate site file

    returns: matplotlib.pyplot.plot of the site with custom settings.
    """
    ### set up the figure 
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()

    ### conditional
    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked=True)

    ### plot data array values 
    site_plot = site_da.plot(
        cmap = plot_cmap, 
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={'label': bar_label})
    
    ### plot site boundaries
    site_gdf.boundary.plot(
        ax=plt.gca(),
        color=boundary_clr)
    
    plt.title(plot_title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    ### save the plot
    fig.savefig(f"{plots_dir}/{site_fig_name}.png", dpi=300)

    return site_plot
            


# %%
# create path to plots, for me, the notebook path fits well
ph_plots_dir = os.path.join(notebook_path, 'soil', 'plots')
os.makedirs(ph_plots_dir, exist_ok=True )

# %%
### plot soil ph for richard bong recreation area
richard_bong_plot = plot_site(site_da=soil_ph_das["RB"], 
                              site_gdf=richard_bong_gdf,
                              plots_dir=ph_plots_dir,
                              site_fig_name='ph_richard_bong',
                              plot_title='Soil pH (60-100 cm depth) for Richard Bong Recreation Area',
                              bar_label='pH',
                              plot_cmap='viridis',
                              boundary_clr='white')

# %%
### plot soil ph for balcones canyonlands national wildlife refuge
balcones_canyonlands_plot = plot_site(site_da=soil_ph_das["BC"], 
                              site_gdf=balcones_canyonlands_gdf,
                              plots_dir=ph_plots_dir,
                              site_fig_name='ph_balcones_canyonlands',
                              plot_title='Soil pH (60-100 cm depth) for Balcones Canyonlands National Wildlife Refuge',
                              bar_label='pH',
                              plot_cmap='viridis',
                              boundary_clr='white')

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
### make directory for topography data
elev_dir = os.path.join(notebook_path, 'topography')
os.makedirs(elev_dir, exist_ok=True)

### make subdir for the two sites 
richard_bong_topo_dir = os.path.join(elev_dir, 'richard_bong')
os.makedirs(richard_bong_topo_dir, exist_ok=True)

balcones_canyonlands_topo_dir = os.path.join(elev_dir, 'balcones_canyonlands')
os.makedirs(balcones_canyonlands_topo_dir, exist_ok=True)

# %%
### login to earth access to access the topography data
earthaccess.login()

# %%
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
### function to get srtm topo data for a site, if it doesn't already exist in the topo dir
def getSRTMTopoData(site_gdf,
                    topo_dir):
    """
    Get SRTM topography data for a given site. 
    If the data already exists in the specified directory, it will be loaded from there. 
    Otherwise, it will be downloaded from Earth Access, unzipped, and processed to extract aspect and slope information.

    site_gdf: A GeoDataFrame containing the geometry of the site.
    topo_dir: The directory where the topography data will be stored.

    returns: A dictionary containing the cropped topography data, aspect, and slope as xarray.DataArrays.
    """

    site_pattern = os.path.join(topo_dir, '*.hgt.zip')

    # ### make a bounding box for the two sites and plot it on the map
    # xmin, ymin, xmax, ymax = site_gdf.total_bounds

    # ### add buffer 
    # buffer = 0.025
    # bounds_buffered = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)
    bounds_buffered = getBuffer(site_gdf, buffer_size=0.025)

    ### look for files in the topo dir
    if not glob(site_pattern):
        ### search for data
        site_srtm_search = earthaccess.search_data(
            short_name = "SRTMGL3",
            bounding_box = bounds_buffered
        )
        ### download data
        earthaccess.download(site_srtm_search, 
                             topo_dir)
        
        ### unzip data
        for zip_file in glob(site_pattern):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(topo_dir)
            print(f"Unzipped {zip_file} to {topo_dir}")
        #return site_srtm_results
        
    else: 
        print(f"SRTM data already exists in {topo_dir}, skipping download.")

    ### open the tile, crop it to the buffered bounds and return it as a data array
    srtm_da_list = []
    srtm_paths = glob(os.path.join(topo_dir, '*.hgt'))
    #srtm_path = glob(os.path.join(topo_dir, '*.hgt'))[0]
    #tile_da = rxr.open_rasterio(srtm_path, masked_and_scale=True).squeeze()
    #cropped_tile = tile_da.rio.clip_box(*bounds_buffered)
    for srtm_path in srtm_paths:
        tile_da = rxr.open_rasterio(srtm_path, masked_and_scale=True).squeeze()
        tmp_cropped_tile = tile_da.rio.clip_box(*bounds_buffered)
        srtm_da_list.append(tmp_cropped_tile)

    ### merge all srtm data arrays
    cropped_tile = rxrm.merge_arrays(srtm_da_list)

    ### get aspect
    aspect_tile = xrspatial.aspect(cropped_tile)
    ### only positive values
    aspect_tile = aspect_tile.where(aspect_tile >= 0)

    ### get slope
    cropped_reporjected = cropped_tile.rio.reproject("EPSG:5070")
    slope_tile = xrspatial.slope(cropped_reporjected)
    slope_tile_4326 = slope_tile.rio.reproject("EPSG:4326")
    slope_tile_4326 = slope_tile_4326.where((slope_tile_4326 > 0) & (slope_tile_4326 < 70))
    
    return {"elevation": cropped_tile, "aspect": aspect_tile, "slope": slope_tile_4326}
    


# %%
### get topo data for both sites
topo_das = {
    # richard bong recreation area
    "RB": getSRTMTopoData(richard_bong_gdf, richard_bong_topo_dir), 
    # indiangrass wildlife sanctuary    
    "BC": getSRTMTopoData(balcones_canyonlands_gdf, balcones_canyonlands_topo_dir) 
}

# %%
# create path to plots, for me, the notebook path fits well
topo_plots_dir = os.path.join(notebook_path, 'topography', 'plots')
os.makedirs(topo_plots_dir, exist_ok=True )

# %% [markdown]
# ### Topography plots for both sites 

# %%
### plot elevation for balcones canyonlands national wildlife refuge
balcones_canyonlands_elev_plot = plot_site(site_da=topo_das["BC"]["elevation"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='elevation_balcones_canyonlands',
                                plot_title='Elevation for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='elevation [m]',
                                plot_cmap='viridis',
                                boundary_clr='white')

# %%
### plot aspect for balcones canyonlands national wildlife refuge
balcones_canyonlands_aspect_plot = plot_site(site_da=topo_das["BC"]["aspect"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='aspect_balcones_canyonlands',
                                plot_title='Aspect for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='Aspect',
                                plot_cmap='viridis',
                                boundary_clr='red')

# %%
### plot slope for balcones canyonlands national wildlife refuge
balcones_canyonlands_slope_plot = plot_site(site_da=topo_das["BC"]["slope"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='slope_balcones_canyonlands',
                                plot_title='Slope for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='Slope',
                                plot_cmap='viridis',
                                boundary_clr='white')

# %%
### plot elevation for richard bong recreation area
richard_bong_elev_plot = plot_site(site_da=topo_das["RB"]["elevation"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='elevation_richard_bong',
                                plot_title='Elevation for Richard Bong Recreation Area',
                                bar_label='elevation [m]',
                                plot_cmap='viridis',
                                boundary_clr='white')

# %%
### plot aspect for richard bong recreation area
richard_bong_aspect_plot = plot_site(site_da=topo_das["RB"]["aspect"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='aspect_richard_bong',
                                plot_title='Aspect for Richard Bong Recreation Area',
                                bar_label='Aspect',
                                plot_cmap='viridis',
                                boundary_clr='red')

# %%
### plot slope for richard bong recreation area
richard_bong_slope_plot = plot_site(site_da=topo_das["RB"]["slope"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=topo_plots_dir,
                                site_fig_name='slope_richard_bong',
                                plot_title='Slope for Richard Bong Recreation Area',
                                bar_label='Slope',
                                plot_cmap='viridis',
                                boundary_clr='white')

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
### make directory for climate data and plots
climate_plots_dir = os.path.join(notebook_path, 'climate', 'plots')
os.makedirs(climate_plots_dir, exist_ok=True )

maca_dir = os.path.join(data_dir, 'maca_dir')
os.makedirs(maca_dir, exist_ok=True)

maca_pattern = os.path.join(maca_dir, '*.nc')

# %%
def convert_temperature_to_celsius(da):
    """
    Convert temperature from Kelvin to Celsius in an xarray.DataArray.
    da: The xarray.DataArray containing temperature data in Kelvin.

    returns: An xarray.DataArray with temperature converted to Celsius.
    """
    return da - 273.15


def convert_longitude(longitude):
    """
    Convert longitude values from 0-360 to -180 to 180 in an xarray.DataArray.
    da: The xarray.DataArray containing longitude data.

    returns: An xarray.DataArray with longitude values converted to the -180 to 180 range.
    """
    return (longitude -360) if longitude > 180 else longitude

# %%
def getMACADateRanges(start_year, end_year):
    """
    Generate a list of date ranges for MACA data based on the provided start and end years.
    start_year: The starting year for the date ranges.
    end_year: The ending year for the date ranges.  
    """
    intervals = []

    # historical
    intervals += [(y, y + 4) for y in range(1950, 2001, 5)]

    # transition
    intervals.append((2005, 2005))

    # future
    intervals += [(y, y + 4) for y in range(2006, 2096, 5)]

    # final
    intervals.append((2096, 2099))

    return [
        f"{s}_{e}"
        for s, e in intervals
        if s <= end_year and e >= start_year
    ]

# %%
def processMACAData(site_name,
                    site_gdf, 
                    date_ranges, 
                    models,
                    rcp_value,
                    variable_names, 
                    maca_dir):
        """
        Process MACA data for a given site, date ranges, climate models, RCP value, and variable names.

        site_name: A short name for the site (e.g., "RB" for Richard Bong Recreation Area).
        site_gdf: A GeoDataFrame containing the geometry of the site.
        date_ranges: A list of date ranges to process (e.g., ["1950_1954", "1955_1959", ...]).
        models: A list of climate models to process (e.g., ["ACCESS1-0", "BCC-CSM1-1", ...]).  
        rcp_value: The RCP value to process (e.g., "rcp85").
        variable_names: A list of variable names to process (e.g., ["pr", ...]).
        maca_dir: The directory where MACA data will be stored.     

        returns: A list of dictionaries containing the site name, climate model, variable name, date range, and the processed xarray.DataArray for each combination of inputs.
        """
        ### initialize list to accumulate results into
        results_list = []

        for date_range in date_ranges:
                for model in models:
                        for variable_name in variable_names:
                                maca_path = os.path.join(maca_dir, f'maca_{model}_{site_name}_{variable_name}_{rcp_value}_{date_range}_CONUS_monthly.nc')
                                ### construct the url for the maca data based on the inputs
                                maca_url = (
                                        "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
                                        "/MACAV2"
                                        f"/{model}"
                                        f"/macav2metdata_{variable_name}"
                                        f"_{model}_r1i1p1"
                                        f"_{rcp_value}"
                                        f"_{date_range}_CONUS"
                                        "_monthly.nc"
                                        )
                                ### check if the maca data already exists, if not, download it and save it to the maca dir
                                if not os.path.exists(maca_path):
                                        maca_da = xr.open_dataset(maca_url, engine="netcdf4").squeeze()
                                        maca_da.to_netcdf(maca_path)
                                        print(f"Downloaded MACA data for {site_name} and saved to {maca_path}")
                                else:
                                        print(f"MACA data for {site_name} already exists at {maca_path}, skipping download.")
                                
                                ### open the maca data and process it to clip to the site boundary
                                maca_da = xr.open_dataset(maca_path).squeeze()

                                ### reproject the site gdf to match the maca data crs
                                site_rpj = site_gdf.to_crs(maca_da.rio.crs)
                                ### convert longitude values to -180 to 180 range
                                maca_da = maca_da.assign_coords(
                                        lon= ("lon", [convert_longitude(lon) for lon in maca_da.lon.values])
                                        )
                                ### set spatial dimensions for the maca data
                                maca_da = maca_da.rio.set_spatial_dims(
                                        x_dim="lon", 
                                        y_dim="lat"
                                        )
                                
                                maca_da_cropped = maca_da.rio.clip_box(*site_rpj.total_bounds, allow_one_dimensional_raster=True)
                                
                                ### create results dictionary and append to results list
                                result = dict(
                                        site_name = site_name,
                                        climate_model = model,
                                        variable_name = variable_name,
                                        data_range = date_range,
                                        da = maca_da_cropped
                                        )
                                
                                results_list.append(result)

        return results_list



# %% [markdown]
# Indiangrass Wildlife Sanctuary:
# 
# -   Warm and wet -> HadGEM2-CC365
# -   Warm and dry -> IPSL-CM5A-MR
# -   Cold and wet -> BNU-ESM
# -   Cold and dry -> inmcm4
# 
# Richard Bong Recreation Area
# 
# -   Warm and wet -> HadGEM2-CC365
# -   Warm and dry -> IPSL-CM5A-MR
# -   Cold and wet -> MRI-CGCM3
# -   Cold and dry -> inmcm4

# %%
### get date ranges for historical and future periods
dates_ranges_historical = getMACADateRanges(1970, 1999) 
dates_ranges_future = getMACADateRanges(2071, 2099)

### list of climate models to process for each site, based on the models used in the future scenarios plots
richard_bong_models = ["HadGEM2-CC365", "IPSL-CM5A-MR", "MRI-CGCM3", "inmcm4"]
balcones_canyonlands_models = ["HadGEM2-CC365", "IPSL-CM5A-MR", "BNU-ESM", "inmcm4"]


# %%
### process MACA data for both sites, for the historical and future periods,
### and for the specified climate models and variables
richard_bing_hist = processMACAData("RB",
                                        richard_bong_gdf, 
                                        dates_ranges_historical, 
                                        richard_bong_models,
                                        "historical", 
                                        ["pr", "tasmin", "tasmax"], 
                                        maca_dir)

richard_bing_future = processMACAData("RB",
                                        richard_bong_gdf, 
                                        dates_ranges_future, 
                                        richard_bong_models,
                                        "rcp85", 
                                        ["pr", "tasmin", "tasmax"], 
                                        maca_dir)

balcones_canyonlands_hist = processMACAData("BC",
                                        balcones_canyonlands_gdf, 
                                        dates_ranges_historical, 
                                        balcones_canyonlands_models,
                                        "historical", 
                                        ["pr", "tasmin", "tasmax"], 
                                        maca_dir)

balcones_canyonlands_future = processMACAData("BC",
                                        balcones_canyonlands_gdf, 
                                        dates_ranges_future, 
                                        balcones_canyonlands_models,
                                        "rcp85", 
                                        ["pr", "tasmin", "tasmax"], 
                                        maca_dir)

# %%
def calc_mean_temp(tasmin_das, tasmax_das, months_range=None):
    """
    Calculate mean temperature from lists of tasmin and tasmax DataArrays.

    Parameters
    ----------
    tasmin_das : list[xr.DataArray]
        Daily minimum temperature DataArrays with a time dimension.
    tasmax_das : list[xr.DataArray]
        Daily maximum temperature DataArrays with a time dimension.
    months_range : list[int], optional
        Months to include (e.g., [4,5,6,7,8,9] for growing season).

    Returns
    -------
    xr.DataArray
        Mean temperature averaged across years.
    """
    # combine all arrays along time
    tasmin = xr.concat(tasmin_das, dim="time").sortby("time")
    tasmax = xr.concat(tasmax_das, dim="time").sortby("time")

    # daily mean temperature from min and max
    tmean = (tasmin + tasmax) / 2.0

    # select months if requested
    if months_range is not None:
        tmean = tmean.sel(time=tmean.time.dt.month.isin(months_range))

    # annual mean temperature
    annual = tmean.resample(time="YE").mean()

    # mean across years
    result = annual.mean(dim="time")

    # ensure spatial metadata
    result = result.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    if result.rio.crs is None:
        result = result.rio.write_crs("EPSG:4326")

    return result


def site_seasonal_temp(site_results,
                       site_name,
                       models,
                       months_range=None):
    """
    Calculate seasonal mean temperature for a site from processed MACA data.

    Parameters
    ----------
    site_results : list[dict]
        List of dictionaries containing processed MACA data for a site.
    site_name : str
        Short name for the site (e.g., "BC").
    models : list[str]
        List of climate model names.
    months_range : list[int], optional
        Months to include in the seasonal calculation.

    Returns
    -------
    dict
        Dictionary with model names as keys and mean temperature DataArrays as values,
        plus a "models_mean" key containing the ensemble mean across all models.
    """
    mean_temp = {}

    for model in models:
        # filter tasmin and tasmax for this site and model
        tasmin_das = [
            r['da'] for r in site_results
                if r['site_name'] == site_name
                and r['climate_model'] == model
                and r['variable_name'] == 'tasmin'
        ]
        tasmax_das = [
            r['da'] for r in site_results
                if r['site_name'] == site_name
                and r['climate_model'] == model
                and r['variable_name'] == 'tasmax'
        ]

        # compute seasonal mean temperature for this model
        tmp_mean_temp = calc_mean_temp(tasmin_das, tasmax_das, months_range=months_range)

        mean_temp[model] = convert_temperature_to_celsius(tmp_mean_temp.air_temperature)

    # ensemble mean across all models
    models_mean = xr.concat(
        [mean_temp[m] for m in models],
        dim="model"
    ).mean(dim="model")

    mean_temp["models_mean"] = models_mean

    return mean_temp

# %%
def calc_mean_precip(das, months_range=None):
    """
    Calculate the mean annual precipitation from a list of xarray.DataArrays.

    Parameters
    ----------
    das : list[xr.DataArray]
        List of precipitation DataArrays with a time dimension.
    months_range : list[int], optional
        Months to include (e.g., [4,5,6,7,8,9] for growing season).

    Returns
    -------
    xr.DataArray
        Mean annual precipitation.
    """

    # combine all arrays along time
    combined = xr.concat(das, dim="time").sortby("time")

    # select months if requested
    if months_range is not None:
        combined = combined.sel(time=combined.time.dt.month.isin(months_range))

    # annual totals
    annual = combined.resample(time="YE").sum()

    # mean across years
    result = annual.mean(dim="time")

    # ensure spatial metadata
    result = result.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    if result.rio.crs is None:
        result = result.rio.write_crs("EPSG:4326")

    return result


def site_seasonal_precip(site_results, 
                         site_name,
                         models, 
                         months_range=None):
    """
    Calculate seasonal mean precipitation for a site from processed MACA data.

    Parameters
    ----------
    site_results : list[dict]
        List of dictionaries containing processed MACA data for a site.
    months_range : list[int], optional
        Months to include in the seasonal calculation (e.g., [4,5,6,7,8,9] for growing season).

    Returns
    -------
    dict
        Dictionary with keys as (model, data_range) and values as mean precipitation DataArrays.
    """

    mean_precip = {}

    for model in models:
        pr_das = [
            r['da'] for r in site_results
                if r['site_name'] == site_name
                and r['climate_model'] == model
                and r['variable_name'] == 'pr'
            ]

        tmp_mean_precip = calc_mean_precip(pr_das, months_range=months_range)

        mean_precip[model] = tmp_mean_precip.precipitation

    models_mean = xr.concat(
        [mean_precip[m] for m in models],
        dim="model"
        ).mean(dim="model")
    
    mean_precip["models_mean"] = models_mean

    return mean_precip



# %%
### calculate seasonal mean temperature and precipitation for both sites, for the historical and future periods, and for the specified climate models
climate_das = {
    "RB": {
        "models" : richard_bong_models,
        # hist is 1970 to 1999
        "hist_pr": site_seasonal_precip(richard_bing_hist, "RB", richard_bong_models),
        "hist_temp": site_seasonal_temp(richard_bing_hist, "RB", richard_bong_models),
        # future is 2071 to 2099
        "future_pr": site_seasonal_precip(richard_bing_future, "RB", richard_bong_models),
        "future_temp": site_seasonal_temp(richard_bing_future, "RB", richard_bong_models)

    },
    "BC": {
        "models" : balcones_canyonlands_models,
        # hist is 1970 to 1999
        "hist_pr": site_seasonal_precip(balcones_canyonlands_hist, "BC", balcones_canyonlands_models),
        "hist_temp": site_seasonal_temp(balcones_canyonlands_hist, "BC", balcones_canyonlands_models),
        # future is 2071 to 2099
        "future_pr": site_seasonal_precip(balcones_canyonlands_future, "BC", balcones_canyonlands_models),
        "future_temp": site_seasonal_temp(balcones_canyonlands_future, "BC", balcones_canyonlands_models)
    }
}

# %%
### plot historical and future seasonal mean temperature and precipitation for richard bong recreation area
richard_bong_climate_hist_pr_plot = plot_site(site_da=climate_das["RB"]["hist_pr"]["models_mean"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_richard_bong_hist_pr',
                                plot_title='Historical Climate for Richard Bong Recreation Area',
                                bar_label='precipitation [mm]',
                                plot_cmap='viridis',
                                boundary_clr='white')

richard_bong_climate_future_pr_plot = plot_site(site_da=climate_das["RB"]["future_pr"]["models_mean"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_richard_bong_future_pr',
                                plot_title='Future Climate for Richard Bong Recreation Area',
                                bar_label='precipitation [mm]',
                                plot_cmap='viridis',
                                boundary_clr='white')

richard_bong_climate_hist_temp_plot = plot_site(site_da=climate_das["RB"]["hist_temp"]["models_mean"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_richard_bong_hist_temp',
                                plot_title='Historical Climate for Richard Bong Recreation Area',
                                bar_label='temperature [°C]',
                                plot_cmap='viridis',
                                boundary_clr='white')

richard_bong_climate_future_temp_plot = plot_site(site_da=climate_das["RB"]["future_temp"]["models_mean"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_richard_bong_future_temp',
                                plot_title='Future Climate for Richard Bong Recreation Area',
                                bar_label='temperature [°C]',
                                plot_cmap='viridis',
                                boundary_clr='white')

# %%
### plot historical and future seasonal mean temperature and precipitation for balcones canyonlands national wildlife refuge
balcones_canyonlands_climate_hist_pr_plot = plot_site(site_da=climate_das["BC"]["hist_pr"]["models_mean"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_balcones_canyonlands_hist_pr',
                                plot_title='Historical Climate for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='precipitation [mm]',
                                plot_cmap='viridis',
                                boundary_clr='white')

balcones_canyonlands_climate_future_pr_plot = plot_site(site_da=climate_das["BC"]["future_pr"]["models_mean"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_balcones_canyonlands_future_pr',
                                plot_title='Future Climate for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='precipitation [mm]',
                                plot_cmap='viridis',
                                boundary_clr='white')

balcones_canyonlands_climate_hist_temp_plot = plot_site(site_da=climate_das["BC"]["hist_temp"]["models_mean"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_balcones_canyonlands_hist_temp',
                                plot_title='Historical Climate for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='temperature [°C]',
                                plot_cmap='viridis',
                                boundary_clr='white')

balcones_canyonlands_climate_future_temp_plot = plot_site(site_da=climate_das["BC"]["future_temp"]["models_mean"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=climate_plots_dir,
                                site_fig_name='climate_balcones_canyonlands_future_temp',
                                plot_title='Future Climate for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='temperature [°C]',
                                plot_cmap='viridis',
                                boundary_clr='white')

# %% [markdown]
# **Reflect and respond**: Make sure to include a description of the climate data and how you selected your models. Include a citation of the MACAv2 data.

# %% [markdown]
# Your response here:
# 
# The MACA v2 (Multivariate Adaptive Constructed Analogs Version 2) dataset is a statistically downscaled climate dataset that provides high-resolution (~4 km) projections of temperature, precipitation, and other climate variables across the United States. It uses outputs from the Coupled Model Intercomparison Project Phase 5 (CMIP5). 
# MACA v2 data was extrated from <a
# href="http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html"> their Thredds server</a>. 
# 
# Two time periods were selected: 1970 to 1999 for historical data on mean annual temperature and  annual precipitation, and 2071 to 2099 for projections of future mean annual temperature and annual precipitation trends. The RCP8.5 scenario, which depicts a future characterized by high emissions and high consumption of fossil fuels, was used for the future projections.
# 
# For the Richard Bong Recreational Area site, there is a moderate increase in precipitation and a substantial rise in annual mean temperature. However, based on current findings regarding habitat suitability, this should have only a minor effect and remain within the optimal and tolerable range for habitat suitability. 
# 
# For Balcones Canyonlands National Wildlife Refuge, there is a moderate decrease in precipitation and an increase in temperature. However, the historical annual mean temperature value here is already substantial higher, and the projected value is even higher, so it may fall outside the range of habitat suitability.
# 
# 
# http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html
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

# %%
def plot_all_harmonized(site_name, harmonized_dict, site_gdf, plots_dir=None):
    """
    Plot all harmonized rasters for one site in a single overview figure.

    site_name: Short site identifier (e.g. 'RB' or 'BC').
    harmonized_dict: Dict returned by harmonize_das() for this site, with keys
                     'soil', 'topo', 'climate_mean'.
    site_gdf: GeoDataFrame containing the site boundary.
    plots_dir: Optional directory path. If provided, saves the figure as
               '<plots_dir>/harmonized_<site_name>.png'.

    returns: matplotlib Figure
    """
    ### define the panels to plot
    panels = []

    # Soil
    panels.append({
        "da": harmonized_dict["soil"],
        "title": "Soil pH",
        "label": "pH",
        "cmap": "viridis"
    })

    ### Topography
    topo_meta = {
        "elevation": {"label": "elevation [m]", "cmap": "terrain"},
        "aspect":    {"label": "aspect [°]",    "cmap": "twilight"},
        "slope":     {"label": "slope [°]",     "cmap": "YlOrRd"},
    }
    for var, meta in topo_meta.items():
        if var in harmonized_dict.get("topo", {}):
            panels.append({
                "da": harmonized_dict["topo"][var],
                "title": var.capitalize(),
                "label": meta["label"],
                "cmap": meta["cmap"]
            })

    ### Climate mean
    climate_meta = {
        "hist_pr":    {"title": "Historical Precipitation",  "label": "precip [mm]",  "cmap": "Blues"},
        "future_pr":  {"title": "Future Precipitation",      "label": "precip [mm]",  "cmap": "Blues"},
        "hist_temp":  {"title": "Historical Temperature",    "label": "temp [°C]",    "cmap": "RdYlBu_r"},
        "future_temp":{"title": "Future Temperature",        "label": "temp [°C]",    "cmap": "RdYlBu_r"},
    }
    for var, meta in climate_meta.items():
        if var in harmonized_dict.get("climate_mean", {}):
            panels.append({
                "da": harmonized_dict["climate_mean"][var],
                "title": meta["title"],
                "label": meta["label"],
                "cmap": meta["cmap"]
            })

    ###clayout
    n = len(panels)
    ncols = 4
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, panel in enumerate(panels):
        ax = axes[i]
        da = panel["da"]
        ### squeeze band dim if present
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        da.plot(ax=ax, cmap=panel["cmap"], cbar_kwargs={"label": panel["label"]})
        site_gdf.boundary.plot(ax=ax, color="red", linewidth=0.8)
        ax.set_title(panel["title"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    #### hide unused axes
    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Harmonized Rasters — {site_name}", fontsize=14, y=1.01)
    plt.tight_layout()

    if plots_dir is not None:
        fig.savefig(f"{plots_dir}/harmonized_{site_name}.png", dpi=150, bbox_inches="tight")

    #return fig

# %%
def harmonize_das(site_name, site_gdf, soil_ph_das, topo_das, climate_das):
    """
    Harmonize the coordinate reference systems and spatial dimensions of topography, soil, and climate DataArrays for a given site.

    Parameters
    ----------
    site_name : str
        Short name for the site (e.g., "RB" for Richard Bong).
    site_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the geometry of the site.
    soil_ph_das : dict 
        Dictionary containing soil pH DataArrays for each site. 
    topo_das : dict
        Dictionary containing topography DataArrays for each site. 
    climate_das : dict
        Dictionary containing climate DataArrays for each site, organized by historical and future scenarios.
    
    Returns
    -------
    """
    climate_models = climate_das[site_name]["models"]

    ### define the boundaries 
    site_bounds = tuple(site_gdf.total_bounds)

    ### ass a small buffer 
    buffer = 0.025
    (xmin, ymin, xmax, ymax) = site_bounds

    ### buffer bounding box
    site_buffer = (xmin - buffer,
                            ymin - buffer,
                            xmax + buffer,
                            ymax + buffer)
    
    ### reference
    ref_da = soil_ph_das[site_name]
    harmonized_soil = ref_da

    ### first harmonize topo data
    harmonized_topo = {}
    for key in topo_das[site_name].keys():
        ### crop the data 
        cropped_da = topo_das[site_name][key].rio.clip_box(*site_buffer)
        ### reproject and match 
        reproj_da = (cropped_da.rio.reproject_match(ref_da))
        ### add it to the list 
        harmonized_topo[key] = reproj_da

    ### harminize climate data
    harmonized_climate = {}
    for model in climate_models:
        cropped_hist_pr_da = climate_das[site_name]["hist_pr"][model].rio.clip_box(*site_buffer)
        cropped_future_pr_da = climate_das[site_name]["future_pr"][model].rio.clip_box(*site_buffer)

        cropped_hist_temp_da = climate_das[site_name]["hist_temp"][model].rio.clip_box(*site_buffer)
        cropped_future_temp_da = climate_das[site_name]["future_temp"][model].rio.clip_box(*site_buffer)
        ### reproject and match 
        reproj_hist_pr_da = cropped_hist_pr_da.rio.reproject_match(ref_da)
        reproj_future_pr_da = cropped_future_pr_da.rio.reproject_match(ref_da)

        reproj_hist_temp_da = cropped_hist_temp_da.rio.reproject_match(ref_da)
        reproj_future_temp_da = cropped_future_temp_da.rio.reproject_match(ref_da)
        ### add to harmonized climate dictionary
        harmonized_climate[model] = {
            "hist_pr": reproj_hist_pr_da,
            "future_pr": reproj_future_pr_da,
            "hist_temp": reproj_hist_temp_da,
            "future_temp": reproj_future_temp_da
        }

    cropped_hist_pr_mean_da = climate_das[site_name]["hist_pr"]["models_mean"].rio.clip_box(*site_buffer)
    cropped_future_pr_mean_da = climate_das[site_name]["future_pr"]["models_mean"].rio.clip_box(*site_buffer)
    
    cropped_hist_temp_mean_da = climate_das[site_name]["hist_temp"]["models_mean"].rio.clip_box(*site_buffer)
    cropped_future_temp_mean_da = climate_das[site_name]["future_temp"]["models_mean"].rio.clip_box(*site_buffer)
    ### reproject and match 
    reproj_hist_pr_mean_da = cropped_hist_pr_mean_da.rio.reproject_match(ref_da)
    reproj_future_pr_mean_da = cropped_future_pr_mean_da.rio.reproject_match(ref_da)
    reproj_hist_temp_mean_da = cropped_hist_temp_mean_da.rio.reproject_match(ref_da)
    reproj_future_temp_mean_da = cropped_future_temp_mean_da.rio.reproject_match(ref_da)

    return {"soil": harmonized_soil, 
            "topo": harmonized_topo, 
            "climate": harmonized_climate,
            "climate_mean": {
                "hist_pr": reproj_hist_pr_mean_da,
                "future_pr": reproj_future_pr_mean_da,
                "hist_temp": reproj_hist_temp_mean_da,
                "future_temp": reproj_future_temp_mean_da
            }
            }
        

# %%
# harmonize data arrays for both sites
richard_bong_harmonized = harmonize_das("RB", richard_bong_gdf, soil_ph_das, topo_das, climate_das)
balcones_canyonlands_harmonized = harmonize_das("BC", balcones_canyonlands_gdf, soil_ph_das, topo_das, climate_das)

# %%
### make directory for harmonized plots
harmonized_plots_dir = os.path.join(notebook_path, 'harmonized_plots')
os.makedirs(harmonized_plots_dir, exist_ok=True )

# %%
plot_all_harmonized("RB", richard_bong_harmonized, richard_bong_gdf, plots_dir=harmonized_plots_dir)

# %%
plot_all_harmonized("BC", balcones_canyonlands_harmonized, balcones_canyonlands_gdf, plots_dir=harmonized_plots_dir)

# %%
### sanity check
print(richard_bong_harmonized["soil"].rio.bounds())
print(richard_bong_harmonized["topo"]["elevation"].rio.bounds())
print(richard_bong_harmonized["climate_mean"]["hist_pr"].rio.bounds())
print(richard_bong_harmonized["climate_mean"]["hist_temp"].rio.bounds())

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
### directory for suitability plots 
suitability_plots_dir = os.path.join(notebook_path, 'suitability_plots')
os.makedirs(suitability_plots_dir, exist_ok=True )

# %%
# Membership functions for fuzzy logic
def trimf(x, a, b, c):
    """Triangular membership: rises a→b, falls b→c."""
    return np.clip(
        np.where(x <= b,
                 (x - a) / (b - a + 1e-10),
                 (c - x) / (c - b + 1e-10)),
        0, 1)

def trapmf(x, a, b, c, d):
    """Trapezoidal membership: rises a→b, flat b→c, falls c→d."""
    return np.clip(
        np.minimum(
            (x - a) / (b - a + 1e-10),
            (d - x) / (d - c + 1e-10)),
        0, 1)

def gaussmf(x, mean, sigma):
    """Gaussian membership: peak at mean, spread controlled by sigma."""
    return np.exp(-0.5 * ((x - mean) / (sigma + 1e-10)) ** 2)

def gaussmf_double(x, mean1, sigma1, mean2, sigma2):
    """Two-sided Gaussian: rises toward mean1, flat between, falls from mean2."""
    left  = gaussmf(x, mean1, sigma1)
    right = gaussmf(x, mean2, sigma2)
    return np.where(x <= mean1, left,
           np.where(x >= mean2, right, 1.0))

# %%
# Per-variable membership functions
# Adjust breakpoints to match your target species' known tolerances.

def soil_ph_suitability(ph):
    """Optimal pH ~6.0–7.0, tolerates 5.5–7.5."""
    return trapmf(ph, 4.5, 5.5, 7.0, 8.0)

def elevation_suitability(elev_m):
    """Optimal 500–1500 m, tolerates 200–2000 m."""
    return trapmf(elev_m, 0, 150, 1000, 2100)

def slope_suitability(slope_deg):
    """Prefers gentle slopes; steep terrain reduces suitability."""
    return trapmf(slope_deg, 0, 0, 10, 30)

def aspect_suitability(aspect_deg):
    """S-SW (180–240°) preferred; but all aspects acceptable."""
    return trapmf(aspect_deg, 0, 180, 240, 360)

# def precip_suitability(precip_mm):
#     """Optimal 600-1000 mm/yr, tolerates 400–1200 mm/yr."""
#     return trapmf(precip_mm, 400, 600, 1000, 1200)

def precip_suitability(precip_mm):
    """Optimal ~800 mm/yr, sigma=200 gives ~60% membership at 600 and 1000 mm."""
    return gaussmf(precip_mm, mean=800, sigma=200)

# def temperature_suitability(temp_c):
#     """Optimal 10–18°C, tolerates 5–22°C."""
#     return trapmf(temp_c, 5, 10, 18, 22)

def temperature_suitability(temp_c):
    """
    Temperature suitability using a double Gaussian.
    Adjust mean1, mean2, and sigmas to your species' thermal tolerance.
    """
    # Example: optimal range 10–18 °C, sharp cold limit, gradual heat falloff
    return gaussmf_double(temp_c,
                          mean1=10,  sigma1=5, 
                          mean2=18,  sigma2=8)  

# %%
def fuzzy_habitat_suitability(soil_ph_da, elev_da, slope_da, aspect_da, precip_da, temperature_da):
    """
    Compute fuzzy habitat suitability score (0–1) from raster DataArrays.

    Parameters
    ----------
    soil_ph_da, elev_da, slope_da, aspect_da, precip_da : xr.DataArray
        Input rasters, all reprojected/resampled to the same grid.
    weights : dict, optional
        Per-variable weights, e.g. {"ph": 0.3, "elev": 0.2, ...}.
        Defaults to equal weights.

    Returns
    -------
    xr.DataArray
        Habitat suitability score between 0 (unsuitable) and 1 (optimal).
    """

    ### apply membership function to each input layer → values in [0, 1
    scores = {
        "ph":     soil_ph_suitability(soil_ph_da),
        "elev":   elevation_suitability(elev_da),
        "slope":  slope_suitability(slope_da),
        "aspect": aspect_suitability(aspect_da),
        "precip": precip_suitability(precip_da),
        "temp":   temperature_suitability(temperature_da)
    }

    suitability = scores["ph"]
    for key in ["elev", "slope", "aspect", "precip", "temp"]:
        suitability *= scores[key]


    suitability.name = "habitat_suitability"
    suitability.attrs["long_name"] = "Fuzzy habitat suitability (0–1)"

    return suitability

# %%
suitability_dict = {}

for site_name in ["RB", "BC"]:
    print(f"Calculating suitability for {site_name}...")
    site_soil = richard_bong_harmonized["soil"] if site_name == "RB" else balcones_canyonlands_harmonized["soil"]
    site_topo = richard_bong_harmonized["topo"] if site_name == "RB" else balcones_canyonlands_harmonized["topo"]
    site_climate_hist_pr = richard_bong_harmonized["climate_mean"]["hist_pr"] if site_name == "RB" else balcones_canyonlands_harmonized["climate_mean"]["hist_pr"]
    site_climate_future_pr = richard_bong_harmonized["climate_mean"]["future_pr"] if site_name == "RB" else balcones_canyonlands_harmonized["climate_mean"]["future_pr"]
    # temp muist be converted to celsius for the suitability function
    site_climate_hist_temp = richard_bong_harmonized["climate_mean"]["hist_temp"] if site_name == "RB" else balcones_canyonlands_harmonized["climate_mean"]["hist_temp"]
    site_climate_future_temp = richard_bong_harmonized["climate_mean"]["future_temp"] if site_name == "RB" else balcones_canyonlands_harmonized["climate_mean"]["future_temp"]

    suitability_hist = fuzzy_habitat_suitability(
        soil_ph_da=site_soil,
        elev_da=site_topo["elevation"],
        slope_da=site_topo["slope"],
        aspect_da=site_topo["aspect"],
        precip_da=site_climate_hist_pr,
        temperature_da=site_climate_hist_temp
    )

    suitability_future = fuzzy_habitat_suitability(
        soil_ph_da=site_soil,
        elev_da=site_topo["elevation"],
        slope_da=site_topo["slope"],
        aspect_da=site_topo["aspect"],
        precip_da=site_climate_future_pr,
        temperature_da=site_climate_future_temp
    )

    suitability_dict[site_name] = {
        "historical": suitability_hist,
        "future": suitability_future
    }

# %% [markdown]
# ## STEP 5: Present your results
# Generate some plots that show your key findings of habitat suitability in your study sites across the different time periods and climate models. Don’t forget to interpret your plots!

# %%
richard_bong_suitability_hist = plot_site(site_da=suitability_dict["RB"]["historical"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=suitability_plots_dir,
                                site_fig_name='suitability_richard_bong_historical',
                                plot_title='Historical Suitability for Richard Bong Recreation Area',
                                bar_label='Fuzzy habitat suitability',
                                plot_cmap='YlGn',
                                boundary_clr='red',
                                vmin=0, vmax=1)

richard_bong_suitability_future = plot_site(site_da=suitability_dict["RB"]["future"], 
                                site_gdf=richard_bong_gdf,
                                plots_dir=suitability_plots_dir,
                                site_fig_name='suitability_richard_bong_future',
                                plot_title='Future Suitability for Richard Bong Recreation Area',
                                bar_label='Fuzzy habitat suitability',
                                plot_cmap='YlGn',
                                boundary_clr='red',
                                vmin=0, vmax=1)


# %%
balcones_canyonlands_suitability_hist = plot_site(site_da=suitability_dict["BC"]["historical"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=suitability_plots_dir,
                                site_fig_name='suitability_balcones_canyonlands_historical',
                                plot_title='Historical Suitability for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='Fuzzy habitat suitability',
                                plot_cmap='YlGn',
                                boundary_clr='red',
                                vmin=0, vmax=1)

balcones_canyonlands_suitability_future = plot_site(site_da=suitability_dict["BC"]["future"], 
                                site_gdf=balcones_canyonlands_gdf,
                                plots_dir=suitability_plots_dir,
                                site_fig_name='suitability_balcones_canyonlands_future',
                                plot_title='Future Suitability for Balcones Canyonlands National Wildlife Refuge',
                                bar_label='Fuzzy habitat suitability',
                                plot_cmap='YlGn',
                                boundary_clr='red',
                                vmin=0, vmax=1)

# %% [markdown]
# Interpret your plots here:
# 
# ### The continental United States shows varying changes in habitat suitability for Indian grass within the RCP8.5 scenario projection
# 
# In this study, I examined the suitability of the habitat for Indian grass (Sorghastrum nutans) on the continental United States.
# The focus was on whether this suitability changes differently in the northern and southern parts of the U.S. under the RCP8.5 scenario (high emissions). 
# 
# Various factors were taken into account, including soil pH, elevation, aspect, slope, as well as historical and projected values for temperature and precipitation. It is primarily the values for temperature and precipitation that are likely to change due to a shifting climate.
# 
# Two sites were examined for this purpose, both with a moderate abundance of Indian grass: Richard Bong Recreationn Area in Wisconsin and Balcones Canyonlands National Wildlife Refuge in Texas. For both sites, a historical period was examined (1970-1999), as well as a period based on the RCP8.5 projection (2071-2099).
# 
# The last four figures in this analysis show the habitat suitability (0 = poor to 1 = good) for the two sites as well as for the two climatological time periods.
# It can be seen that, for the Richard Bong Recreation Areas in Wisconsin, the suitability of the habitat is projected to remain stable or increase slightly in the projection for the future. Mean model results show a slight increase in precipitation and substantial increase in mean annual temperature but within the suitable range for Indian grass. For the Balcones Canyonlands National Wildlife Refuge site, it is clear that habitat suitability is projected to decline in the future, making it less suitable for Indian grass. Annual precipitation is decreasing considerably, and the average annual temperature is rising significantly.
# 
# If we consider these two locations to be representative of the northern and southern parts of the continental United States, we can expect that the suitability of the habitat for Indian grass would tend to shift northward in this extreme climate change scenario characterized by persistently high emissions.
# 
# 
# 
# 


