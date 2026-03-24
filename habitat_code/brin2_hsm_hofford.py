# %% [markdown]
# # Habitat suitability under climate change
# 
# This coding challenge is to develop a fuzzy logic habitat suitability model 
# for the invasive perennial grass species, *Bromus inermis* (Smooth brome).
# 
# I draw on species distribution data from GBIF, climate data from the MACAv2 dataset,
# soil data from Polaris, and topographic data from the NASA Shuttle Radar Topography Mission (SRTMGL3).
# 
# Using species tolerance information from the USDA, extension documents, and the scientific literature,
# I create tolerance ranges for this species and investigate habitat suitability at two national grasslands: 
# Pawnee National Grassland (NE Colorado) and Thunder Basin National Grassland (NE Wyoming).
# I investigate how this suitability might change across four different climate scenarios representing
# hot/dry, hot/wet, cold/dry, and cold/wet climate futures. 

# %% [markdown]
# ## STEP 1: Study overview
# 
# ### Step 1a: Select a species
# ***Bromus inermis***, also known as smooth brome, is a perennial non-native grass common across the United States and distrubuted across the world. Smooth brome is sometimes planted for soil stabilization or for forage (cite), but in some cases can become a monoculture and drastically decrease native species biodiversity (cite).
# 
# Smooth brome is native to ... and was introduced in ... for ... 
# 
# This species is not currently listed as a noxious weed in Colorado or Wyoming, yet many land managers recongize the potential negative impact of this species and are working to control it (cite Louisville). It may become even more relevant as managers are increasingly using pre-emergent herbicides like imaziflan to control annual invasive grasses. Smooth brome is a perennial grass and thus not affected by this type of herbicide. After the monoculture of annual invasives are removed, perennial invasives like smooth brome can outcompete natives and become a secondary issue.
# 
# 
# **Question:** Climate change can also alter the impact and spread of invasive species. It is important to understand the climate, environmental, and ecological variables that control the niche boundary of invasive species in order to make effective management decisions that will be relevant under a changing climate. For this project, I was curious if the habitat of protected areas like Pawnee and Thunder Basin Naitonal Grasslands and the surronding areas will become more suitable for smooth brome under different climate change scenarios. 

# %% [markdown]
# ### Step 0: library import and project path setup

# %%
## Libraries

# paths
import os
from glob import glob
import pathlib
from pathlib import Path

# gbif packages
import pygbif.occurrences as occ
import pygbif.species as species
from getpass import getpass

# unzipping
import zipfile
import time

# spatial data
import geopandas as gpd
import xrspatial
import fiona

# data
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrm
import xarray as xr
from math import floor, ceil
import skfuzzy as fuzz

# invalid geometries
from shapely.geometry import MultiPolygon, Polygon

# visualization
import holoviews as hc
import hvplot.pandas
import hvplot.xarray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # literally could not get the legend to behave without this

# api
import requests
import earthaccess

# %%
## file paths

data_dir = os.path.join(

    pathlib.Path.home(),
    "Documents",
    "Earth Data Cert",
    "Earth-Analytics-AY25",
    "GitRepos",
    "spring-03-habitat-suitability-climate-change-natshoff",
    'data'
)

os.makedirs(data_dir, exist_ok=True)

# %% [markdown]
# ### Step 1a: Load GBIF data for smooth brome

# %%
## gbif data dir
gbif_dir = os.path.join(data_dir, 'gbif_brin2')

os.makedirs(gbif_dir, exist_ok=True)

# %%
## GBIF login
reset_credentials = False

## gbif dictionary
credentials = dict(
    GBIF_USER = (input, 'GBIF username'),
    GBIF_PWD = (getpass, 'GBIF password'),
    GBIF_EMAIL = (input, 'GBIF email')
)

for env_variable, (prompt_func, prompt_text) in credentials.items():
    if reset_credentials and (env_variable in os.environ):
        os.environ.pop(env_variable)

    if not env_variable in os.environ:
        os.environ[env_variable] = prompt_func(prompt_text)

# %%
backbone = species.name_backbone(name="bromus inermis")
backbone

# %%
## species name
species_name = "Bromus inermis"

# species info from gbif
species_info = species.name_lookup(species_name,
                                   rank = 'SPECIES')

# first result
first_result = species_info['results'][0]
first_result

# %%
## species key
species_key = first_result['nubKey']

## check
first_result['species'], species_key

# %%
## file path
gbif_pattern = os.path.join(gbif_dir, '*.csv')

## download once
if not glob(gbif_pattern):

    # submit query
    gbif_query = occ.download([
        f"speciesKey = {species_key}",
        "hasCoordinate = True",
    ])

    # download once
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
        download_key = os.environ['GBIF_DOWNLOAD_KEY']

        # wait
        wait = occ.download_meta(download_key)['status']
        while not wait=='SUCCEEDED':
            wait = occ.download_meta(download_key)['status']
            time.sleep(5)

    ## download the data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'],
        path = data_dir
    )

    ## unzip the file
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path = gbif_dir)


## find csv path
gbif_path = glob(gbif_pattern)[0]

# %%
gbif_df = pd.read_csv(
    gbif_path,
    delimiter = '\t'
)

gbif_df.head()

# %%
## convert into spatial df
gbif_gdf = (
    gpd.GeoDataFrame(
        gbif_df,
        geometry=gpd.points_from_xy(
            gbif_df.decimalLongitude,
            gbif_df.decimalLatitude
        ),
        crs = 'EPSG:4326'
    )
)

# %%
## plot
gbif_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Smooth Brome Occurrence in GBIF',
    fill_color = None,
    line_color = 'orange',
    frame_width = 600
)

# %% [markdown]
# ***Figure 1:*** The distribution of smooth brome observation in the GBIF database. Smooth brome is a widespread species with observations recorded across it's native range of Eurasia, and introduced range in North America.

# %% [markdown]
# ### Step 1b: Select study sites
# I chose to compare suitability across two National Grasslands, Pawnee and Thunder Basin. Both are managed by the USDA Forest Service and represent diverse grassland/shrubland ecosystems. Noteably, both grasslands are highly fragmented with private parcels interspersed between USDA managed land. This type of fragmentation can make vroad invasive species management difficult, so I was especially curious to asses habitat suitability for smooth brome in areas around the national grasslands.

# %%
## site directory
site_dir = Path(data_dir) / "sites_COWY"
site_dir.mkdir(parents = True, exist_ok=True)

# %%
## item id and file names
item_id = "6759abcfd34edfeb8710a004" # this is the same for both

padnames = ["PADUS4_1_State_CO_GDB_KMZ.zip",
              "PADUS4_1_State_WY_GDB_KMZ.zip"]

# %%
def download_and_unzip_padus(item_id, padnames, site_dir):
    """
    Download and unzip PADUS GDB files from ScienceBase if not already downloaded.

    Args:
        item_id (str): ScienceBase item ID
        padnames (list): list of filenames to download
        site_dir (Path or str): directory to save files to

    Returns: None
    """
    site_dir = Path(site_dir)

    for filename in padnames:
        zip_path = site_dir / filename
        extract_folder = site_dir / zip_path.stem

        # check if already extracted
        if extract_folder.exists():
            print(f"Already downloaded and extracted: {filename}")
            continue

        # download if zip not already present
        if not zip_path.exists():
            url = f"https://www.sciencebase.gov/catalog/file/get/{item_id}?name={filename}"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        # unzip
        extract_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

    print("Done!")

# %%
## Execute download function
download_and_unzip_padus(
    item_id=item_id,
    padnames=padnames,
    site_dir=site_dir
)

# %%
## loop through folder and extract fee data layer for CO and WY
states = ["CO", "WY"]

pa_shp = {}

for state in states:
    pa_path = site_dir / f"PADUS4_1_State_{state}_GDB_KMZ" /f"PADUS4_1_State{state}.gdb"
    pa_shp[state] = gpd.read_file(pa_path, layer = f"PADUS4_1Fee_State_{state}")

pa_shp = gpd.GeoDataFrame(pd.concat(pa_shp.values(), ignore_index=True))

# %%
pa_shp.head(3)

# %%
## filter to just the national grasslands of interest
pa_shp = pa_shp[(pa_shp['Unit_Nm'].isin(['Thunder Basin National Grassland', 'Pawnee National Grassland'])) & (pa_shp['Own_Name'] == 'USFS')]
pa_shp

# %%
pa_shp = pa_shp.to_crs(epsg = 4326)

# %%
pa_shp.hvplot(
    geo=True,
    tiles='EsriImagery',
    title='National Grasslands of Interest',
    c='Unit_Nm',
    cmap=['#F4DBC7', '#BBBA88'],
    fill_alpha=0.3,
    line_color='Unit_Nm',
    line_alpha=1.0,
    legend=True,
    frame_width=600,
    frame_height=400
)

# %% [markdown]
# ***Figure 2:*** Pawnee and Thunder Basin National Grasslands. Note the high degree of fragmentation in each.

# %%
brin2_sites = gpd.overlay(gbif_gdf, pa_shp, how = 'intersection')

# %%
pd.set_option('display.max_rows', None)

# %%
# sum number of occurrence per site
value_counts = brin2_sites['Unit_Nm'].value_counts()
value_counts

# %%
## Make separate gdf for each site

# Thunder Basin (WY)
thu_gdf = pa_shp[pa_shp['Unit_Nm'] == 'Thunder Basin National Grassland']

# Pawnee (CO)
paw_gdf = pa_shp[pa_shp['Unit_Nm'] == 'Pawnee National Grassland']


# %% [markdown]
# **Pawnee National Grassland**
# Location: Northeastern Colorado
# Ecology: This area is primarily short (e.g. buffalo grass, blue grama) and mixed-grass 
# (western wheatgrass, needle-and-thread grass) prairie with additional vegetation types 
# (e.g. shrub steppe, scarp woodlands) at higher elevation areas 
# [(Hazlett, 1998)](https://research.fs.usda.gov/treesearch/25015).
# 
# **Thunder Basin National Grassland**
# Location: Northeastern Wyoming
# Ecology: This grassland is interesting because it lies along the ecotone between the 
# Great Plains (mixed-grass prairie) to the east and the shrub steppe to the west. This 
# follows a gradient of precipitation, temperature, and elevation. Typical grassland 
# species are blue grama, needle-and-thread, and western wheatgrass, while shrubs are 
# primarily Wyoming big sagebrush 
# [(Porensky & Blumenthal, 2016)](https://link.springer.com/article/10.1007/s10530-016-1225-z).
# 
# **Projections for the future**
# Both grasslands are highly fragmented and face risks from climate change and land 
# development. Because private parcels are intermixed within the grasslands, management 
# is difficult to apply across a broad scale. This may also make climate-adapted 
# management more challenging, as land managers will need to contend with both climate 
# variability and changes in land use from private owner development. Land development 
# also represents a vector for invasive species introduction and establishment, as species 
# can be transported on construction equipment [(Westbrooks, 1998)](https://digitalcommons.usu.edu/govdocs/490/), or outcompete natives after 
# disturbance [(Hobbs & Huenneke, 1992)](https://conbio.onlinelibrary.wiley.com/doi/abs/10.1046/j.1523-1739.1992.06030324.x).

# %% [markdown]
# ### Step 1c: Select time periods
# 
# I chose the time periods **2006-2035** and **2036-2070**. I was curious to compare a historical baseline (2006-2035) to a management relevent projection in the near future (2036-2070). 

# %% [markdown]
# ### Step 1d: Select climate models
# 
# **Climate model selection**
# I used the [INHABIT tool](https://gis.usgs.gov/inhabit/) from the USGS [(Jarnavich et at., 2024)](https://neobiota.pensoft.net/article/134842/) to determine what environmental factors were most associated with high abundance of smooth brome. The INHABIT tool uses species distribution models to predict habitat suitability for various invasive species. This is similar to our fuzzy model approach but more robust, as these models (MAXNET, BRT, GLM, MARS, RF) are trained using actual plant observations, instead of us manually determining environmental tolerance from the literature.
# 
# For the maximum entropy (MAXENT) model, growing season precipitation and mean min winter temperature were the most important for predicting smooth brome high abundance. 
# ![image.png](images\inhabit_vars.png). Therefore, I tried to chose climate scenarios using the [Climate Toolbox](https://climatetoolbox.org/tool/Future-Climate-Scatter) that captured the most variation in these characteristics (Hegewisch & Abatzoglou, Future Climate Scatter web tool).
# 
# I chose the following models to try and capture hot/dry, hot/wet, cold/dry, and cold/wet climate futures. Note: the models for cold/dry and cold/wet are swapped for Pawnee and Thunder Basin, as they perform differently between these two locations. See figure 3.
# 
# | Future Type | Climate Model |
# |---|---|
# | Hot/Dry | HadGEM2-ES365 |
# | Hot/Wet | MIROC-ESM |
# | Cold/Dry | IPSL-CM5A-MR (Pawnee) / MRI-CGCM3 (Thunder Basin) |
# | Cold/Wet | IPSL-CM5A-MR (Thunder Basin) / MRI-CGCM3 (Pawnee) |

# %%
### Choosen climate models

# read in csv
paw_df = pd.read_csv('pawnee_data_subset.csv', skiprows=[1], index_col=0)
thu_df = pd.read_csv('thunder_data_subset.csv', skiprows=[1], index_col=0)

# convert precip and temp to numeric values
for df in [paw_df, thu_df]:
    df['X-VALUE'] = pd.to_numeric(df['X-VALUE'])
    df['Y-VALUE'] = pd.to_numeric(df['Y-VALUE'])

x_label = 'Dec-Jan-Feb Daily Minimum Temperature (°C)'
y_label = 'Mar-Apr-May Total Precipitation (mm)'

# color assignments per site
# assigned the mean_future and mean_historic different colors
site_colors = {
    'paw': {
        'model':            '#F4DBC7',   # base pawnee color
        'mean_historical':  '#A0522D',   # dark brown
        'mean_future':      '#FAF0E6',   # light cream
    },
    'thu': {
        'model':            '#BBBA88',   # base thunder color
        'mean_historical':  '#4A4A2A',   # dark olive
        'mean_future':      '#E2E2C8',   # light sage
    }
}

def add_color_group(df, colors):
    """Assign a color to each row based on model and scenario."""
    def assign(row):
        if row['MODEL'] != '20CMIP5ModelMean':
            return colors['model']
        elif row['SCENARIO'] == 'HISTORICAL':
            return colors['mean_historical']
        else:
            return colors['mean_future']
    df = df.copy()
    df['color'] = df.apply(assign, axis=1)
    return df

paw_df = add_color_group(paw_df, site_colors['paw'])
thu_df = add_color_group(thu_df, site_colors['thu'])

def make_scatter(df, title):
    """Overlay one scatter layer per color group so hvplot respects literal colors."""
    layers = []
    for color, group_df in df.groupby('color'):
        layers.append(
            group_df.hvplot.scatter(
                x='X-VALUE', y='Y-VALUE',
                color=color,
                line_color='black',
                size=120,
                hover_cols=['MODEL', 'SCENARIO', 'TIME PERIOD'],
                xlabel=x_label,
                ylabel=y_label,
                title=title,
                frame_width=400,
                frame_height=350
            )
        )
    return layers[0] * layers[1] * layers[2] if len(layers) == 3 else layers[0]

paw_plot = make_scatter(paw_df, 'Pawnee National Grassland \nClimate Model Projections \nRCP 8.5 Emissions Scenario')
thu_plot = make_scatter(thu_df, 'Thunder Basin National Grassland \nClimate Model Projections \nRCP 8.5 Emissions Scenario')

paw_plot + thu_plot

# %% [markdown]
# ***Figure 3:*** Selected climate models for each site. Note that the historical mean (1971-2000) for each site is shown in a darker color, and the future mean of 20 CMIP5 climate models (2040-2069) is shown in a lighter color for each site. Climate models were selected to represent hot/dry, hot/wet, cold/dry, and cold/wet climate futures.

# %% [markdown]
# ## STEP 2: Data access
# 
# ### Step 2a: Soil data
# I used the [POLARIS dataset](http://hydrology.cee.duke.edu/POLARIS/) for pH and bulk density at a depth of 5-15 cm (per the rooting depth noted in [Gist & Smith, 1948](https://acsess.onlinelibrary.wiley.com/doi/abs/10.2134/agronj1948.00021962004000110008x)). This is also consistent with personal experience digging up plant rhizomes of this species. It should be noted that roots can extend as deep as 3 meters in some places [(USDA)](https://www.fs.usda.gov/database/feis/plants/graminoid/broine/all.html#52).

# %%
## function for getting urls for the soil data from POLARIS
def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):
    """
    Function for generating urls to download multiple soil characteristics from POLARIS

    Args:
    soil_prop (str): soil property we want
    stat (str): statistic (e.g. mean, median, max)
    soil_depth (str): soil depth in cm
    gdf_bounds: array of site boundaries

    Returns:
    list: a list of POLARIS urls we can loop through
    """

    ## extract bounds
    min_lon, min_lat, max_lon, max_lat = gdf_bounds

    ## snap boundaries to whole degrees
    site_min_lon = floor(min_lon)
    site_min_lat = floor(min_lat)
    site_max_lon = ceil(max_lon)
    site_max_lat = ceil(max_lat)

    ## output list
    all_soil_urls = []

    ## loop through lat/long to get tiles
    for lon in range(floor(site_min_lon), ceil(site_max_lon)):
        for lat in range(floor(site_min_lat), ceil(site_max_lat)):

            ## define the corners
            current_max_lon = lon + 1
            current_max_lat = lat + 1

            ## url template
            url_template = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
                # soil_prop
                "{soil_prop}/"
                # stat
                "{stat}/"
                # depth
                "{soil_depth}/"
                # gdf_bounds
                "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"
            )

            ## fill in the template
            soil_url = url_template.format(
                soil_prop = soil_prop,
                stat = stat,
                soil_depth = soil_depth,
                min_lat = lat, max_lat = current_max_lat,
                min_lon = lon, max_lon = current_max_lon
            )

            ## add urls to list
            all_soil_urls.append(soil_url)

    return all_soil_urls


# %%
## for loop for pulling multiple sites and properties

# sites
sites = {
    'thu': thu_gdf,
    'paw': paw_gdf
}

# props
soil_props = ['ph', 'bd']

soil_urls = {}

for site_name, gdf in sites.items():
    soil_urls[site_name] = {}
    for prop in soil_props:
        soil_urls[site_name][prop] = create_polaris_urls(
            soil_prop=prop,
            stat='mean',
            soil_depth='5_15',
            gdf_bounds=gdf.total_bounds
        )

# %%
soil_urls

# %%
def build_da(urls, bounds):

    """
    Download data and convert to a data array for the list of urls

    Args:
    urls (list): list of polaris urls
    bounds (tuple): site boundaries

    Returns:
    xarray.DataArray: merged dataarray
    """

    all_das = []

    # buffer
    buffer = 0.025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    # process urls sequentially
    for url in urls:

        # open raster, masking missing data, remove extra dims
        tile_da = rxr.open_rasterio(url,
                                    mask_and_scale=True).squeeze()
        
        # crop the tile buffered boundaries
        cropped_da  = tile_da.rio.clip_box(*bounds_buffer)

        # store cropped tile
        all_das.append(cropped_da)

    # combine into a single raster
    merged = rxrm.merge_arrays(all_das)

    # return final raster
    return merged

# %%
soil_das = {}

for site_name, gdf in sites.items():
    soil_das[site_name] = {}
    for prop, urls in soil_urls[site_name].items():
        soil_das[site_name][prop] = build_da(urls, gdf.total_bounds)

# %%
soil_das

# %%
prop_labels = {'ph': 'Soil pH', 'bd': 'Bulk Density (g/cm³)'}
prop_cmaps = {'ph': 'viridis', 'bd': 'YlOrBr'}

for prop in soil_props:
    thu_da = soil_das['thu'][prop]
    paw_da = soil_das['paw'][prop]

    vmin = min(float(thu_da.min()), float(paw_da.min()))
    vmax = max(float(thu_da.max()), float(paw_da.max()))
    cmap = prop_cmaps[prop]

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    thu_da.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    thu_gdf_reproj = thu_gdf.to_crs(thu_da.rio.crs)
    thu_gdf_reproj.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)
    xmin, ymin, xmax, ymax = thu_gdf_reproj.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(f'Thunder Basin - {prop_labels[prop]} (5-15 cm)')

    img = paw_da.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    paw_gdf_reproj = paw_gdf.to_crs(paw_da.rio.crs)
    paw_gdf_reproj.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1)
    xmin, ymin, xmax, ymax = paw_gdf_reproj.total_bounds
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_title(f'Pawnee - {prop_labels[prop]} (5-15 cm)')

    fig.colorbar(img, cax=cax, orientation='horizontal', label=prop_labels[prop])
    plt.show()

# %% [markdown]
# ***Figure 4:*** Soil characteristics (pH and bulk density) for each site. Note the alkeline soils near Pawnee National Grassland.

# %%
## define soils directory for us to output rasters to
soils_dir = os.path.join(site_dir, 'soils_dir')

os.makedirs(soils_dir, exist_ok=True)

# %%
## function save xarray.DataArray as a raster
def export_raster(da, raster_path):
    """
    Export raster to file

    Args:
    da (xarray.DataArray): input raster layer
    raster_path (str or Path): full output file path

    Returns: None
    """
    os.makedirs(os.path.dirname(raster_path), exist_ok=True)
    
    da = da.copy()
    da.attrs.pop('_FillValue', None)
    da.encoding.pop('_FillValue', None)
    
    da.rio.to_raster(raster_path)

# %%
## save all rasters out
for site_name, props in soil_das.items():
    for prop, da in props.items():
        raster_path = os.path.join(soils_dir, site_name, f"soil_{prop}_{site_name}.tif")
        export_raster(da=da, raster_path=raster_path)

# %% [markdown]
# ### Step 2b: Topographic data
# 
# I pulled elevation, aspect, and slope from the [NASA Shuttle Radar Topography Mission](https://www.earthdata.nasa.gov/data/catalog/lpcloud-srtmgl3-003), although I drop aspect and slope in the final suitabiltiy assignment as they are very uniform for these grassland sites. 

# %%
### elevation directory
elev_dir = os.path.join(data_dir, "topography")
os.makedirs(elev_dir, exist_ok=True)

# %%
# earthaccess login
earthaccess.login()

# %%
## SRTM data
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")

for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
def get_topo_data(site_name, site_gdf, topo_dir):
    """
    Download SRTM elevation data and derive slope and aspect rasters for a site.

    Args:
    site_name (str): short name for the site (e.g. 'thu', 'paw')
    site_gdf (GeoDataFrame): site boundary GeoDataFrame
    topo_dir (Path or str): directory to store downloaded topo data

    Returns:
    dict: {'elevation': DataArray, 'slope': DataArray, 'aspect': DataArray}
    """

    site_topo_dir = os.path.join(topo_dir, site_name)
    os.makedirs(site_topo_dir, exist_ok=True)

    srtm_pattern = os.path.join(site_topo_dir, "*.hgt.zip")

    # buffer bounds
    buffer = 0.025
    xmin, ymin, xmax, ymax = tuple(site_gdf.total_bounds)
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    # download if not already present
    if not glob(srtm_pattern):
        srtm_search = earthaccess.search_data(
            short_name='SRTMGL3',
            bounding_box=bounds_buffer
        )
        earthaccess.download(srtm_search, site_topo_dir)
    else:
        print(f"{site_name}: Already downloaded!")

    # open and merge tiles
    da_list = []
    for srtm_path in glob(srtm_pattern):
        tile_da = rxr.open_rasterio(srtm_path, mask_and_scale=True).squeeze()
        da_list.append(tile_da.rio.clip_box(*bounds_buffer))
    elev_da = rxrm.merge_arrays(da_list)

    # aspect (calculated in geographic CRS)
    aspect_da = xrspatial.aspect(elev_da)

    # slope (requires equal-area projection)
    elev_rpj = elev_da.rio.reproject('EPSG:5070')
    slope_da = xrspatial.slope(elev_rpj).rio.reproject('EPSG:4326')

    return {'elevation': elev_da, 'slope': slope_da, 'aspect': aspect_da}

# %%
topo_das = {}
for site_name, gdf in sites.items():
    topo_das[site_name] = get_topo_data(site_name, gdf, elev_dir)

# %%
topo_das

# %%
topo_labels = {
    'elevation': 'Elevation (m)',
    'slope': 'Slope (degrees)',
    'aspect': 'Aspect (degrees)'
}

topo_cmaps = {
    'elevation': 'terrain',
    'slope': 'copper',
    'aspect': 'twilight'
}

for topo_var in topo_labels:
    thu_da = topo_das['thu'][topo_var]
    paw_da = topo_das['paw'][topo_var]

    vmin = min(float(thu_da.min()), float(paw_da.min()))
    vmax = max(float(thu_da.max()), float(paw_da.max()))
    cmap = topo_cmaps[topo_var]

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    thu_da.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    thu_gdf_reproj = thu_gdf.to_crs(thu_da.rio.crs)
    thu_gdf_reproj.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)
    xmin, ymin, xmax, ymax = thu_gdf_reproj.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(f'Thunder Basin - {topo_labels[topo_var]}')

    img = paw_da.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    paw_gdf_reproj = paw_gdf.to_crs(paw_da.rio.crs)
    paw_gdf_reproj.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1)
    xmin, ymin, xmax, ymax = paw_gdf_reproj.total_bounds
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_title(f'Pawnee - {topo_labels[topo_var]}')

    fig.colorbar(img, cax=cax, orientation='horizontal', label=topo_labels[topo_var])
    plt.show()

# %% [markdown]
# ***Figure 5:*** Topgraphy characteristics (elevation, slope, and aspect) for each site. Pawnee is higher elevation overall but Thunder Basin exists across a greater range of elevations.

# %%
## save all rasters out
for site_name, props in topo_das.items():
    for prop, da in props.items():
        raster_path = os.path.join(elev_dir, site_name, f"topo_{prop}_{site_name}.tif")
        export_raster(da=da, raster_path=raster_path)

# %% [markdown]
# ### Step 2c: Climate model data
# 
# I pulled monthly precipitation and minimum temperature data from the following climate models for 2006-2035 and 2036-2070 from the [MACAv2](http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html) dataset.
# 
# | Future Type | Climate Model |
# |---|---|
# | Hot/Dry | HadGEM2-ES365 |
# | Hot/Wet | MIROC-ESM |
# | Cold/Dry | IPSL-CM5A-MR (Pawnee) / MRI-CGCM3 (Thunder Basin) |
# | Cold/Wet | IPSL-CM5A-MR (Thunder Basin) / MRI-CGCM3 (Pawnee) |
# 
# I then calculated growing season precipitation by summing precipitation across Apr-Oct for each year and taking the mean across years for each time period. I calculated mean min winter temperature by taking the mean of Dec-Feb minimum temperatures for each time period. 
# 

# %%
### Download climate data
maca_dir = os.path.join(data_dir, 'maca_dir')
os.makedirs(maca_dir, exist_ok=True)

maca_pattern = os.path.join(maca_dir, ".nc")
maca_pattern

# %%
## temperature conversion function
def convert_temperature(temp):
    """
    Convert Kelvin to celcius

    Args:
    temp (int): temperature in K

    Returns:
    temp (int): temperature in C
    """

    return temp - 273.15    

# %%
### convert longitude function
def convert_longitude(longitude):
    """
    Convert longitude 

    Args:
    longitude (int): maca longitude (0 to 360)

    Returns:
    longitude (int): norm longitude (-180 to 180)
    """
    
    return(longitude  - 360) if longitude > 180 else longitude

# %%
def get_maca_date_ranges(start_year, end_year):
    """
    Generate MACA-style date range strings between two years,
    accounting for irregular intervals at the historical/future boundary.
    Includes any interval that overlaps with the requested range.

    Args:
        start_year (int): first year of the desired range
        end_year (int): last year of the desired range

    Returns:
        list: list of MACA date range strings
    """
    
    all_intervals = (
        [(y, y + 4) for y in range(1950, 2001, 5)]
        + [(2005, 2005)]
        + [(y, y + 4) for y in range(2006, 2096, 5)]
        + [(2096, 2099)]
    )

    # include interval if it overlaps at all with requested range
    return [
        f"{s}_{e}"
        for s, e in all_intervals
        if s <= end_year and e >= start_year
    ]

# %%
def download_maca_da(site_dict, years_list, models_list, rcp_value, climVars_list, maca_dir):
    """
    Download and process MACA climate data for multiple sites, time periods, models, and variables.

    Args:
        site_dict (dict): site names mapped to GeoDataFrames (e.g. {'thu': thu_gdf, 'paw': paw_gdf})
        years_list (list): date range strings (e.g. ['2041_2045', '2071_2075'])
            - Use get_maca_date_ranges() to generate these (e.g. get_maca_date_ranges(1970, 2030))
        models_list (list): climate model names (e.g. ['MIROC-ESM', 'IPSL-CM5A-MR'])
        rcp_value (str): RCP scenario (e.g. 'rcp85')
            - Note: historical data are automatically pulled if a year range <= 2005 is specified
        climVars_list (list): climate variable names (e.g. ['pr', 'tasmin'])
        maca_dir (str): directory to save downloaded files

    Returns:
        list: list of dicts, each containing cropped and processed climate DataArray with metadata
    """

    results = []

    # for loop for each site
    for site_name, site_gdf in site_dict.items():

        # get site bounds in EPSG:4326 for spatial subsetting
        site_4326 = site_gdf.to_crs('EPSG:4326')
        xmin, ymin, xmax, ymax = site_4326.total_bounds

        # add buffer around site bounds
        buffer = 0.025
        xmin, ymin, xmax, ymax = xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer

        # convert lon bounds to 0-360 for MACA subsetting
        lon_min_360 = xmin + 360
        lon_max_360 = xmax + 360

        # for loop for each date
        for date_range in years_list:
            
            # for loop for each model
            for model in models_list:

                # for loop for each clim variable
                for clim_var in climVars_list:

                    # auto-select rcp based on date range
                    year_start = int(date_range.split('_')[0])
                    effective_rcp = 'historical' if year_start <= 2005 else rcp_value

                    # define local path
                    maca_path = os.path.join(
                        maca_dir,
                        f"maca_{model}_{site_name}_{clim_var}_{effective_rcp}_{date_range}_CONUS_monthly.nc"
                    )

                    # define url
                    maca_url = (
                        "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
                        "MACAV2/"
                        f"{model}/"
                        f"macav2metdata_{clim_var}"
                        f"_{model}_r1i1p1_"
                        f"{effective_rcp}_"
                        f"{date_range}_"
                        "CONUS_monthly.nc"
                    )

                    # download only if not already saved
                    if not os.path.exists(maca_path):
                        ds = xr.open_dataset(maca_url).squeeze()
                        da_raw = ds[list(ds.data_vars)[0]]

                        # subset to site bounding box using OPeNDAP before downloading
                        da_subset = da_raw.sel(
                            lat=slice(ymin, ymax),
                            lon=slice(lon_min_360, lon_max_360)
                        )

                        da_subset.to_netcdf(maca_path)
                        print(f"Downloaded: {os.path.basename(maca_path)}")
                    else:
                        print(f"Already exists: {os.path.basename(maca_path)}")

                    # open saved subset
                    ds = xr.open_dataset(maca_path).squeeze()
                    maca_da = ds[list(ds.data_vars)[0]]

                    # convert longitude from 0-360 to -180-180
                    maca_da = maca_da.assign_coords(
                        lon=("lon", [convert_longitude(l) for l in maca_da.lon.values])
                    )

                    # set spatial dims
                    maca_da = maca_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

                    # clip to exact site bounds
                    site_rpj = site_gdf.to_crs(maca_da.rio.crs)
                    maca_da_cropped = maca_da.rio.clip_box(*site_rpj.total_bounds)

                    # convert temperature from K to C for temp variables
                    if clim_var in ['tasmin', 'tasmax']:
                        maca_da_cropped = convert_temperature(maca_da_cropped)

                    results.append(dict(
                        site_name=site_name,
                        climate_model=model,
                        climate_var=clim_var,
                        date_range=date_range,
                        rcp=effective_rcp,
                        da=maca_da_cropped
                    ))

    return results

# %%
## Get two time ranges

## sequential 30 year blocks
time_range = get_maca_date_ranges(2006,2066)

## choosing non-sequential ranges
#time_range = (get_maca_date_ranges(1970, 2005) + 
#get_maca_date_ranges(2030, 2060))

time_range

# %%
## define lists and variables

site_dict = {'thu': thu_gdf, 'paw': paw_gdf}
years_list = time_range
models_list = ['MIROC-ESM','IPSL-CM5A-MR','MRI-CGCM3','HadGEM2-ES365']
rcp_value = 'rcp85'
climVars_list = ['pr','tasmin']

# %%
maca_results = download_maca_da(
    site_dict=site_dict,
    years_list=years_list,
    models_list=models_list,
    rcp_value=rcp_value,
    climVars_list=climVars_list,
    maca_dir=maca_dir
)

# %%
maca_results

# %%
def calc_growing_season_precip(das):
    """
    Mean annual cumulative growing season (Apr-Oct) precip across a period.
    Args:
        das: list of monthly precip DataArrays
    Returns:
        2D DataArray (spatial mean of annual sums)
    """
    combined = xr.concat(das, dim='time').sortby('time')
    # keep only April-October
    gs = combined.sel(time=combined.time.dt.month.isin(range(4, 11)))
    # sum within each year, then mean across years
    result = gs.resample(time='YE').sum().mean(dim='time')
    result = result.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    if result.rio.crs is None:
        result = result.rio.write_crs('EPSG:4326')
    return result


# %%
def calc_mean_winter_tasmin(das):
    """
    Mean winter (Dec-Feb) min temperature across a period.
    Args:
        das: list of monthly tasmin DataArrays
    Returns:
        2D DataArray (spatial mean of seasonal means)
    """
    combined = xr.concat(das, dim='time').sortby('time')
    # keep only December, January, February
    winter = combined.sel(time=combined.time.dt.month.isin([12, 1, 2]))
    # mean within each year, then mean across years
    result = winter.resample(time='YE').mean().mean(dim='time')
    result = result.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    if result.rio.crs is None:
        result = result.rio.write_crs('EPSG:4326')
    return result


# %%
# define periods
period1_ranges = set(get_maca_date_ranges(2006, 2035))
period2_ranges = set(get_maca_date_ranges(2036, 2070))

periods = {
    '2006-2035': period1_ranges,
    '2036-2070': period2_ranges
}

sites = list({r['site_name'] for r in maca_results})
models = list({r['climate_model'] for r in maca_results})

# %%
# store results: derivedClim_das[period_label][site][model][var_label]
derivedClim_das = {}

for period_label, date_ranges in periods.items():
    derivedClim_das[period_label] = {}
    for site in sites:
        derivedClim_das[period_label][site] = {}
        for model in models:
            derivedClim_das[period_label][site][model] = {}

            # --- growing season precip ---
            pr_das = [
                r['da'] for r in maca_results
                if r['site_name'] == site
                and r['climate_model'] == model
                and r['climate_var'] == 'pr'
                and r['date_range'] in date_ranges
            ]
            if pr_das:
                gs_precip = calc_growing_season_precip(pr_das)
                derivedClim_das[period_label][site][model]['gs_precip'] = gs_precip
                raster_path = os.path.join(
                    maca_dir, site, model,
                    f"gs_precip_{site}_{model}_{period_label}.tif"
                )
                export_raster(gs_precip, raster_path)

            # --- mean winter min temp ---
            tasmin_das = [
                r['da'] for r in maca_results
                if r['site_name'] == site
                and r['climate_model'] == model
                and r['climate_var'] == 'tasmin'
                and r['date_range'] in date_ranges
            ]
            if tasmin_das:
                winter_temp = calc_mean_winter_tasmin(tasmin_das)
                derivedClim_das[period_label][site][model]['winter_tasmin'] = winter_temp
                raster_path = os.path.join(
                    maca_dir, site, model,
                    f"winter_tasmin_{site}_{model}_{period_label}.tif"
                )
                export_raster(winter_temp, raster_path)

# %%
derived_var_labels = {
    'gs_precip': 'Mean Cumulative Growing Season Precip (mm)',
    'winter_tasmin': 'Mean Winter Min Temperature (°C)'
}
derived_var_cmaps = {
    'gs_precip': 'viridis',
    'winter_tasmin': 'plasma'
}

period_labels = list(periods.keys())
row_combos = [(site, model) for site in sites for model in models]
n_rows = len(row_combos)
n_cols = len(period_labels)

for var_label, cbar_label in derived_var_labels.items():

    # collect all DataArrays to compute shared color scale
    all_das = [
        derivedClim_das[p][site][model][var_label]
        for p in period_labels
        for site, model in row_combos
        if var_label in derivedClim_das[p][site][model]
    ]

    if not all_das:
        continue

    vmin = min(float(da.min()) for da in all_das)
    vmax = max(float(da.max()) for da in all_das)
    cmap = derived_var_cmaps[var_label]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.reshape(n_rows, n_cols)

    for row_idx, (site, model) in enumerate(row_combos):
        for col_idx, period_label in enumerate(period_labels):

            ax = axes[row_idx, col_idx]
            da = derivedClim_das.get(period_label, {}).get(site, {}).get(model, {}).get(var_label)

            if da is None:
                ax.set_visible(False)
                continue

            da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
            site_dict[site].to_crs(da.rio.crs).plot(
                ax=ax, facecolor='none', edgecolor='white', linewidth=1
            )
            ax.set_title(f'{site.upper()} | {model}\n{period_label}')
            ax.set_xlabel('')
            ax.set_ylabel('')

    fig.suptitle(cbar_label, fontsize=13, y=1.01)
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap
        ),
        ax=axes,
        orientation='horizontal',
        label=cbar_label,
        shrink=0.5,
        pad=0.05
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ***Figure 5:*** Climate characteristics (winter mean min temperature and mean growing season precipitation) for each site and four climate models. 

# %% [markdown]
# **Differences across sites**
# As mentioned earlier, Thunder Basin spans a precipitation and temperature gradient (wetter/cooler in the east, drier/warmer in the west). This gradient is notable across different climate models. This contrasts with Pawnee, which is more homogenous across the entire site.
# 
# **Differences across models**
# | Future Type | Climate Model |
# |---|---|
# | Hot/Dry | HadGEM2-ES365 |
# | Hot/Wet | MIROC-ESM |
# | Cold/Dry | IPSL-CM5A-MR (Pawnee) / MRI-CGCM3 (Thunder Basin) |
# | Cold/Wet | IPSL-CM5A-MR (Thunder Basin) / MRI-CGCM3 (Pawnee) |
# 
# The effect of the hot/dry (HadGEM2-ES365) and hot/wet (MIROC-ESM) models are clear across both sites, but most evident when comparing temperatures between time periods. It's important to note that even the cold climate models show temperature increases for the 2036-2070 time period. See above for a description of how I chose these specific climate models.

# %% [markdown]
# ## STEP 3: Harmonize data
# 
# I harmonized all climate, topographic, and climate data for each site. I opted to use the highest resolution data (POLARIS pH, 30m) as my template.

# %%
## List of available data layers
soil_das
topo_das
derivedClim_das

# %%
def harmonize_site_das(soil_das, topo_das, derivedClim_das, sites):
    """
    Reproject and align soil, topo, and derived climate DataArrays to a 
    common grid per site. Uses soil pH as the reference grid.

    Args:
        soil_das (dict): {site: {prop: DataArray}}
        topo_das (dict): {site: {var: DataArray}}
        derivedClim_das (dict): {period: {site: {model: {var_label: DataArray}}}}
        sites (list): site names to process

    Returns:
        dict: {
            site: {
                'soil':    {prop: DataArray},
                'topo':    {var: DataArray},
                'climate': {period: {model: {var_label: DataArray}}}
            }
        }
    """
    harmonized = {}

    for site in sites:
        print(f"\nHarmonizing site: {site}")

        # use soil pH as reference grid
        ref_da = soil_das[site]['ph']

        print(f"  Reference grid: {ref_da.rio.crs}, shape={ref_da.shape}, res={ref_da.rio.resolution()}")

        harmonized[site] = {'soil': {}, 'topo': {}, 'climate': {}}

        # soil: skip reproject for the reference layer itself
        for prop, da in soil_das.get(site, {}).items():
            harmonized[site]['soil'][prop] = da if da is ref_da else da.rio.reproject_match(ref_da)
            print(f"  soil/{prop}: shape={harmonized[site]['soil'][prop].shape}")

        # topo
        for var, da in topo_das.get(site, {}).items():
            harmonized[site]['topo'][var] = da.rio.reproject_match(ref_da)
            print(f"  topo/{var}: reprojected to {harmonized[site]['topo'][var].shape}")

        # climate
        for period, period_data in derivedClim_das.items():
            if site not in period_data:
                continue
            harmonized[site]['climate'][period] = {}
            for model, model_data in period_data[site].items():
                harmonized[site]['climate'][period][model] = {}
                for var_label, da in model_data.items():
                    harmonized[site]['climate'][period][model][var_label] = da.rio.reproject_match(ref_da)
                    print(f"  climate/{period}/{model}/{var_label}: reprojected to {harmonized[site]['climate'][period][model][var_label].shape}")

    return harmonized

# %%
sites = list({r['site_name'] for r in maca_results})
harmonized_das = harmonize_site_das(soil_das, topo_das, derivedClim_das, sites)

# %%
for site, site_data in harmonized_das.items():

    # pick one representative from each data type
    soil_prop = list(site_data['soil'].keys())[0]
    topo_var  = list(site_data['topo'].keys())[0]
    first_period = list(site_data['climate'].keys())[0]
    first_model  = list(site_data['climate'][first_period].keys())[0]
    clim_var     = list(site_data['climate'][first_period][first_model].keys())[0]

    layers = {
        f'soil: {soil_prop}':  site_data['soil'][soil_prop],
        f'topo: {topo_var}':   site_data['topo'][topo_var],
        f'climate: {clim_var}\n({first_period} | {first_model})':
            site_data['climate'][first_period][first_model][clim_var]
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Harmonization check — {site.upper()}', fontsize=13)

    site_gdf = site_dict[site]

    for ax, (label, da) in zip(axes, layers.items()):
        da.plot(ax=ax, add_colorbar=True, robust=True)
        site_gdf.to_crs(da.rio.crs).plot(
            ax=ax, facecolor='none', edgecolor='white', linewidth=1
        )
        ax.set_title(label)
        ax.set_xlabel(''); ax.set_ylabel('')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ***Figure 6:*** Data harmonization check to ensure no issues.

# %%
print(harmonized_das['thu']['soil']['ph'].rio.bounds())
print(harmonized_das['thu']['topo']['elevation'].rio.bounds())
print(harmonized_das['thu']['climate']['2006-2035']['MIROC-ESM']['gs_precip'].rio.bounds())

# %% [markdown]
# ## STEP 4: Develop a fuzzy logic model
# 
# I developed suitability thresholds for each environmental variable as follows:
# 
# | Characteristic | Optimal Range | Tolerance Range | Membership | Sources | Notes |
# |---|---|---|---|---|---|
# | Soil pH | 6.0 – 7.0 | 4.5 – 8.0 | Trapezoid | [Penn State Extension](https://extension.psu.edu/smooth-bromegrass); [FEIS](https://www.fs.usda.gov/database/feis/plants/graminoid/broine/all.html); [USDA PLANTS](https://plants.usda.gov/plant-profile/BRIN2/characteristics) | Lower bound from FEIS; general tolerance from USDA PLANTS |
# | Bulk Density | 0.1 – 1.4 g/cm³ | > 1.55 g/cm³ restricts root growth | Trapezoid | [MU Extension](https://extension.missouri.edu/publications/g4672); [USDA PLANTS](https://plants.usda.gov/plant-profile/BRIN2/characteristics); [NRCS Soil Health Guide](https://www.nrcs.usda.gov/sites/default/files/2022-11/Bulk%20Density%20-%20Soil%20Health%20Guide_0.pdf) | Soil type (silt/clay loam) converted to bulk density using NRCS guide |
# | Elevation | 0 – 3235 m | Cannot survive above 3235 m | Trapezoid | [FEIS](https://www.fs.usda.gov/database/feis/plants/graminoid/broine/all.html) | Upper limit based on highest recorded occurrence in Utah; no distinct optimal identified |
# | Growing Season Precip (Apr–Oct) | 381 – 1524 mm | 279 mm (min) – 1524 mm (max) | Trapezoid | [FEIS](https://www.fs.usda.gov/database/feis/plants/graminoid/broine/all.html); [USDA PLANTS](https://plants.usda.gov/plant-profile/BRIN2/characteristics) | Lower optimal and tolerance from FEIS; upper tolerance from USDA PLANTS |
# | Winter Min Temp (Dec–Feb) | > -39°C | Cannot survive below -41.67°C; no upper limit | Linear ramp | [USDA PLANTS](https://plants.usda.gov/plant-profile/BRIN2/characteristics) | No distinct optimal found; suitability increases linearly above lower tolerance threshold |

# %%
### Membership functions

# I wanted to use this based on INHABIT data but the site was down - instead pivoted to trapezoid
def gaussian_membership(da, optimal, tolerance):
    """
    Gaussian fuzzy membership function.
    Returns a DataArray with values 0-1, where 1 = optimal.

    Args:
        da (xr.DataArray): input raster
        optimal (float): optimal value for the species
        tolerance (float): spread (std dev); controls how quickly
                           suitability drops off from the optimum
    Returns:
        xr.DataArray: suitability scores 0-1
    """
    return np.exp(-((da - optimal) ** 2) / (2 * tolerance ** 2))


# Trapezoid
def trapezoid_membership(da, low_tolerance, low_optimal, high_optimal, high_tolerance):
    """
    Trapezoidal fuzzy membership function.
    Scores 1.0 within [low_optimal, high_optimal],
    ramps linearly to 0 at low_tolerance and high_tolerance.

    Args:
        da (xr.DataArray): input raster
        low_tolerance (float): lower edge of tolerance range (score = 0)
        low_optimal (float): lower edge of optimal range   (score = 1)
        high_optimal (float): upper edge of optimal range   (score = 1)
        high_tolerance (float): upper edge of tolerance range (score = 0)
    Returns:
        xr.DataArray: suitability scores 0-1
    """
    x_range = np.linspace(float(da.min()), float(da.max()), 1000)
    membership = fuzz.trapmf(x_range, [low_tolerance, low_optimal, high_optimal, high_tolerance])
    scores = np.interp(da.values, x_range, membership)
    return da.copy(data=scores)


# Linear ramp for temperature
def linear_ramp_membership(da, low_tol, high_val=None):
    """
    Linear ramp membership: 0 at low_tol, increases linearly to 1.0 at high_val.
    Values below low_tol are clipped to 0, values above high_val clipped to 1.

    Args:
        da (xr.DataArray): input raster
        low_tol (float): value at which score = 0 (lower cutoff)
        high_val (float or None): value at which score reaches 1.0.
                                  If None, uses the raster maximum.
    Returns:
        xr.DataArray: suitability scores 0-1
    """
    if high_val is None:
        high_val = float(da.max())
    scores = (da - low_tol) / (high_val - low_tol)
    return scores.clip(0, 1)


# Apply membership
def apply_membership(da, params):
    if params['type'] == 'gaussian':
        return gaussian_membership(da, params['optimal'], params['tolerance'])
    elif params['type'] == 'trapezoid':
        return trapezoid_membership(
            da, params['low_tol'], params['low_opt'],
            params['high_opt'], params['high_tol']
        )
    elif params['type'] == 'linear_ramp':
        return linear_ramp_membership(da, params['low_tol'], params.get('high_val'))
    else:
        raise ValueError(f"Unknown membership type: {params['type']}")


# Fuzzy params from the table
fuzzy_params = {
    # pH: classic trapezoid
    'ph': {
        'type': 'trapezoid',
        'low_tol': 4.5, 'low_opt': 6.0, 'high_opt': 7.0, 'high_tol': 8.0
    },
    # bulk density: one-sided upper, lower floor at 0.1 to avoid artifacts
    'bd': {
        'type': 'trapezoid',
        'low_tol': 0.1, 'low_opt': 0.1, 'high_opt': 1.4, 'high_tol': 1.55
    },
    # elevation: hard upper cutoff at 3235m
    'elevation': {
        'type': 'trapezoid',
        'low_tol': 0.1, 'low_opt': 0.1, 'high_opt': 3235.0, 'high_tol': 3235.0
    },
    # growing season precip: ramp up from 279, plateau 381-1524, hard upper cutoff
    'gs_precip': {
        'type': 'trapezoid',
        'low_tol': 279.0, 'low_opt': 381.0, 'high_opt': 1524.0, 'high_tol': 1524.0
    },
    # winter min temp: linear ramp from -41 (unsuitable) upward
    # high_val=None uses the raster max as the top of the ramp
    'winter_tasmin': {
        'type': 'linear_ramp',
        'low_tol': -41.0,
        'high_val': 0   # or set e.g. -10.0 if you want full suitability at -10°C
    },
}


# Suitability calculation
def compute_suitability(harmonized_das, fuzzy_params, site_dict, output_dir):
    """
    Compute fuzzy logic habitat suitability for each site x period x model.

    Variables used (from fuzzy_params keys):
        soil:    ph, bd
        topo:    elevation  (slope excluded)
        climate: gs_precip, winter_tasmin

    Args:
        harmonized_das (dict): output of harmonize_site_das()
        fuzzy_params (dict): {var_name: membership params dict}
        site_dict (dict): {site_name: GeoDataFrame}
        output_dir (str): root directory to save suitability rasters

    Returns:
        dict: {site: {period: {model: suitability_DataArray}}}
    """
    suitability_results = {}

    for site, site_data in harmonized_das.items():
        print(f"\nComputing suitability for site: {site}")
        suitability_results[site] = {}

        # --- static layers: same for all periods/models ---
        static_layers = {}

        # soil
        for prop in ['ph', 'bd']:
            if prop in site_data['soil'] and prop in fuzzy_params:
                static_layers[prop] = apply_membership(site_data['soil'][prop], fuzzy_params[prop])
                print(f"  static layer added: soil/{prop}")

        # topo — slope excluded
        for var in ['elevation']:
            if var in site_data['topo'] and var in fuzzy_params:
                static_layers[var] = apply_membership(site_data['topo'][var], fuzzy_params[var])
                print(f"  static layer added: topo/{var}")

        # --- dynamic layers: per period x model ---
        for period, period_data in site_data['climate'].items():
            suitability_results[site][period] = {}

            for model, model_data in period_data.items():
                print(f"  {site} | {period} | {model}")

                clim_layers = {}
                for var in ['gs_precip', 'winter_tasmin']:
                    if var in model_data and var in fuzzy_params:
                        clim_layers[var] = apply_membership(model_data[var], fuzzy_params[var])

                # combine all layers by multiplying
                all_layers = list(static_layers.values()) + list(clim_layers.values())

                if not all_layers:
                    print(f"    No layers found - skipping")
                    continue

                combined = all_layers[0]
                for layer in all_layers[1:]:
                    combined = combined * layer

                suitability_results[site][period][model] = combined

                # save raster
                raster_path = os.path.join(
                    output_dir, site, period,
                    f"suitability_{site}_{period}_{model}.tif"
                )
                export_raster(combined, raster_path)
                print(f"    Saved: {os.path.basename(raster_path)}")

    return suitability_results

# %%
# Run suitability
suitability_dir = os.path.join(data_dir, 'suitability')
suitability_results = compute_suitability(
    harmonized_das, fuzzy_params, site_dict, suitability_dir
)

# %% [markdown]
# ## STEP 5: Present your results
# Generate some plots that show your key findings of habitat suitability in your study sites across the different time periods and climate models. Don’t forget to interpret your plots!

# %%
# Plot indiviual suitabiltiy to see what is affecting suitability at each site
def plot_suitability_layers(harmonized_das, fuzzy_params, site_dict,
                             site, period, model):
    site_data = harmonized_das[site]
    clim_data = site_data['climate'][period][model]

    all_layers = {}

    for prop in ['ph', 'bd']:
        if prop in site_data['soil'] and prop in fuzzy_params:
            all_layers[prop] = apply_membership(site_data['soil'][prop], fuzzy_params[prop])

    for var in ['elevation']:
        if var in site_data['topo'] and var in fuzzy_params:
            all_layers[var] = apply_membership(site_data['topo'][var], fuzzy_params[var])

    for var in ['gs_precip', 'winter_tasmin']:
        if var in clim_data and var in fuzzy_params:
            all_layers[var] = apply_membership(clim_data[var], fuzzy_params[var])

    combined = list(all_layers.values())[0]
    for layer in list(all_layers.values())[1:]:
        combined = combined * layer
    all_layers['COMBINED'] = combined

    n_layers = len(all_layers)
    n_cols = 3
    n_map_rows = int(np.ceil(n_layers / n_cols))

    site_gdf = site_dict[site].to_crs(list(all_layers.values())[0].rio.crs)

    # gridspec: map rows + 1 dedicated colorbar row
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_map_rows + 0.6))
    gs = gridspec.GridSpec(
        n_map_rows + 1, n_cols,
        height_ratios=[5] * n_map_rows + [0.4],
        hspace=0.4, wspace=0.3
    )

    map_axes = [fig.add_subplot(gs[r, c])
                for r in range(n_map_rows)
                for c in range(n_cols)]
    cax = fig.add_subplot(gs[n_map_rows, :])

    for idx, (layer_name, da) in enumerate(all_layers.items()):
        ax = map_axes[idx]
        is_combined = layer_name == 'COMBINED'
        da.plot(ax=ax, cmap='RdYlGn', vmin=0, vmax=1, add_colorbar=False)
        site_gdf.plot(
            ax=ax, facecolor='none',
            edgecolor='white',
            linewidth=1.5 if is_combined else 1.0
        )
        ax.set_title(layer_name, fontweight='bold' if is_combined else 'normal')
        ax.set_xlabel('')
        ax.set_ylabel('')

    for idx in range(len(all_layers), len(map_axes)):
        map_axes[idx].set_visible(False)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap='RdYlGn'),
        cax=cax,
        orientation='horizontal',
        label='Suitability Score (0 = unsuitable, 1 = optimal)'
    )

    fig.suptitle(
        f'Individual Suitability Layers — {site.upper()} | {period} | {model}',
        fontsize=13
    )
    plt.show()

# %%
# --- call it for a specific combination ---
plot_suitability_layers(
    harmonized_das, fuzzy_params, site_dict,
    site='thu',
    period='2006-2035',
    model='MRI-CGCM3'
)

# %% [markdown]
# ***Figure 7:*** Site suitabiltiy for Thunder Basin. Note the low precipitation in the SE strongly limits suitability for smooth brome.

# %%
# --- call it for a specific combination ---
plot_suitability_layers(
    harmonized_das, fuzzy_params, site_dict,
    site='paw',
    period='2006-2035',
    model='MRI-CGCM3'
)

# %% [markdown]
# ***Figure 8:*** Site suitabiltiy for Pawnee. Note the low precipitation in the SE moderately limits suitability for smooth brome, but the alkaline soils of the central region are the strongest limiting factor.

# %%
# plot all climate models and time periods overall suitability
models = list({model for site_data in suitability_results.values()
               for period_data in site_data.values()
               for model in period_data.keys()})
periods = list(list(suitability_results.values())[0].keys())
sites = list(suitability_results.keys())

for model in models:

    n_rows = len(sites)
    n_cols = len(periods)

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows + 0.6))
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[5] * n_rows + [0.4],
        hspace=0.4, wspace=0.3
    )

    map_axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
                for r in range(n_rows)]
    cax = fig.add_subplot(gs[n_rows, :])

    for row_idx, site in enumerate(sites):
        for col_idx, period in enumerate(periods):

            ax = map_axes[row_idx][col_idx]
            da = suitability_results.get(site, {}).get(period, {}).get(model)

            if da is None:
                ax.set_visible(False)
                continue

            da.plot(ax=ax, cmap='RdYlGn', vmin=0, vmax=1, add_colorbar=False)
            site_dict[site].to_crs(da.rio.crs).plot(
                ax=ax, facecolor='none', edgecolor='black', linewidth=1
            )
            ax.set_title(f'{site.upper()} | {period}')
            ax.set_xlabel('')
            ax.set_ylabel('')

    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap='RdYlGn'),
        cax=cax,
        orientation='horizontal',
        label='Habitat Suitability (0 = unsuitable, 1 = optimal)'
    )
    fig.suptitle(f'Habitat Suitability — {model}', fontsize=14)
    plt.show()

# %% [markdown]
# ***Figure 9:*** Overall suitability for both sites (Pawnee and Thunder Basin), both time periods (2006-2035; 2036-2070), and four climate models.

# %% [markdown]
# ## Results Summary
# 
# | Future Type | Climate Model |
# |---|---|
# | Hot/Dry | HadGEM2-ES365 |
# | Hot/Wet | MIROC-ESM |
# | Cold/Dry | IPSL-CM5A-MR (Pawnee) / MRI-CGCM3 (Thunder Basin) |
# | Cold/Wet | IPSL-CM5A-MR (Thunder Basin) / MRI-CGCM3 (Pawnee) |
# 
# Habitat suitability projections under RCP 8.5 (2006-2035; 2036-2070) varied across climate models and sites. MIROC-ESM, the hot/wet future scenario, projected increased suitability across both Pawnee and Thunder Basin, likely reflecting the positive relationship between smooth brome growth and increased growing season precipitation. In contrast, IPSL-CM5A-MR (cold/dry at Pawnee) projected decreased suitability across the entirety of Pawnee National Grassland, while showing modest increases in the northeastern portion of Thunder Basin. HadGEM2-ES365 (hot/dry) also decreased suitability at Pawnee and produced only marginal gains in the northeastern region of Thunder Basin. MRI-CGCM3 (cold/dry at Thunder Basin) projected a slight increase in suitability at Pawnee.
# 
# Across models, suitability patterns were driven by growing season precipitation and soil pH, with drier future scenarios consistently producing less suitable conditions. The suitability of the NE portion of Thunder Basin is a reflection of that area experiencing enough rainfall to buffer the effects of a drier climate future. The SE portion (more grassland),does not experience enough rainfall and thus is unsuitable under drier scenarios.
# 
# Temperature responses were more difficult to capture due to the limited availability of optimal winter temperature data for smooth brome. The current linear ramp membership function captures the lower threshold (-41.67°C) but does not reflect any thermal optima, which definetely exists for this species. This modelwould benefit strongly from more specific temperature threshold information.


