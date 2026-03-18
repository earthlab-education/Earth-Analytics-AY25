# %% [markdown]
# # Habitat Suitability for the Great Basin Bristlecone Pine under Climate Change
# 
# 3/17/26
# 
# ## Introduction
# 
# <div style="display: flex; align-items: flex-start; gap: 20px;">
# 
# <div style="flex: 2;">
# 
# <p>
# The Great Basin Bristlecone Pine (<i>Pinus longaeva</i>) is known for being one of the longest living tree species on Earth. These trees live in harsh conditions at high elevation (8,000 to 12,000 ft) in Nevada, California, and Utah. They are incredibly slow growing, and some years, they do not even add rings to their trunks. One tree in Great Basin National Park has an estimated age of 4,700 to 5,200 years old (<a href="https://www.nps.gov/grba/planyourvisit/identifying-bristlecone-pines.htm">National Park Service</a>).
# </p>
# 
# <p>
# Other varieties of Bristlecone Pines include the Rocky Mountain Bristlecone Pine and Foxtail Pines, but they do not live nearly as long as Great Basin Bristlecone Pines (<a href="https://www.nps.gov/grba/planyourvisit/identifying-bristlecone-pines.htm">National Park Service</a>).
# </p>
# 
# <h3>Bristlecone Pine Habitat</h3>
# 
# <p>
# The Great Basin Bristlecone Pine occurs commonly in thin, rocky substrates, usually derived from limestone and dolomite. These soils are alkaline, high in calcium and magnesium, and low in phosphorus, meaning the pH values of these soils are generally above 7 (basic), according to the <a href="https://research.fs.usda.gov/feis/species-reviews/pinlon">United States Geological Survey (USGS)</a>. The climate in these locations is generally very cold in winter and dry during the summer. Mean precipitation in these areas is about 12 inches per year. The annual average maximum temperature is around 60°F, and average minimum temperatures are about 18°F (<a href="https://research.fs.usda.gov/feis/species-reviews/pinlon">USGS</a>).
# </p>
# 
# <p>
# The Bristlecone Pine is more prolific on south and west facing slopes but can also be found on other slope directions. It generally occurs on slopes ranging from 10 to 50 degrees (<a href="https://research.fs.usda.gov/feis/species-reviews/pinlon">USGS</a>).
# </p>
# 
# </div>
# 
# <div style="flex: 1; text-align: center;">
# 
# <img src="https://www.nps.gov/grba/planyourvisit/images/Mt-Washingotn-Bristlecone2-GRBA.jpg?maxwidth=650&autorotate=false&quality=78&format=webp" width="80%">
# 
# <div style="font-size: 12px; color: gray;">NPS</div>
# 
# </div>
# 
# </div>
# 
# 
# ### Habitat Suitability Criteria
# 
# - Soil pH: >7.0
# - Elevation: 8,000 - 12,000 ft
# - Aspect: South or west preferred (values range from 0.7 to 1.0)
# - Slope: 10 - 50 degrees
# - Precipitation: 30-70 cm per year
# - Max average annual temperature: 10.0 through 18.0 degrees C
# - Min average annual temperature -4.0 through 4.0 degrees C
# 
# **Sources for Habitat suitability:**   
# National Park Service. Bristlecone Pines. https://www.nps.gov/grba/planyourvisit/identifying-bristlecone-pines.htm
# 
# United States Geological Survey. Pinus longaeva, Great Basin bristlecone pine. https://research.fs.usda.gov/feis/species-reviews/pinlon
# 
# Whitebark Pine Ecosystem Foundation. Great Basin bristlecone pine. https://whitebarkfound.org/five-needle-pines/great-basin-bristlecone-pine/
# 
# 
# ### Bristlecone Pines and Climate Change
# Due to warming global temperatures, I'm interested in understanding if and how the range of the Great Basin Bristlecone Pine could be affected. This tree prefers high elevation climates, in part due to the lack of other nearby vegetation that is an increased wildfire risk. However, future changes in temperature and precipitation may also impact the ability of this species to survive in its current habitats. Therefore, I will create a habitat suitability map using an assortment of data to predict the Bristlecon Pines future habitat suitability range for two sites in Nevada. 
# 
# ### Study Sites
# For this study, I will select two sites where Great Basin Bristlecone Pines are common. The first site is [Great Basin National Park](https://www.nps.gov/grba/index.htm) in the Northeastern part of Nevada. The other site is a portion of [Humboldt-Toiyabe National Forest](https://www.nationalforests.org/forest/humboldt-toiyabe-national-forest/) just west of Las Vegas. This will allow for comparison of how habitat suitability is changing in the northern and southern parts of the state for this species. 
# 
# ### Data
# 
# - Species Occurrence data. [Global Biodiversity Information Facility](gbif.org).
# - Boundary data for Great Basin National Park (GBNP) and Humboldt-Toiyabe National Forest (HTNF). [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview).
# - Soil data. [USGS POLARIS Data](https://www.usgs.gov/publications/polaris-properties-30-meter-probabilistic-maps-soil-properties-over-contiguous-united).
# - Topography Data. [USGS Shuttle Radar Topography Mission (SRTM)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1).
# - Climate Model Data. [Multivariate Adaptive Constructed Analogs (MACA) Datasets](https://climate.northwestknowledge.net/MACA/index.php).
# - Choosing climate Models. [ClimateToolBox.org](https://climatetoolbox.org/)
# 
# 

# %% [markdown]
# ### Python workflow to create habitat suitability maps
# Below, I will use the habitat information and outline above to create habitat suitability maps using future climate models for the Great Basin Bristlecone Pine. 

# %%
### import libraries
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
import xarray as xr
import rioxarray as rxr
import rioxarray.merge as rxrmerge
import fiona
from math import floor, ceil
from shapely.geometry import box
from rasterio.enums import Resampling

### invalid geometries
from shapely.geometry import MultiPolygon, Polygon

### visualization
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

### for api use
import requests

### earthaccess for DEMs
import earthaccess

### progress bar
from tqdm.notebook import tqdm

# %% [markdown]
# ### Set a home directory where we save everything

# %%
### set up file paths
data_dir = os.path.join(
    pathlib.Path.home(),
    'earth-analytics',
    'data',
    'hab_suit'
)

os.makedirs(data_dir, exist_ok=True)

# %% [markdown]
# ### Prepare to download Bristlecone pine occurrence data from GBIF

# %%
### set a directory for the GBIF data
gbif_dir = os.path.join(data_dir, 'gbif_bristlecone')

# %% [markdown]
# ### GBIF Login

# %%
### reset credentials
reset_credentials = False

### make dictionary for GBIF username and pass
credentials = dict(
    GBIF_USER=(input, 'GBIF username:'),
    GBIF_PWD=(getpass, 'GBIF password'),
    GBIF_EMAIL=(input, 'GBIF email'),
)

### loop through credentials and enter them
for env_variable, (prompt_func, prompt_text) in credentials.items():

    if reset_credentials and (env_variable in os.environ):
        os.environ.pop(env_variable)

    if not env_variable in os.environ:
        os.environ[env_variable] = prompt_func(prompt_text)

# %% [markdown]
# ### Select the species using Latin name (Pinus longaeva)

# %%
### species name
species_name = 'Pinus longaeva'

### species info from GBIF
species_info = species.name_lookup(species_name,
                                   rank = 'SPECIES')

### grab the first result
first_result = species_info['results'][0]
first_result

# %%
### get the species key
species_key = first_result['nubKey']

### check that
print(first_result['species'], species_key)

### assign species code
species_key = 5285258

# %%
### make a file path for data 
gbif_pattern = os.path.join(gbif_dir, '*.csv')

### download it once
if not glob(gbif_pattern):

    ### submit query
    gbif_query = occ.download([
        f"speciesKey = {species_key}",
        "hasCoordinate = True",
    ])
   
    ### only download once
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
        download_key = os.environ['GBIF_DOWNLOAD_KEY']

        ### wait for the download to build
        wait = occ.download_meta(download_key)['status']
        while not wait == 'SUCCEEDED':
            wait = occ.download_meta(download_key)['status']
            time.sleep(5)

    ### download data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'],
        path = data_dir
    )

    ### unzip the file
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path = gbif_dir)

### find csv file path
gbif_path = glob(gbif_pattern)[0]

# %%
gbif_df = pd.read_csv(
    gbif_path,
    delimiter = '\t'
)

gbif_df.head()

# %%
### make it spatial
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
gbif_gdf

# %%
### plot it
gbif_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Great Basin Bristlecone Pine Occurences in GBIF',
    fill_color = None,
    line_color = 'purple',
    framewidth = 600
)

# %% [markdown]
# ### Get site boundaries from [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview).

# %%
### site directory
site_dir = Path(data_dir) / "site_bristlecone_nv"
site_dir.mkdir(parents=True, exist_ok=True)

### define the location of the data
ITEM_ID = "6759abcfd34edfeb8710a004"
FILENAME = "PADUS4_1_State_NV_GDB_KMZ.zip"

### make the url
url = f"https://www.sciencebase.gov/catalog/file/get/{ITEM_ID}?name={FILENAME}"

### place for the shapefile to live
output_path = site_dir / FILENAME

### API call
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# %% [markdown]
# ### Extract

# %%
### unzip

### place for unzipped file
zip_path = Path(output_path)

### folder for the data
extract_folder = zip_path.parent

### create the folder if it doesn't exist
extract_folder.mkdir(parents = True, exist_ok = True)

### unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# %%
### get the layers we want (Nevada)
pa_path = extract_folder / "PADUS4_1_StateNV.gdb"

### list layers
layers = fiona.listlayers(pa_path)
layers

# %%
print(layers)


# %%
### read in the file
pa_shp = gpd.read_file(pa_path, layer = "PADUS4_1Fee_State_NV")

### check the crs
pa_shp.crs

### assign crs to match GBIF
pa_shp = pa_shp.to_crs(epsg = 4326)

### Fix invalid geometries to prevent spatial operation errors 
pa_shp['geometry'] = pa_shp['geometry'].apply(
    lambda geom: geom.make_valid() if not isinstance(geom,
                                                     MultiPolygon) and not geom.is_valid else geom)

### drop remaining invalid geoms
pa_shp = pa_shp[pa_shp.geometry.is_valid]

### drop missing geometry
pa_shp = pa_shp.dropna(subset=['geometry'])

### subset the data
subset_pa = pa_shp

## plot the subset
pa_shp.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Nevada Protected Areas',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %% [markdown]
# ### Select polygons that intersect with GBIF data

# %%
### intersect with GBIF data
bristlecone_nv = gpd.overlay(gbif_gdf, pa_shp, how = 'intersection')

### sum the number of occurences per site
value_counts = bristlecone_nv['Loc_Nm'].value_counts()
value_counts

# %% [markdown]
# ### Select site in Humboldt-Toiyabe National Forest (HTNF)

# %%
### subset to Humboldt-Toiyabe National Forest
htnf_gdf = pa_shp[pa_shp['Loc_Nm'] == 'Humboldt-Toiyabe National Forest']
htnf_gdf

# %%
### plot humboldt shapefile
htnf_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Humboldt-Toiyabe National Forest (HTNF)',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %% [markdown]
# This national forest covers, a lot of different areas. I'm interested picking a smaller area of similar size to Great Basin National Park for comparison. Since this is a climate related analysis, I'm particularly interested in the site farthest south (near Los Vegas), and compare to the northern site (Great Basin National Park).

# %%
### right now it's a multipolygon, so we need to separate it to select individual polygons
hum_separate_polys = htnf_gdf.explode(index_parts=False)

# Bounding box around Lake Mead/Las Vegas
lake_mead_bbox = box(-115.6, 35.8, -114.7, 36.5)

htnf_south_gdf = hum_separate_polys[
    hum_separate_polys.intersects(lake_mead_bbox)
]

# %%
### plot humboldt selected shapefile
htnf_south_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Humboldt-Toiyabe National Forest west of Las Vegas',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %%
### remove small polygons inside boundary to prevent errors
htnf_south_gdf["geometry"] = htnf_south_gdf.geometry.apply(
    lambda geom: Polygon(geom.exterior) if geom.geom_type == "Polygon" else geom
)

# %%
### plot humboldt selected shapefile
htnf_south_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Humboldt-Toiyabe National Forest west of Las Vegas',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %% [markdown]
# ### Select Great Basin National Park

# %%
### subset to Great Basin National Park
gbnp_gdf = pa_shp[pa_shp['Loc_Nm'] == 'GRBA']
gbnp_gdf

# %%
### plot Great Basin shapefile
gbnp_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Great Basin National Park',
    fill_color = None,
    line_color = "white",
    frame_width = 600
)

# %%
### combine into single gdf
sites_gdf = gpd.GeoDataFrame(pd.concat([htnf_south_gdf, gbnp_gdf], ignore_index=True))
sites_gdf

# %%
### perform a spatial join to select only the occurrence values in the polygons
gbif_in_sites = gpd.sjoin(
    gbif_gdf,
    sites_gdf,
    how="inner",
    predicate="within"   # keeps points inside polygons
)

# %%
sites_plot = sites_gdf.hvplot(
    geo=True,
    tiles="EsriImagery",
    fill_color=None,
    line_color="white",
    frame_width=600
)

gbif_plot = gbif_in_sites.hvplot(
    geo=True,
    color="purple",
    size=6
)

(sites_plot * gbif_plot).opts(title="GBIF Observations Within GBNP and HTNF")

# %%
### get a count of the number of occurrences in each site
gbif_in_sites.groupby("Unit_Nm").size()

# %% [markdown]
# # Download Soil Data. [USGS POLARIS Data](https://www.usgs.gov/publications/polaris-properties-30-meter-probabilistic-maps-soil-properties-over-contiguous-united).

# %%
### create file structure for GBNP and HTNF soil data

### GBNP raster folder
ph_raster_dir_gbnp = os.path.join(data_dir, "soil", "gbnp", "rasters")
os.makedirs(ph_raster_dir_gbnp, exist_ok=True)

### GBNP plot folder
ph_plots_dir_gbnp = os.path.join(data_dir, "soil", "gbnp", "plots")
os.makedirs(ph_plots_dir_gbnp, exist_ok = True)

### HTNF soil folder
htnf_soil_dir = Path(data_dir) / "soil"

### HTNF raster folder
ph_raster_dir_htnf = os.path.join(htnf_soil_dir, "htnf", "rasters")
os.makedirs(ph_raster_dir_htnf, exist_ok=True)

### HTNF plot folder
htnf_soil_plot_dir = Path(htnf_soil_dir) / "htnf"
ph_plot_dir_htnf = os.path.join(htnf_soil_plot_dir, "plots")
os.makedirs(ph_plot_dir_htnf, exist_ok=True)

# %%
### write a function to get the urls for the soil data from POLARIS
def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):

    """ Function to generate dataset of POLARIS urls using site boundary
    
    Args:
    soil_prop (str): soil property that we want (pH, bulk density, etc)
    stat (str): summary statistic (mean, pH, etc)
    soil_depth (str): soil depth in cm
    gdf_bounds: array of site boundaries
    
    Return:
    list: a list of POLARIS URLs"""

    ### extract bounding box for the site
    min_lon, min_lat, max_lon, max_lat = gdf_bounds

    ### snap boundaries to whole degrees
    site_min_lon = floor(min_lon)
    site_min_lat = floor(min_lat)
    site_max_lon = ceil(max_lon)
    site_max_lat = ceil(max_lat)

    ### initialize output list
    all_soil_urls = []

    ### loop through lat and lon to get each tile in study area
    for lon in range(site_min_lon, site_max_lon):
        for lat in range(site_min_lat, site_max_lat):

            ### define the corners of the tile
            current_max_lon = lon + 1
            current_max_lat = lat + 1

            ### define the url template
            soil_template = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"

                ### placeholders for parameters we want to vary
                "{soil_prop}/"
                "{stat}/"
                "{soil_depth}/"
                "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"                
            )

            ### fill in the template with the parameters for one complete URL
            soil_url = soil_template.format(
                soil_prop = soil_prop,
                stat = stat,
                soil_depth = soil_depth,
                min_lat = lat, max_lat = current_max_lat,
                min_lon = lon, max_lon = current_max_lon
            )

            ### add the url to the list
            all_soil_urls.append(soil_url)

    return all_soil_urls       

# %%
### function to open the raster tiles, mask and scale them, clip them to the site, and merge them
def build_da(urls, bounds):
    """
    build a dataarray from a list of urls

    Args: 
    url (list): list of urls where the data live
    bounds (tuple): site boundaries

    Returns:
    xarray.DataArray: merged DataArray
    """

    ### intialize an empty list
    all_das = []

    ### add buffer
    buffer = 0.025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    ### process 1 url tile at a time
    for url in urls:

        ### open raster, mask missing data, remove any extra dimensions
        tile_da = rxr.open_rasterio(url,
                                    mask_and_scale=True).squeeze()
        
        ### unpack the bounds and crop the tile to the buffered boundaries
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)

        ### store cropped tile
        all_das.append(cropped_da)

    ### cropped into single raster    
    merged = rxrmerge.merge_arrays(all_das)

    return merged

# %%
### Create a function to export the rasters 
def export_raster(da, raster_path, data_dir):

    """
    Export raster to file
    
    Args:
    raster (xarray.DataArray): input raster layer
    raster_path (str): output raster directory
    data_dir (str): path of data directory
    
    Returns: None
    """

    output_file = raster_path

    # remove problematic attribute
    da.attrs.pop("_FillValue", None)

    da.rio.to_raster(output_file)

# %%
### function for customizable plots
def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title,
              bar_label, plot_cmap, boundary_clr, tif_file = False):
    
    """
    Create custom site plot

    Args:
    site_da (xarray.DataArray): input site raster
    site_gdf (geopandas.GeoDataFrame: site boundary gdf
    plots_dir (str): path of plots directory for saving plots
    site_fig_name (str) site figure name
    plot_title (str): plot title
    bar_label (str): plot bar variable name
    plot_cmap (str): colormap for the plot
    boundary_clr (str): color for site boundary
    tif_file (bool): indicate site file

    Returns:
    matplotlib.pyplot.plot: a plot of site values
    """

    ### set up the figure
    fig = plt.figure(figsize = (8, 6))
    ax = plt.axes()

    ### conditional 
    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked = True)

    ### plot dataarray values
    site_plot = site_da.plot(cmap = plot_cmap,
                             cbar_kwargs = {'label': bar_label})
    
    ### plot site boundary
    site_gdf.boundary.plot(ax = plt.gca(), color = boundary_clr)

    plt.title(f'{plot_title}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    fig.savefig(f"{plots_dir}/{site_fig_name}.png")

    return site_plot

# %%
### wrapper function
def download_polaris(site_name, site_gdf, soil_prop, stat, soil_depth,
                    plot_path, plot_title, data_dir, plots_dir):

    """ 
    Retrieve POLARIS data, build DataArray, plot site, and export raster
    
    Args:
    site_name (str): name of the site, used to name exported raster file
    site_gdf (geopandas.GeoDataFrame): boundary of site, used for bounding box
    soil_prop (str): soil property of interest
    stat (str): summary statistic for POLARIS data
    soil_depth (str): soil depth in cm
    plot_path (str): string used to build plot filename
    plot_title (str): text for title of plot
    data_dir (str): path of the data directory where rasters will be saved
    plots_dir (str): path for plots directory where png plot files will be saved
    
    Returns:
    xarray.DataArray: soil DataArray for given location
    """

    ### collect the soil URLs
    site_polaris_urls = create_polaris_urls(soil_prop, stat, soil_depth, site_gdf.total_bounds)

    ### download rasters, gather into single file
    site_soil_da = build_da(site_polaris_urls, tuple(site_gdf.total_bounds))

    ### export as a raster
    raster_path = os.path.join(data_dir, f"soil_{soil_prop}.tif")

    export_raster(site_soil_da, raster_path, data_dir)

    ### plot site
    plot_site(site_soil_da, site_gdf, plots_dir,
              f'{plot_path}-Soil', f'{plot_title} Soil',
              soil_prop, 'viridis', 'white')

    ### return the soil raster
    return site_soil_da

# %%
### Create a dictionary to loop through for running the wrapper function over both sites
sites = {
    "gbnp": {
        "gdf": gbnp_gdf,
        "data_dir": ph_raster_dir_gbnp,
        "plots_dir": ph_plots_dir_gbnp
    },
    "htnf": {
        "gdf": htnf_south_gdf,
        "data_dir": ph_raster_dir_htnf,
        "plots_dir": ph_plot_dir_htnf
    }
}

# %%
### Run the wrapper function over both sites
soil_results = {}

for site_name, site_info in sites.items():

    soil_results[site_name] = download_polaris(
        site_name = site_name,
        site_gdf = site_info["gdf"],
        soil_prop = "ph",
        stat = "mean",
        soil_depth = "15_30",
        plot_path = f"plot_15_30_{site_name}",
        plot_title = f"ph_15_30_cm_{site_name}",
        data_dir = site_info["data_dir"],
        plots_dir = site_info["plots_dir"]
    )

### save the results to objects
gbnp_ph_da = soil_results["gbnp"]
htnf_ph_da = soil_results["htnf"]

# %% [markdown]
# # Download Topographic Data. [USGS Shuttle Radar Topography Mission (SRTM)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1)
# 

# %%
### Create folders for topographic data

topo_dir = os.path.join(data_dir, "topography")
os.makedirs(topo_dir, exist_ok=True)

### subfolder for GBNP
topo_gbnp_dir = os.path.join(topo_dir, "gbnp")
os.makedirs(topo_gbnp_dir, exist_ok=True)

### subfolder for Humboldt-Toiyabe National Forest
topo_htnf_dir = os.path.join(topo_dir, "htnf")
os.makedirs(topo_htnf_dir, exist_ok=True)

# %% [markdown]
# ### Login to NASA EarthData

# %%
### set up earth access
earthaccess.login()

# %%
### search for SRTM data
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %% [markdown]
# We are interested in SRTMGL1 NASA Shuttle Radar Topography Mission Global 1 arc second V003.

# %%
### create a function to download SRTM data
def download_srtm(site_name, site_gdf, topo_site_dir, buffer=0.025):
    """
    Download SRTM DEM data for a study area if not already downloaded.

    Args:
        site_name (str): name of site (used for messages)
        site_gdf (GeoDataFrame): boundary of site
        topo_site_dir (str): directory where SRTM files should be stored
        buffer (float): buffer around bounding box (degrees)

    Returns:
        tuple: buffered bounding box
        str: file pattern for downloaded SRTM files
    """

    ### file pattern
    srtm_pattern = os.path.join(topo_site_dir, "*hgt.zip")

    ### study area bounds
    elev_bounds = tuple(site_gdf.total_bounds)

    ### add buffer
    xmin, ymin, xmax, ymax = elev_bounds
    elev_bounds_buffer = (
        xmin - buffer,
        ymin - buffer,
        xmax + buffer,
        ymax + buffer
    )

    ### check if files already exist
    if not glob(srtm_pattern):

        print(f"Downloading SRTM data for {site_name}...")

        ### search data
        srtm_search = earthaccess.search_data(
            short_name="SRTMGL3",
            bounding_box=elev_bounds_buffer
        )

        ### download
        earthaccess.download(
            srtm_search,
            topo_site_dir
        )

    else:
        print(f"SRTM files already downloaded for {site_name}")

    return elev_bounds_buffer, srtm_pattern

# %%
### run the SRTM download function for the two sites
gbnp_elev_bounds_buffer, gbnp_srtm_pattern = download_srtm(
    site_name="gbnp",
    site_gdf=gbnp_gdf,
    topo_site_dir=topo_gbnp_dir
)

htnf_elev_bounds_buffer, htnf_srtm_pattern = download_srtm(
    site_name="htnf",
    site_gdf=htnf_south_gdf,
    topo_site_dir=topo_htnf_dir
)

# %%
### function to prepare elevation DEM to the study areas

def prepare_plot_dem(site_name, site_gdf, srtm_pattern, elev_bounds_buffer):
    """
    Build DEM from SRTM tiles, export GeoTIFF, and plot it.

    Args:
        site_name (str): site name for title and output filename
        site_gdf (GeoDataFrame): site boundary
        srtm_pattern (str): path pattern for SRTM tiles
        elev_bounds_buffer (tuple): buffered bounding box

    Returns:
        xarray.DataArray: merged DEM
    """

    ### collect tiles
    srtm_da_list = []

    for srtm_path in glob(srtm_pattern):

        tile_da = rxr.open_rasterio(srtm_path, mask_and_scale=True).squeeze()

        srtm_cropped_da = tile_da.rio.clip_box(*elev_bounds_buffer)

        srtm_da_list.append(srtm_cropped_da)

    ### merge tiles
    srtm_da = rxrmerge.merge_arrays(srtm_da_list)

    ### save GeoTIFF
    topo_site_dir = os.path.dirname(srtm_pattern)
    raster_path = os.path.join(topo_site_dir, f"{site_name}_dem.tif")

    ### fix encoding conflict before export
    srtm_da = srtm_da.copy()
    srtm_da.encoding.clear()

    srtm_da.rio.to_raster(raster_path)

    print(f"Saved DEM raster: {raster_path}")

    ### plotting
    fig, ax = plt.subplots(figsize=(8,6))

    dem_plot = srtm_da.plot(ax=ax, cmap='terrain')

    site_gdf.boundary.plot(ax=ax, color='black')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    dem_plot.colorbar.set_label("Elevation [meters]")

    ax.set_title(f"Elevation DEM for {site_name.upper()} Study Area")

    plt.show()

    return srtm_da

# %%
### run the function for both sites
gbnp_srtm_da = prepare_plot_dem(
    site_name="gbnp",
    site_gdf=gbnp_gdf,
    srtm_pattern=gbnp_srtm_pattern,
    elev_bounds_buffer=gbnp_elev_bounds_buffer
)

htnf_srtm_da = prepare_plot_dem(
    site_name="htnf",
    site_gdf=htnf_south_gdf,
    srtm_pattern=htnf_srtm_pattern,
    elev_bounds_buffer=htnf_elev_bounds_buffer
)

# %% [markdown]
# ### Next, get aspect data

# %%
### function to get aspect layer from elevation data, and plot it
def plot_aspect(site_name, srtm_da, site_gdf, topo_site_dir):
    """
    Calculate, save, and plot terrain aspect.

    Args:
        site_name (str): name of the site
        srtm_da (xarray.DataArray): DEM raster
        site_gdf (GeoDataFrame): site boundary
        topo_site_dir (str): directory where aspect raster will be saved

    Returns:
        xarray.DataArray: aspect raster
    """

    ### compute aspect
    aspect_da = xrspatial.aspect(srtm_da)

    ### convert -180–180 → 0–360
    aspect_da = (aspect_da + 360) % 360

    ### save raster
    raster_path = os.path.join(topo_site_dir, f"{site_name}_aspect.tif")

    aspect_da = aspect_da.copy()
    aspect_da.encoding.clear()

    aspect_da.rio.to_raster(raster_path)

    print(f"Saved aspect raster: {raster_path}")

    ### create figure
    fig, ax = plt.subplots(figsize=(8,6))

    ### plot raster
    aspect_plot = aspect_da.plot(ax=ax, cmap='terrain')

    ### overlay boundary
    site_gdf.boundary.plot(ax=ax, color='black')

    ### axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ### tick spacing
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ### decimal formatting
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ### colorbar label
    aspect_plot.colorbar.set_label("Aspect (degrees)")

    ### title
    ax.set_title(f"Terrain Aspect for {site_name.upper()} Study Area")

    plt.show()

    return aspect_da

# %%
### run the function for the two sites
gbnp_aspect = plot_aspect(
    site_name="gbnp",
    srtm_da=gbnp_srtm_da,
    site_gdf=gbnp_gdf,
    topo_site_dir=topo_gbnp_dir
)

htnf_aspect = plot_aspect(
    site_name="htnf",
    srtm_da=htnf_srtm_da,
    site_gdf=htnf_south_gdf,
    topo_site_dir=topo_htnf_dir
)

# %% [markdown]
# ### Next, get slope data

# %%
### function to get the slope

def plot_slope(site_name, srtm_da, site_gdf, topo_site_dir):
    """
    Calculate, save, and plot terrain slope.

    Args:
        site_name (str): name of the site
        srtm_da (xarray.DataArray): DEM raster
        site_gdf (GeoDataFrame): site boundary
        topo_site_dir (str): directory where slope raster will be saved

    Returns:
        xarray.DataArray: slope raster
    """

    ### reproject DEM to projected CRS
    rpj = srtm_da.rio.reproject("EPSG:5070")

    ### calculate slope
    slope = xrspatial.slope(rpj)

    ### convert back to geographic CRS for plotting
    slope_4326 = slope.rio.reproject("EPSG:4326")

    ### save raster
    raster_path = os.path.join(topo_site_dir, f"{site_name}_slope.tif")

    slope_4326 = slope_4326.copy()
    slope_4326.encoding.clear()

    slope_4326.rio.to_raster(raster_path)

    print(f"Saved slope raster: {raster_path}")

    ### create plot
    fig, ax = plt.subplots(figsize=(8,6))

    slope_plot = slope_4326.plot(ax=ax, cmap="terrain")

    site_gdf.boundary.plot(ax=ax, color="black")

    ### axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ### tick spacing
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ### reduce decimals
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ### colorbar label
    slope_plot.colorbar.set_label("Slope (degrees)")

    ### title
    ax.set_title(f"Terrain Slope for {site_name.upper()} Study Area")

    plt.show()

    return slope_4326

# %%
### run the function for both sites
gbnp_slope = plot_slope(
    site_name="gbnp",
    srtm_da=gbnp_srtm_da,
    site_gdf=gbnp_gdf,
    topo_site_dir=topo_gbnp_dir
)

htnf_slope = plot_slope(
    site_name="htnf",
    srtm_da=htnf_srtm_da,
    site_gdf=htnf_south_gdf,
    topo_site_dir=topo_htnf_dir
)

# %% [markdown]
# # Download Climate Model Data. [Multivariate Adaptive Constructed Analogs (MACA) Datasets](https://climate.northwestknowledge.net/MACA/index.php).

# %% [markdown]
# ### Choosing Climate Models
# 
# I will be running the habitat suitability model across four different climate models for these categories:
# 
# - Warm and Dry
# - Warm and Wet
# - Cold and Dry
# - Cold and Wet
# 
# For each site location, I used the scattor plot feature on [ClimateToolBox.org](https://climatetoolbox.org/) to identify an appropriate model for each of the above categories. Below are the models I chose, and will be using. 
# 
# ### Climate Models for each site (GBNP and HTNF)
# Criteria:   
# Lower Emission (RCP 4.5)
# 
# #### GBNP:
# - Warm and Dry: MIROC-ESM
# - Warm and Wet: HadGEM2-ES365
# - Cold and Dry: inmcm4
# - Cold and Wet: CNRM-CM5
# 
# #### HTNF
# - Warm and Dry: MIROC-ESM
# - Warm and Wet: CanESM2
# - Cold and Dry: GFDL-ESM2M
# - Cold and Wet: CNRM-CM5
# 

# %%
### make a directory for climate data
maca_dir = os.path.join(data_dir, 'maca_dir')
os.makedirs(maca_dir, exist_ok=True)

maca_pattern = os.path.join(maca_dir, '*nc')
maca_pattern

# %%
### Function to convert Kelvin to C (climate model data comes in K)

def convert_temperature(temp):
    """Convert Kelvin to C (climate model data comes in K)"""

    return temp - 273.15

# %%
### Function to convert longitude vals depending on the data

def convert_longitude(lon):
    """
    Convert longitude values between -180-180 and 0-360 conventions.

    Longitude converted to the opposite convention:
        - values < 0 are shifted to 0-360
        - values > 180 are shifted to -180-180
    """
    # convert -180 - 180 to 0 - 360
    if lon < 0:
        return lon + 360

    # convert 0 - 360 to -180 - 180
    if lon > 180:
        return lon - 360

    return lon

# %%
### define 30 year date ranges based on the available MACA data
def make_30yr_ranges(start, end, step=5):
    """
    Generate year ranges for MACA climate data (e.g., "2006_2010").

    Args:
        start (int): starting year of the period
        end (int): ending year of the period
        step (int): number of years per range (default is 5)

    Returns:
        list of str: list of year range strings in "startYear_endYear" format
    """
    
    ranges = []
    year = start

    while year <= end:
        end_year = min(year + step - 1, end)
        ranges.append(f"{year}_{end_year}")
        year += step

    return ranges

# %%
### run the function to make the 30 year ranges

early_century = make_30yr_ranges(2006, 2035)
late_century = make_30yr_ranges(2071, 2099)

periods = {
    "early": early_century,
    "late": late_century
}

# %%
### define the climate models for GBNP and HTNF

sites = {
    "GBNP": {
        "gdf": gbnp_gdf,
        "models": [
            "MIROC-ESM",     # warm dry
            "HadGEM2-ES365", # warm wet
            "inmcm4",        # cold dry
            "CNRM-CM5"       # cold wet
        ]
    },

    "HTNF": {
        "gdf": htnf_south_gdf,
        "models": [
            "MIROC-ESM",    # warm dry
            "CanESM2",      # warm wet
            "GFDL-ESM2M",   # cold dry
            "CNRM-CM5"      # cold wet
        ]
    }
}

# %% [markdown]
# ### Download location specific climate data
# 
# I want to download climate model data only for the specific sites (GBNP and HTNF). Below, I will extract the full MACA latitude/longitude grid so that I can subset data for each site.

# %%
### create a reference dataset
reference_model = "MIROC-ESM"
reference_range = "2006_2010"

### create a reference url for downloading
reference_url = (
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
    f"/MACAV2/{reference_model}/"
    f"macav2metdata_pr_{reference_model}_r1i1p1_rcp45_{reference_range}_CONUS_monthly.nc"
)

### open the data
ref_ds = xr.open_dataset(reference_url)

### get the full CONUS coordinates
lat = ref_ds["lat"].values
lon = ref_ds["lon"].values

# %%
### define climate model variables of interest
variables = {
    "pr": "precipitation",
    "tasmax": "air_temperature",
    "tasmin": "air_temperature"
}

# %%
def download_maca_climate_data(
    data_dir,
    sites,
    periods,
    reference_model="MIROC-ESM",
    reference_range="2006_2010",
    rcp="rcp45"
):
    """
    Download and locally save subsetted MACA climate data for multiple sites,
    models, variables, and time periods.

    Args:
        data_dir (str): base directory where climate data will be stored
        sites (dict): dictionary of site info including GeoDataFrames and model lists
        periods (dict): dictionary of period names mapped to year-range lists
        reference_model (str): MACA model used to access the reference grid
        reference_range (str): MACA time chunk used to access the reference grid
        rcp (str): emissions scenario (default is "rcp45")

    Returns:
        tuple:
            - maca_dir (str): directory where downloaded files are stored
            - var_dirs (dict): variable-specific subdirectories
            - site_indices (dict): precomputed index slices for each site
    """



    ### make a directory for climate data
    maca_dir = os.path.join(data_dir, "maca_dir")
    os.makedirs(maca_dir, exist_ok=True)

    ### define MACA variables of interest
    variables = {
        "pr": "precipitation",
        "tasmax": "air_temperature",
        "tasmin": "air_temperature"
    }

    ### make subdirectories for each variable
    var_dirs = {}
    for var in variables:
        var_dir = os.path.join(maca_dir, var)
        os.makedirs(var_dir, exist_ok=True)
        var_dirs[var] = var_dir

    ### create a reference dataset url
    reference_url = (
        "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
        f"/MACAV2/{reference_model}/"
        f"macav2metdata_pr_{reference_model}_r1i1p1_{rcp}_{reference_range}_CONUS_monthly.nc"
    )

    ### open reference dataset
    ref_ds = xr.open_dataset(reference_url)

    ### get full MACA coordinate grid
    lat = ref_ds["lat"].values
    lon = ref_ds["lon"].values

    ### precompute lat/lon index slices for each site
    site_indices = {}

    for site_name, site_info in sites.items():

        gdf = site_info["gdf"]

        ### extract bounding box
        minx, miny, maxx, maxy = gdf.total_bounds

        ### convert longitude to MACA 0–360 format
        minx = convert_longitude(minx)
        maxx = convert_longitude(maxx)

        ### add a small buffer to avoid edge clipping
        buffer = 0.25
        minx -= buffer
        maxx += buffer
        miny -= buffer
        maxy += buffer

        ### find matching MACA grid indices
        lat_idx = np.where((lat >= miny) & (lat <= maxy))[0]
        lon_idx = np.where((lon >= minx) & (lon <= maxx))[0]

        ### store slices for later subsetting
        site_indices[site_name] = {
            "lat_slice": slice(lat_idx.min(), lat_idx.max()),
            "lon_slice": slice(lon_idx.min(), lon_idx.max())
        }

    ### loop through all sites, models, variables, and time chunks
    for site_name, site_info in sites.items():

        for model in site_info["models"]:

            for var in variables:

                for period_name, date_ranges in periods.items():

                    for date_range in date_ranges:

                        print(f"\nProcessing {site_name} | {model} | {date_range} | {var}")

                        maca_url = (
                            "http://thredds.northwestknowledge.net:8080/thredds/dodsC"
                            f"/MACAV2/{model}/"
                            f"macav2metdata_{var}_{model}_r1i1p1_{rcp}_{date_range}_CONUS_monthly.nc"
                        )

                        save_path = os.path.join(
                            var_dirs[var],
                            f"{site_name}_{model}_{period_name}_{date_range}_{var}.nc"
                        )

                        if os.path.exists(save_path):
                            print("Already downloaded")
                            continue

                        ### open remote dataset
                        ds = xr.open_dataset(maca_url, decode_times=False)

                        ### subset using precomputed site slices
                        lat_slice = site_indices[site_name]["lat_slice"]
                        lon_slice = site_indices[site_name]["lon_slice"]

                        subset = ds.isel(
                            lat=lat_slice,
                            lon=lon_slice
                        )

                        ### select climate variable
                        da = subset[variables[var]]

                        ### convert temperature from Kelvin to Celsius
                        if var in ["tasmin", "tasmax"]:
                            da = convert_temperature(da)
                            da.attrs["units"] = "degC"

                        ### save clipped dataset
                        da.to_netcdf(save_path)

                        print("Saved:", save_path)

    return maca_dir, var_dirs, site_indices

# %%
maca_dir, var_dirs, site_indices = download_maca_climate_data(
    data_dir=data_dir,
    sites=sites,
    periods=periods,
    reference_model="MIROC-ESM",
    reference_range="2006_2010",
    rcp="rcp45"
)

# %% [markdown]
# ### Combine climate data into 30 year chunks
# Right now, the climate data has been downloaded in 5 year increments because this is how it is available through the API. However, this needs to be combined into files containing the 30 year data in order to get 30 year averages more easily.

# %%
def combine_maca_30yr_files(maca_dir, sites, periods, variables):
    """
    Combine downloaded 5-year MACA NetCDF files into 30-year datasets.

    Args:
        maca_dir (str): directory where downloaded MACA files are stored
        sites (dict): dictionary of site info including model lists
        periods (dict): dictionary of period names mapped to year-range lists
        variables (dict): dictionary of MACA variable names and dataset variable names

    Returns:
        str: path to directory containing combined 30-year NetCDF files
    """

    ### make a directory for combined climate data
    combined_dir = os.path.join(maca_dir, "combined_30yr")
    os.makedirs(combined_dir, exist_ok=True)

    ### define subdirectories where downloaded variable files are stored
    var_dirs = {
        "pr": os.path.join(maca_dir, "pr"),
        "tasmax": os.path.join(maca_dir, "tasmax"),
        "tasmin": os.path.join(maca_dir, "tasmin")
    }

    ### loop through variables, sites, models, and time periods
    for var in variables:

        var_dir = var_dirs[var]

        for site_name, site_info in sites.items():

            for model in site_info["models"]:

                for period_name in periods:

                    ### find all 5-year files for this site/model/period/variable
                    pattern = os.path.join(
                        var_dir,
                        f"{site_name}_{model}_{period_name}_*_{var}.nc"
                    )

                    files = sorted(glob(pattern))

                    if len(files) == 0:
                        continue

                    print(f"Combining {site_name} | {model} | {period_name} | {var}")

                    datasets = []

                    ### open each file and load into memory
                    for f in files:
                        ds = xr.open_dataset(f)
                        datasets.append(ds.load())
                        ds.close()

                    ### concatenate all chunks along the time dimension
                    combined = xr.concat(datasets, dim="time")

                    ### save the combined 30-year dataset
                    out_path = os.path.join(
                        combined_dir,
                        f"{site_name}_{model}_{period_name}_30yr_{var}.nc"
                    )

                    combined.to_netcdf(out_path)

                    print("Saved:", out_path)

    return combined_dir

# %%
### run the function
combined_dir = combine_maca_30yr_files(
    maca_dir=maca_dir,
    sites=sites,
    periods=periods,
    variables=variables
)

# %%
### Make a function to plot the results

def plot_combined_climate(combined_dir, sites, model, period):
    """
    Plot mean precipitation, tasmin, and tasmax from combined 30-year MACA datasets.

    Args:
        combined_dir (str): directory containing combined 30-year NetCDF files
        sites (dict): dictionary of site boundaries
        model (str): climate model name to plot
        period (str): time period to plot ("early" or "late")

    Returns:
        None
    """

    ### define variables to plot
    variables = ["pr", "tasmin", "tasmax"]

    ### map short variable names to dataset variable names
    var_lookup = {
        "pr": "precipitation",
        "tasmin": "air_temperature",
        "tasmax": "air_temperature"
    }

    ### define colormaps
    cmaps = {
        "pr": "Blues",
        "tasmin": "coolwarm",
        "tasmax": "coolwarm"
    }

    def load_dataset(site, variable, model, period):
        """
        Load one combined climate dataset and return a 30-year mean map.
        """

        fpath = os.path.join(
            combined_dir,
            f"{site}_{model}_{period}_30yr_{variable}.nc"
        )

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"No file found: {fpath}")

        ds = xr.open_dataset(fpath)

        da = ds[var_lookup[variable]]

        ### convert temperature if still in Kelvin
        if variable in ["tasmin", "tasmax"]:
            if float(da.mean(skipna=True)) > 100:
                da = da - 273.15
                da.attrs["units"] = "degC"

        ### fix longitude convention
        da = da.assign_coords(
            lon=("lon", [convert_longitude(l) for l in da.lon.values])
        )

        ### sort longitude and define spatial metadata
        da = da.sortby("lon")
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        da = da.rio.write_crs("EPSG:4326")

        ### calculate 30-year mean
        return da.mean(dim="time")

    ### create plot grid
    fig, axes = plt.subplots(
        3, len(sites),
        figsize=(6 * len(sites), 12)
    )

    ### loop through variables and sites
    for row, var in enumerate(variables):

        for col, (site_name, site_gdf) in enumerate(sites.items()):

            ax = axes[row, col]

            mean_map = load_dataset(site_name, var, model, period)

            mean_map.plot(
                ax=ax,
                cmap=cmaps[var],
                robust=True
            )

            site_gdf.boundary.plot(
                ax=ax,
                color="red",
                linewidth=2
            )

            ax.set_title(f"{site_name} | {var} | {model} | {period}")

    plt.tight_layout()
    plt.show()

# %%
### run the plotting function
plot_sites = {
    "GBNP": gbnp_gdf,
    "HTNF": htnf_south_gdf
}

plot_combined_climate(
    combined_dir=combined_dir,
    sites=plot_sites,
    model="MIROC-ESM",
    period="early"
)

# %% [markdown]
# ### Convert climate data to DataArrays to allow for harmonization
# Right now, the data is downloaded and combined as .nc files. We need the files in DataArray format to harmonize with our other data types (elevation, soil). The function below converts the climate data into DataArrays.

# %%
def load_climate_from_nc(site, model, period, variable, combined_dir):
    """
    Load one combined 30-year climate NetCDF as a DataArray.
    Converts temperature to Celsius.
    Returns a 2D mean climate surface.
    """

    fpath = os.path.join(
        combined_dir,
        f"{site}_{model}_{period}_30yr_{variable}.nc"
    )

    ds = xr.open_dataset(fpath)

    if variable == "pr":
        da = ds["precipitation"]
    else:
        da = ds["air_temperature"]

    # fix longitude if needed
    da = da.assign_coords(
        lon=("lon", [convert_longitude(l) for l in da.lon.values])
    )

    # make sure longitude is increasing
    da = da.sortby("lon")

    # tell rioxarray which dims are spatial
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")

    # collapse 30-year monthly stack to one 2D surface
    da = da.mean(dim="time")

    return da

# %% [markdown]
# # Data Harmonization
# Below are the data harmonization functions and steps that I used to get all the rasters to match each other. I need all the rasters/DataArrays to be perfectly aligned, and identical in terms of cell size, dimensions, and coordinate reference system. To do this, I'll choose a reference raster (the elevation DEM), and match all the other rasters to it. I then need to loop all this over the two sites and through all the data.

# %%
def make_reference_elevation(elev_da, site_gdf, site_name=None):
    """
    Clip an elevation raster to a site boundary and use it as the reference grid.

    Args:
        elev_da (xarray.DataArray): elevation raster
        site_gdf (GeoDataFrame): site boundary
        site_name (str, optional): name of the site

    Returns:
        xarray.DataArray: clipped elevation raster to use as the reference grid
    """
    ref = elev_da.rio.clip(
        site_gdf.geometry,
        site_gdf.crs,
        drop=True
    )
    if site_name is not None:
        ref.name = f"{site_name} Elevation"
    return ref


def harmonize_raster(da, reference_da, site_gdf=None, name=None):
    """
    Reproject, align, and optionally clip a raster to match a reference grid.

    Args:
        da (xarray.DataArray): raster to harmonize
        reference_da (xarray.DataArray): reference raster defining the target grid
        site_gdf (GeoDataFrame, optional): site boundary used for clipping
        name (str, optional): name to assign to the output raster

    Returns:
        xarray.DataArray: harmonized raster aligned to the reference grid
    """
    # if climate came from NetCDF, lon/lat may be spatial dims
    if "lon" in da.dims and "lat" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # if already x/y, leave it alone
    if da.rio.crs is None:
        da = da.rio.write_crs(reference_da.rio.crs)

    # match reference grid
    out = da.rio.reproject_match(reference_da)

    # optional clip
    if site_gdf is not None:
        out = out.rio.clip(
            site_gdf.geometry,
            site_gdf.crs,
            drop=True
        )

    # one more match after clip to guarantee exact alignment
    out = out.rio.reproject_match(reference_da)

    if name is not None:
        out.name = name

    return out


def print_info(da):
    """
    Print basic spatial information for a raster.

    Args:
        da (xarray.DataArray): raster to summarize

    Returns:
        None
    """

    res_x, res_y = da.rio.resolution()
    print(
        f"{da.name} "
        f"shape: {da.shape} "
        f"res: {(res_x, res_y)} "
        f"bounds: {da.rio.bounds()}"
    )


def check_alignment(rasters):
    """
    Print spatial information for a list of rasters to check alignment.

    Args:
        rasters (list of xarray.DataArray): rasters to compare

    Returns:
        None
    """
    
    for da in rasters:
        print_info(da)

# %%
def load_and_harmonize_climate_nc(
    site,
    model,
    period,
    variable,
    combined_dir,
    reference_da,
    site_gdf=None
):
    """
    Load, process, and harmonize a combined climate NetCDF dataset.

    Args:
        site (str): site name (e.g., "GBNP" or "HTNF")
        model (str): climate model name
        period (str): time period ("early" or "late")
        variable (str): climate variable ("pr", "tasmin", or "tasmax")
        combined_dir (str): directory containing combined 30-year NetCDF files
        reference_da (xarray.DataArray): reference raster defining target grid
        site_gdf (GeoDataFrame, optional): site boundary for clipping

    Returns:
        xarray.DataArray: 30-year mean climate raster aligned to the reference grid

    Notes:
        - Temperature data is converted from Kelvin to Celsius if needed.
        - Longitude values are adjusted to match -180–180 convention.
        - Bilinear interpolation is used for resampling climate variables.
    """

    fpath = os.path.join(
        combined_dir,
        f"{site}_{model}_{period}_30yr_{variable}.nc"
    )

    ds = xr.open_dataset(fpath)

    if variable == "pr":
        da = ds["precipitation"]
    else:
        da = ds["air_temperature"]

        # convert Kelvin to Celsius only if needed
        if float(da.mean(skipna=True)) > 100:
            da = da - 273.15
            da.attrs["units"] = "degC"

    # fix longitude
    da = da.assign_coords(
        lon=("lon", [convert_longitude(l) for l in da.lon.values])
    )
    da = da.sortby("lon")

    # tell rioxarray what the spatial dims are
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")

    # reduce to one 2D mean climate surface
    da = da.mean(dim="time")

    # IMPORTANT: restore spatial metadata after mean
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")

    # bilinear interpolation ONLY for climate layers
    da_h = da.rio.reproject_match(
        reference_da,
        resampling=Resampling.bilinear
    )

    if site_gdf is not None:
        da_h = da_h.rio.clip(
            site_gdf.geometry,
            site_gdf.crs,
            drop=True
        )

        # snap back to reference after clip
        da_h = da_h.rio.reproject_match(
            reference_da,
            resampling=Resampling.bilinear
        )

    da_h.name = f"{site} {variable}"

    return da_h

# %%
def harmonize_all_site_layers(
    maca_dir,
    sites,
    site_rasters,
    model_period="early",
    climate_variables=("pr", "tasmin", "tasmax"),
    check=True
):
    """
    Create harmonized static and climate rasters for all sites and models.

    Args:
        maca_dir (str): base MACA directory
        sites (dict): site config dictionary with gdf and models
        site_rasters (dict): per-site raster inputs
        model_period (str): climate period to load (e.g., "early" or "late")
        climate_variables (tuple): climate vars to process
        check (bool): whether to print alignment summaries

    Returns:
        dict: nested dictionary of harmonized rasters
    """

    combined_dir = os.path.join(maca_dir, "combined_30yr")
    results = {}

    for site_name, site_info in sites.items():
        print(f"\nProcessing {site_name}")

        site_gdf = site_info["gdf"]
        models = site_info["models"]

        elev_da = site_rasters[site_name]["elevation"]
        slope_da = site_rasters[site_name]["slope"]
        aspect_da = site_rasters[site_name]["aspect"]
        ph_da = site_rasters[site_name]["ph"]

        # reference grid
        reference_da = make_reference_elevation(elev_da, site_gdf, site_name)

        # static rasters
        slope_h = harmonize_raster(
            da=slope_da,
            reference_da=reference_da,
            site_gdf=site_gdf,
            name=f"{site_name} Slope"
        )

        aspect_h = harmonize_raster(
            da=aspect_da,
            reference_da=reference_da,
            site_gdf=site_gdf,
            name=f"{site_name} Aspect"
        )

        ph_h = harmonize_raster(
            da=ph_da,
            reference_da=reference_da,
            site_gdf=site_gdf,
            name=f"{site_name} pH"
        )

        results[site_name] = {
            "reference": reference_da,
            "static": {
                "slope": slope_h,
                "aspect": aspect_h,
                "ph": ph_h
            },
            "climate": {}
        }

        # climate rasters for each model
        for model in models:
            print(f"  Model: {model}")

            results[site_name]["climate"][model] = {}

            for variable in climate_variables:
                print(f"    Variable: {variable}")

                da_h = load_and_harmonize_climate_nc(
                    site=site_name,
                    model=model,
                    period=model_period,
                    variable=variable,
                    combined_dir=combined_dir,
                    reference_da=reference_da,
                    site_gdf=site_gdf
                )

                results[site_name]["climate"][model][variable] = da_h

        if check:
            print(f"\nAlignment check for {site_name}")
            first_model = models[0]

            check_alignment([
                results[site_name]["reference"],
                results[site_name]["static"]["slope"],
                results[site_name]["static"]["aspect"],
                results[site_name]["static"]["ph"],
                results[site_name]["climate"][first_model]["pr"],
                results[site_name]["climate"][first_model]["tasmin"],
                results[site_name]["climate"][first_model]["tasmax"]
            ])

    return results

# %%
### input rasters for each site used in harmonization 
site_rasters = {
    "GBNP": {
        "elevation": gbnp_srtm_da,
        "slope": gbnp_slope,
        "aspect": gbnp_aspect,
        "ph": gbnp_ph_da
    },
    "HTNF": {
        "elevation": htnf_srtm_da,
        "slope": htnf_slope,
        "aspect": htnf_aspect,
        "ph": htnf_ph_da
    }
}

# %%
### run the function to harmonize everything
harmonized = harmonize_all_site_layers(
    maca_dir=maca_dir,
    sites=sites,
    site_rasters=site_rasters,
    model_period="early",
    climate_variables=("pr", "tasmin", "tasmax"),
    check=True
)

# %% [markdown]
# Next, I want to visualize the rasters to do a final check to make sure everything lines up. 

# %%
def validate_site_layers(
    reference,
    layers,
    boundary_gdf=None,
    plot_examples=True,
    ncols=3,
    cmap="viridis"
):
    """
    Run validation checks on harmonized raster layers for a site.

    Args:
        reference (xarray.DataArray): reference raster defining the target grid
        layers (list of xarray.DataArray): raster layers to validate against the reference
        boundary_gdf (GeoDataFrame, optional): site boundary used for plotting overlay
        plot_examples (bool, optional): whether to plot the reference and layers
        ncols (int, optional): number of columns in the plot grid
        cmap (str, optional): colormap used for plotting raster layers

    Returns:
        None

    Notes:
        - Prints summary information for the reference raster.
        - Checks alignment of each layer against the reference raster, including
          shape, resolution, bounds, and CRS.
        - Prints value ranges and missing-data summaries for all rasters.
        - Optionally plots the reference raster and all layers, with site boundary
          overlay if provided.
    """

    def _safe_name(da):
        return da.name if da.name is not None else "Unnamed raster"

    def _cell_size(da):
        res_x, res_y = da.rio.resolution()
        return abs(res_x), abs(res_y)

    print("\n" + "=" * 70)
    print(f"VALIDATION FOR SITE: {_safe_name(reference)}")
    print("=" * 70)

    # Reference summary
    print("\nREFERENCE RASTER")
    print("-" * 70)
    print("name:", _safe_name(reference))
    print("shape:", reference.shape)
    print("resolution:", reference.rio.resolution())
    print("cell size:", _cell_size(reference))
    print("bounds:", reference.rio.bounds())
    print("crs:", reference.rio.crs)

    ref_shape = reference.shape
    ref_bounds = reference.rio.bounds()
    ref_res = reference.rio.resolution()
    ref_crs = reference.rio.crs


    # Alignment / metadata checks
    print("\nALIGNMENT CHECKS")
    print("-" * 70)

    for da in layers:
        print(f"\n{_safe_name(da)}")
        print("shape:", da.shape, "| match:", da.shape == ref_shape)
        print("resolution:", da.rio.resolution(), "| match:", da.rio.resolution() == ref_res)
        print("cell size:", _cell_size(da))
        print("bounds:", da.rio.bounds(), "| match:", da.rio.bounds() == ref_bounds)
        print("crs:", da.rio.crs, "| match:", da.rio.crs == ref_crs)

    # Value range checks
    print("\nVALUE RANGES")
    print("-" * 70)

    for da in [reference] + layers:
        name = _safe_name(da)
        arr = da.values
        valid = arr[~np.isnan(arr)]

        if valid.size == 0:
            print(f"{name}: all values are NaN")
            continue

        print(
            f"{name}: "
            f"min={float(np.nanmin(arr)):.3f}, "
            f"max={float(np.nanmax(arr)):.3f}, "
            f"mean={float(np.nanmean(arr)):.3f}"
        )

    # Missing data checks
    print("\nNODATA SUMMARY")
    print("-" * 70)

    for da in [reference] + layers:
        name = _safe_name(da)
        arr = da.values
        total = arr.size
        missing = int(np.isnan(arr).sum())
        pct = (missing / total) * 100 if total > 0 else np.nan

        print(f"{name}: missing={missing} / {total} ({pct:.2f}%)")

    # Plot checks
    if plot_examples:
        all_layers = [reference] + layers
        n = len(all_layers)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, da in zip(axes, all_layers):
            da.plot(ax=ax, cmap=cmap, add_colorbar=True)

            if boundary_gdf is not None:
                boundary_gdf.boundary.plot(
                    ax=ax,
                    color="white",
                    linewidth=1.5
                )

            ax.set_title(_safe_name(da))
            ax.set_axis_off()

        for ax in axes[len(all_layers):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    print("\nValidation complete.\n")

# %%
### run the function to check the results of the harmonization for GBNP
validate_site_layers(
    reference=harmonized["GBNP"]["reference"],
    layers=[
        harmonized["GBNP"]["static"]["slope"],
        harmonized["GBNP"]["static"]["aspect"],
        harmonized["GBNP"]["static"]["ph"],
        harmonized["GBNP"]["climate"]["MIROC-ESM"]["pr"],
        harmonized["GBNP"]["climate"]["MIROC-ESM"]["tasmin"],
        harmonized["GBNP"]["climate"]["MIROC-ESM"]["tasmax"]
    ],
    boundary_gdf=sites["GBNP"]["gdf"],
    plot_examples=True
)

# %%
validate_site_layers(
    reference=harmonized["HTNF"]["reference"],
    layers=[
        harmonized["HTNF"]["static"]["slope"],
        harmonized["HTNF"]["static"]["aspect"],
        harmonized["HTNF"]["static"]["ph"],
        harmonized["HTNF"]["climate"]["MIROC-ESM"]["pr"],
        harmonized["HTNF"]["climate"]["MIROC-ESM"]["tasmin"],
        harmonized["HTNF"]["climate"]["MIROC-ESM"]["tasmax"]
    ],
    boundary_gdf=sites["HTNF"]["gdf"],
    plot_examples=True
)

# %% [markdown]
# Results of harmonization show that everything is lining up.

# %% [markdown]
# # Fuzzy Logic Model
# 
# ### Habitat Suitability Criteria (inputs for the model)
# 
# - Soil pH: >7.0
# - Elevation: 8,000 - 12,000 ft
# - Aspect: South or west preferred (values range from 0.7 to 1.0)
# - Slope: 10 - 50 degrees
# - Precipitation: 30-70 cm per year
# - Max average annual temperature: 10.0 through 18.0 degrees C
# - Min average annual temperature -4.0 through 4.0 degrees C

# %%
def fuzzy_between(da, low_opt, high_opt, low_tol, high_tol):
    """
    Create a fuzzy membership surface for values within an optimal range.

    Args:
        da (xarray.DataArray): input raster values to evaluate
        low_opt (float): lower bound of the optimal range
        high_opt (float): upper bound of the optimal range
        low_tol (float): tolerance below the lower optimal bound
        high_tol (float): tolerance above the upper optimal bound

    Returns:
        xarray.DataArray: fuzzy membership raster scaled from 0 to 1

    Notes:
        - Membership is 1 inside the optimal range.
        - Membership decreases linearly to 0 outside the optimal range
          across the specified tolerance limits.
        - Values below the lower tolerance limit or above the upper
          tolerance limit are assigned 0.
    """
    low_min = low_opt - low_tol
    high_max = high_opt + high_tol

    out = xr.zeros_like(da, dtype=float)

    # assign rising membership below the optimal range
    rising = (da > low_min) & (da < low_opt)
    out = xr.where(rising, (da - low_min) / (low_opt - low_min), out)

    # assign full membership within the optimal range
    plateau = (da >= low_opt) & (da <= high_opt)
    out = xr.where(plateau, 1.0, out)

    # assign falling membership above the optimal range
    falling = (da > high_opt) & (da < high_max)
    out = xr.where(falling, (high_max - da) / (high_max - high_opt), out)

    return out.clip(0, 1)


def fuzzy_greater(da, threshold, tol):
    """
    Create a fuzzy membership surface for values greater than a threshold.

    Args:
        da (xarray.DataArray): input raster values to evaluate
        threshold (float): value at which membership reaches 1
        tol (float): tolerance below the threshold over which membership
            increases linearly from 0 to 1

    Returns:
        xarray.DataArray: fuzzy membership raster scaled from 0 to 1

    Notes:
        - Membership is 0 below threshold - tol.
        - Membership increases linearly from 0 to 1 between
          threshold - tol and threshold.
        - Membership remains 1 at and above the threshold.
    """
    low_min = threshold - tol
    out = xr.zeros_like(da, dtype=float)

    # assign ramped membership below the threshold
    ramp = (da > low_min) & (da < threshold)
    out = xr.where(ramp, (da - low_min) / (threshold - low_min), out)

    # assign full membership at and above the threshold
    out = xr.where(da >= threshold, 1.0, out)

    return out.clip(0, 1)


def fuzzy_aspect_soft(da, preferred=(180, 270), sigma=45, min_weight=0.7):
    """
    Create a soft fuzzy membership surface for aspect preference.

    Args:
        da (xarray.DataArray): aspect raster in degrees
        preferred (tuple, optional): preferred aspect center values in degrees
        sigma (float, optional): spread of the aspect preference curve
        min_weight (float, optional): minimum membership assigned to
            non-preferred aspects

    Returns:
        xarray.DataArray: fuzzy membership raster scaled from min_weight to 1

    Notes:
        - Preferred aspects receive the highest suitability.
        - Suitability decreases smoothly with circular distance from the
          preferred aspect centers.
        - All aspects retain at least the minimum membership value.
    """
    # define helper function for circular distance on a 0–360 degree scale
    def circ_dist(a, center):
        d = abs(a - center)
        return xr.where(d > 180, 360 - d, d)

    memberships = []
    for center in preferred:
        d = circ_dist(da, center)
        memberships.append(np.exp(-(d ** 2) / (2 * sigma ** 2)))

    best = xr.concat(memberships, dim="pref").max("pref")

    # rescale so all aspects retain some suitability
    return min_weight + (1 - min_weight) * best


def run_fuzzy_model(
    reference,
    elev,
    slope,
    aspect,
    soil,
    pr,
    tasmin,
    tasmax,
    site_name,
    model,
    period,
    output_dir
):
    """
    Build and save a fuzzy habitat suitability raster from aligned inputs.

    Args:
        reference (xarray.DataArray): reference raster defining the target grid
        elev (xarray.DataArray): harmonized elevation raster
        slope (xarray.DataArray): harmonized slope raster
        aspect (xarray.DataArray): harmonized aspect raster
        soil (xarray.DataArray): harmonized soil pH raster
        pr (xarray.DataArray): harmonized precipitation raster
        tasmin (xarray.DataArray): harmonized minimum temperature raster
        tasmax (xarray.DataArray): harmonized maximum temperature raster
        site_name (str): site name used in output naming
        model (str): climate model name used in output naming
        period (str): climate period name used in output naming
        output_dir (str): directory where the output raster will be saved

    Returns:
        dict: dictionary containing the final habitat suitability raster and
        all intermediate fuzzy membership rasters

    Notes:
        - All input rasters must already be harmonized to the same grid.
        - Fuzzy membership functions are applied separately to elevation,
          slope, soil pH, aspect, precipitation, minimum temperature,
          and maximum temperature.
        - The final suitability raster is calculated as the product of all
          fuzzy membership layers.
        - The output raster is saved as a GeoTIFF.
    """

    # Define habitat criteria for elevation
    elev_fuzzy = fuzzy_between(
        elev,
        low_opt=2440,
        high_opt=3810,
        low_tol=300,
        high_tol=300
    )

    # Define habitat criteria for slope
    slope_fuzzy = fuzzy_between(
        slope,
        low_opt=10,
        high_opt=50,
        low_tol=10,
        high_tol=10
    )

    # Define habitat criteria for soil pH
    soil_fuzzy = fuzzy_greater(
        soil,
        threshold=7.0,
        tol=3.0
    )

    # Define habitat criteria for aspect
    aspect_fuzzy = fuzzy_aspect_soft(
        aspect,
        preferred=(180, 270),
        sigma=45,
        min_weight=0.7
    )

    # Define habitat criteria for precipitation
    # climate rasters are 30-year mean monthly values
    pr_fuzzy = fuzzy_between(
        pr,
        low_opt=30,
        high_opt=70,
        low_tol=15,
        high_tol=15
    )

    # Define habitat criteria for max temperature
    # temperatures are in C
    tasmax_fuzzy = fuzzy_between(
        tasmax,
        low_opt=10.0,
        high_opt=18.0,
        low_tol=15,
        high_tol=15
    )

    # Define habitat criteria for min temperature
    tasmin_fuzzy = fuzzy_between(
        tasmin,
        low_opt=-4.0,
        high_opt=4,
        low_tol=15,
        high_tol=15
    )

    # combine all fuzzy membership layers into one suitability surface
    suitability = xr.ones_like(reference, dtype=float)

    suitability = (
        suitability
        * elev_fuzzy
        * slope_fuzzy
        * soil_fuzzy
        * aspect_fuzzy
        * pr_fuzzy
        * tasmin_fuzzy
        * tasmax_fuzzy
    )

    suitability.name = "habitat_suitability"

    # restore spatial metadata from the reference raster
    suitability = suitability.assign_coords({
        "x": reference["x"],
        "y": reference["y"]
    })
    suitability = suitability.rio.write_crs(reference.rio.crs)
    suitability = suitability.rio.write_transform(reference.rio.transform())

    out_file = os.path.join(
        output_dir,
        f"{site_name}_{model}_{period}_fuzzy_habitat.tif"
    )

    suitability.rio.to_raster(out_file)
    print("Saved:", out_file)

    return {
        "suitability": suitability,
        "elev_fuzzy": elev_fuzzy,
        "slope_fuzzy": slope_fuzzy,
        "soil_fuzzy": soil_fuzzy,
        "aspect_fuzzy": aspect_fuzzy,
        "pr_fuzzy": pr_fuzzy,
        "tasmin_fuzzy": tasmin_fuzzy,
        "tasmax_fuzzy": tasmax_fuzzy,
    }

# %% [markdown]
# ### Test out the results of the fuzzy logic model for one site 

# %%
# create output directory for habitat suitability rasters
fuzzy_output_dir = os.path.join(data_dir, "habitat_suitability")
os.makedirs(fuzzy_output_dir, exist_ok=True)

# run habitat model for one site / model / period (test run)
gbnp_test = run_fuzzy_model(
    reference=harmonized["GBNP"]["reference"],
    elev=harmonized["GBNP"]["reference"],
    slope=harmonized["GBNP"]["static"]["slope"],
    aspect=harmonized["GBNP"]["static"]["aspect"],
    soil=harmonized["GBNP"]["static"]["ph"],
    pr=harmonized["GBNP"]["climate"]["MIROC-ESM"]["pr"],
    tasmin=harmonized["GBNP"]["climate"]["MIROC-ESM"]["tasmin"],
    tasmax=harmonized["GBNP"]["climate"]["MIROC-ESM"]["tasmax"],
    site_name="GBNP",
    model="MIROC-ESM",
    period="early",
    output_dir=fuzzy_output_dir
)

# %% [markdown]
# Plot the results

# %%
### create plot of results for 1 site
fig, ax = plt.subplots(figsize=(7, 7))
gbnp_test["suitability"].plot(ax=ax, cmap="YlGn", vmin=0, vmax=1)
gbnp_gdf.boundary.plot(ax=ax, color="black", linewidth=1)
ax.set_title("GBNP | MIROC-ESM | early | habitat suitability")
ax.set_axis_off()
plt.tight_layout()
plt.show()

# %% [markdown]
# This appears to be a reasonable suitability map. Darker greens indicate a higher level of habitat suitability, meaning more of the criteria in that location meet the requirements for Great Basin Bristlecone Pines. Below, I will check all the layers that went into making this map to see if they look reasonable.

# %%
# define layers and final suitability surface for plotting
layers_to_plot = [
    gbnp_test["elev_fuzzy"],
    gbnp_test["slope_fuzzy"],
    gbnp_test["soil_fuzzy"],
    gbnp_test["aspect_fuzzy"],
    gbnp_test["pr_fuzzy"],
    gbnp_test["tasmin_fuzzy"],
    gbnp_test["tasmax_fuzzy"],
    gbnp_test["suitability"]
]

# define plot titles for each layer
titles = [
    "Elevation fuzzy",
    "Slope fuzzy",
    "Soil fuzzy",
    "Aspect fuzzy",
    "Precip fuzzy",
    "Tasmin fuzzy",
    "Tasmax fuzzy",
    "Final suitability"
]

# create subplot grid for visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()


# plot each fuzzy layer and overlay site boundary
for ax, da, title in zip(axes, layers_to_plot, titles):
    da.plot(ax=ax, cmap="YlGn", vmin=0, vmax=1, add_colorbar=False)
    gbnp_gdf.boundary.plot(ax=ax, color="white", linewidth=1)
    ax.set_title(title)
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# %% [markdown]
# This allows me to check each layer individually to make sure there isn't a problem with the criteria.

# %% [markdown]
# Next, I need to be able to display more of the suitability maps for more models, time periods, etc. Below, I will organize the data to do this.

# %%
# organize static rasters, boundaries, and model lists for each site
site_static = {}

for site_name in harmonized:
    site_static[site_name] = {
        "reference": harmonized[site_name]["reference"],
        "elev": harmonized[site_name]["reference"],
        "slope": harmonized[site_name]["static"]["slope"],
        "aspect": harmonized[site_name]["static"]["aspect"],
        "soil": harmonized[site_name]["static"]["ph"],
        "boundary": sites[site_name]["gdf"],
        "models": sites[site_name]["models"]
    }

# define climate periods to evaluate
periods = ["early", "late"]

# %% [markdown]
# I would like to save the final suitability maps as .pngs in case I'd like to use them for reports. The function below allows me to save the results as .pngs.

# %%
# create output directory for fuzzy habitat suitability PNG maps
png_output_dir = os.path.join(data_dir, "fuzzy_habitat_pngs")
os.makedirs(png_output_dir, exist_ok=True)

def save_suitability_png(suitability_da, boundary_gdf, site_name, model, period, png_dir):
    """
    Create and save a PNG map of habitat suitability for a given site, model, and period.

    Args:
        suitability_da (xarray.DataArray): habitat suitability raster (values 0–1)
        boundary_gdf (GeoDataFrame): site boundary used for map overlay
        site_name (str): site name used in plot title and output filename
        model (str): climate model name used in plot title and output filename
        period (str): climate period used in plot title and output filename
        png_dir (str): directory where the PNG file will be saved

    Returns:
        None

    Notes:
        - The suitability raster is plotted using a 0–1 color scale (YlGn colormap).
        - The site boundary is overlaid for spatial reference.
        - Output PNG filenames follow the pattern:
          {site_name}_{model}_{period}_fuzzy_habitat.png
        - Figures are saved at 300 dpi and closed after saving to free memory.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    suitability_da.plot(
        ax=ax,
        cmap="YlGn",
        vmin=0,
        vmax=1
    )

    boundary_gdf.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1
    )

    ax.set_title(f"{site_name} | {model} | {period}")
    ax.set_axis_off()
    plt.tight_layout()

    out_png = os.path.join(
        png_dir,
        f"{site_name}_{model}_{period}_fuzzy_habitat.png"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved PNG:", out_png)

# %% [markdown]
# Below, I loop through each site, climate model, and time period (early or late century) to generate the final habitat suitability maps. For each combination, it loads and harmonizes climate data, runs the suitability model, and saves the results as raster files and .png files. 

# %%
# create output directory for fuzzy habitat suitability rasters
fuzzy_output_dir = os.path.join(data_dir, "fuzzy_habitat")
os.makedirs(fuzzy_output_dir, exist_ok=True)

# list to collect results for all model runs
all_results = []


# iterate over each site and its associated inputs
for site_name, info in site_static.items():
    print(f"\n=== Running site: {site_name} ===")

    # extract harmonized static rasters and boundary
    reference = info["reference"]
    elev = info["elev"]
    slope = info["slope"]
    aspect = info["aspect"]
    soil = info["soil"]
    boundary = info["boundary"]

    # iterate over all climate models and periods
    for model in info["models"]:
        for period in periods:
            print(f"Processing {site_name} | {model} | {period}")

            # load + harmonize climate from combined .nc
            pr_h = load_and_harmonize_climate_nc(
                site=site_name,
                model=model,
                period=period,
                variable="pr",
                combined_dir=combined_dir,
                reference_da=reference,
                site_gdf=boundary
            )

            tasmin_h = load_and_harmonize_climate_nc(
                site=site_name,
                model=model,
                period=period,
                variable="tasmin",
                combined_dir=combined_dir,
                reference_da=reference,
                site_gdf=boundary
            )

            tasmax_h = load_and_harmonize_climate_nc(
                site=site_name,
                model=model,
                period=period,
                variable="tasmax",
                combined_dir=combined_dir,
                reference_da=reference,
                site_gdf=boundary
            )

            result = run_fuzzy_model(
                reference=reference,
                elev=elev,
                slope=slope,
                aspect=aspect,
                soil=soil,
                pr=pr_h,
                tasmin=tasmin_h,
                tasmax=tasmax_h,
                site_name=site_name,
                model=model,
                period=period,
                output_dir=fuzzy_output_dir
            )

            save_suitability_png(
                suitability_da=result["suitability"],
                boundary_gdf=boundary,
                site_name=site_name,
                model=model,
                period=period,
                png_dir=png_output_dir
            )

            all_results.append({
                "site": site_name,
                "model": model,
                "period": period,
                "suitability": result["suitability"]
            })

# %% [markdown]
# Below, I have a function to plot the results in a grid, organized by climate model and time period (early and late century).

# %%
def plot_site_suitability_grid(results, site_name, boundary_gdf, gbif_gdf=None):
    """
    Plot a grid of habitat suitability rasters for a given site across models and periods.

    Args:
        results (list of dict): list of model outputs containing site, model,
            period, and suitability raster
        site_name (str): site name used to filter results and label the plot
        boundary_gdf (GeoDataFrame): site boundary used for map overlay
        gbif_gdf (GeoDataFrame, optional): GBIF occurrence points to overlay
            on each subplot

    Returns:
        None

    Notes:
        - Creates a grid of plots with rows representing climate models and
          columns representing time periods (e.g., early and late).
        - Each subplot displays the habitat suitability raster (0-1 scale)
          with the site boundary overlaid.
        - GBIF points are clipped to the site boundary before plotting if
          provided.
        - Subplots are automatically hidden if a model/period combination
          is not available.
        - A shared color scale is used across all plots for consistency.
    """

    site_results = [r for r in results if r["site"] == site_name]

    models = sorted(list(set(r["model"] for r in site_results)))
    periods = ["early", "late"]

    # subset GBIF points to the site boundary if provided
    gbif_site = None
    if gbif_gdf is not None:
        gbif_site = gbif_gdf.to_crs(boundary_gdf.crs)
        gbif_site = gpd.clip(gbif_site, boundary_gdf)

    fig, axes = plt.subplots(
        len(models), len(periods),
        figsize=(5 * len(periods), 4 * len(models)),
        squeeze=False
    )

    for i, model in enumerate(models):
        for j, period in enumerate(periods):
            ax = axes[i, j]

            match = [
                r for r in site_results
                if r["model"] == model and r["period"] == period
            ]

            if len(match) == 0:
                ax.set_visible(False)
                continue

            da = match[0]["suitability"]

            da.plot(
                ax=ax,
                cmap="YlGn",
                vmin=0,
                vmax=1,
                add_colorbar=(j == len(periods) - 1)
            )

            boundary_gdf.to_crs(da.rio.crs).boundary.plot(
                ax=ax,
                color="black",
                linewidth=1
            )

            if gbif_site is not None and len(gbif_site) > 0:
                gbif_site.to_crs(da.rio.crs).plot(
                    ax=ax,
                    color="black",
                    markersize=5,
                    alpha=0.8
                )

            ax.set_title(f"{model} | {period} century")
            ax.set_axis_off()

    plt.suptitle(f"{site_name} Habitat Suitability", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Plot the final habitat suitability maps with GBIF occurrence data

# %%
plot_site_suitability_grid(
    results=all_results,
    site_name="GBNP",
    boundary_gdf=gbnp_gdf,
    gbif_gdf=gbif_gdf
)

plot_site_suitability_grid(
    results=all_results,
    site_name="HTNF",
    boundary_gdf=htnf_south_gdf,
    gbif_gdf=gbif_gdf
)

# %% [markdown]
# Early Century = 2006-2035   
# Late Century = 2071-2099
# 
# #### Model types for GBNP:
# - Warm and Dry: MIROC-ESM
# - Warm and Wet: HadGEM2-ES365
# - Cold and Dry: inmcm4
# - Cold and Wet: CNRM-CM5
# 

# %% [markdown]
# Early Century = 2006-2035   
# Late Century = 2071-2099
# 
# #### Model types for HTNF
# - Warm and Dry: MIROC-ESM
# - Warm and Wet: CanESM2
# - Cold and Dry: GFDL-ESM2M
# - Cold and Wet: CNRM-CM5

# %% [markdown]
# ### Results
# The above maps show habitat suitability for all four climate models for both early and late century time periods (2006-2035), and (2071-2099), using the green colorbar. The colorbar ranges from 0 to 1, where 0 indicates habitat areas that are completely non-compatible with the Great Basin Bristlecone Pine, and 1 represents habitat areas that are completely ideal for the species. Additionally, I have plotted the GBIF occurrence data for the bristlecone pine on top of the habitat suitability maps to get a better sense of where the species lives, and how it might be affected in the future by climate change. 

# %% [markdown]
# ### Discussion
# 
# The early century habitat suitability maps (left side), all indicate a strong level of habitat suitability (>0.5) for the areas that overlap with the majority of occurrence data points for the species. This is expected for the early century, since Great Basin bristlecone pines are known to be very tolerant of harsh climate conditions ([Whitebark Pine Ecosystem Foundation](https://whitebarkfound.org/five-needle-pines/great-basin-bristlecone-pine/
# )), and the values for early century temperature and precipitation are consistent with the species' habitat requirements. 
# 
# The late century habitat suitability maps (right side) also indicate a strong level of habitat suitability for the areas overlapping with the species occurrence data points. However, the perimeter of suitable habitats is slightly decreased on some of the climate models, in particular, the MIROC-ESM maps for both sites. The MIROC-ESM is climate model pridicting a warmer and drier climate compared to the other models, so this makes sense that this would predict a slightly reduced habitat range in the future. Overall, the fact that habitat suitability is not greatly affected for the bristlecone pine is good news for this species. This is assuming that carbon emissions follow a trajectory similar to the RCP 4.5 emissions scenario. However, an RCP 8.5 or "business as usual" scenario ([CarbonBrief](https://www.carbonbrief.org/explainer-the-high-emissions-rcp8-5-global-warming-scenario/)), would likely impact habitat suitability more drastically. This analysis could easily be adapted to include RCP 8.5 climate model data. 
# 
# Great Basin National Park is the northern site for this study. It has a larger buffer of suitable habitat surrounding occurrence data when compared to the Humboldt-Toiyabe National Forest site in southern Nevada. This appears to be largely due to the elevation requirements of the species, and shows that Great Basin National Park has a larger area above 8,000 ft that is suitable for the species. 

# %% [markdown]
# ### Conclusion
# This workflow demonstrates how to create a habitat suitability map for a species, and allows for the comparison of different sites and climate models. Creating an accurate habitat suitability map requires several different data types, including data for elevation, slope, soil pH, precipitation, and temperature. Once all this data is downloaded, its very important that all the rasters are clipped and aligned to exactly the same sizes. I was able to create data harmonization functions that do this efficiently, using a reference raster, and matching all subsequent rasters to it. Finally, I used a fuzzy logic model to perform raster algebra, where cells are added, pixel by pixel to identify which pixels meet all the habitat suitability requirements. The results show that habitat suitability was identified for locations overlapping with where the species currently exists (GBIF). This is promising evidence that the habitat suitability model ran accurately. Additionally, the late century habitat suitability maps show a slight decrease in habitat range for the species, especially for the MIROC-ESM (warm and dry) climate model. However, the overall range of habitat suitability for the species remains intact through these late century predictions, assuming an RCP 4.5 emissions scenario. 

# %% [markdown]
# ### Works Cited
# 
# Carbon Brief. The high-emissions ‘RCP8.5’ global warming scenario. https://www.carbonbrief.org/explainer-the-high-emissions-rcp8-5-global-warming-scenario/
# 
# National Aeronautics and Space Administration. Climate Model: Temperature Change (RCP 4.5), 2006-2100. https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-45-2006-2100/
# 
# National Park Service. Bristlecone Pines. https://www.nps.gov/grba/planyourvisit/identifying-bristlecone-pines.htm
# 
# National Park Service. Great Basin National Park. https://www.nps.gov/grba/index.htm 
# 
# United States Geological Survey. Pinus longaeva, Great Basin bristlecone pine. https://research.fs.usda.gov/feis/species-reviews/pinlon
# 
# US Forest Service. Humboldt-Toiyabe National Forest. https://www.fs.usda.gov/r04/humboldt-toiyabe
# 
# Whitebark Pine Ecosystem Foundation. Great Basin bristlecone pine. https://whitebarkfound.org/five-needle-pines/great-basin-bristlecone-pine/


