# Climate Change in the Northern Great Basin
## Increasing Rainfall in Elko, NV

### Site Description 
Elko is located in Northwestern Nevada, with a population of over 20,000 people. It is a hub for gold mining (Nevada is the 5th largest gold producer in the world, behind four other countries, and most of that is mined near Elko), as well ranching, tourism, and the casino industry.  It is located on the banks of the Humboldt River. Very isolated, it is the largest city in 130 miles in each direction. It also has my home for three years. 

<img
src="Elko Couty Floods.jpg"
alt="Floods in Elko County, NV" 
width="50%" />

Elko County has had one or two large floods while in the last 7 years, including a one that caused an earthen dam to fail. This caused me to become curious if average precipitation had been increasing in Elko

### Data Description
Data for precipitation was gathered from the Global Historical Climatology Network - Daily (GHCN-Daily) for the Elko Airport Station. Precipitation data was collected from February 1888 to September 2024. 

### Methods Overview 
Data was downloaded from the above source via API for the entire range of precipitation available.  Total precipitation was then summed for each year.  The data then was plotted against an OLS regression trend line.  Though not the most robust way to determine precipitation trends, it was used since total annual precipitation values were being used, and the point of the data was to see if general trends were going up or down. 
### Results 
#### Graph of Annual Precipitation in Elko, Nevada since 1888. Trend line indicates that it has risen 1.5% over the last 130 years

<img
src="ElkoPrecipitation.png"
alt="Graph showing annual precipitation in Elko, NV Plotted against a Trendline" 
width="25%" />

The plotted data against the trendline shows that annual precipitation in Elko has risen, though very slowly.  This is likely due to the effects of climate change.  It is important to note that this data does not show that flooding events are more likely, as it only shows the total amount of rain in a given year.  Another analysis of daily precipitation, using different statistical methods like the Mann-Kendall (M-K) test could be used to see if flooding events are becoming more common.  Still this shows that annual rainfall in Elko, NV is increasing 

#### Sources

> <embed type="text/html" 
src="HW-4 Climate Change.html"
width="800" 
height="200">
