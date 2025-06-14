# Landscape Clustering by an FFT footprint
**TL;DR** → This project is supposed to create clusters of similar "looking" landscapes. Those should be easily viewable in Google Earth Pro. 

## Future considerations

- I might create a connection or compatible files for GIS-Software like qGIS.

## Why?

The story behind this project is a personal travel story. I was on vacation in Scotland where I instantly fell in love with the landscape. On my various vacations in Germany I got the impression that there must be kinds of landscapes similar to the diverse Scottish ones across Europe. That created the idea behind the project:

**To quickly and visually find comparable landscapes that are experienced similarly by hiking and driving.** 

## What’s new

This is a code refacoring of my original algorithm created in late 2024. Since then I had intense courses in python and related topics. So it was time to improve the code in many regards:

- DEM data is now directly loaded via API (unless it’s already been downloaded) as GeoTIFFs:
  In the first version coordinates and DEM files were hardcoded into the notebook...
- There are supposed to be a few different kind of outputs
  1. **RGB GeoTIFFs** that depict the created clusters visually. 
  2. **KML Files** (Keyhole Markup Language) that refer to the GeoTIFFs above. These files are optional as the GeoTIFFs should already have sufficient metadata to be loaded into Google Earth Pro directly.
  3. **SVG Contours** that are vector data (instead of pixel data like GeoTIFFs), to be used in illustrative software like Adobe Illustrator, McNeel Rhino or Adobe Photoshop. The final purpose is to create nicer illustrations for this project, which I will add to this readme in the future.
- Support for configuration files:
  All settings, like the number of clusters, original image coordinates and so on were hardcoded into the original notebook. With this new version there’ll be configuration files (JSON) that summarize the settings to be used by the notebook.

## How to use

Once the project reaches a usable status again (like the original code), I will put instructions here. 

There’ll be a primary and some secondary notebooks for different purposes.

### Rough idea for the division of the notebooks

Primary:

- The one that creates the clusters in the abovementioned formats

Secondary:

- One to load DEM data
- One to create the SVG files from the found clusters

## Results

Once there are visual results, I’ll put them here.

## What cannot be achieved (, yet)

While it is quite fascinating to see that the clusters found by the algorithm indeed appear similar in their topographical structure, there are, of course, aspects that are neglected in this approach entirely:

- Flora and fauna
- Climate
- Soil types
- And more meta aspects like
  - cultural ones like
    - the people
    - the language
  - civilization related
    - typical infrastructure
    - signposts
    - population density
    - architecture

Including those, would very much likely result in clusters that are country specific, unless the scale of some of the features is reduced again. In plain english that means: There’s no place like Scotland, except Scotland. :-)
