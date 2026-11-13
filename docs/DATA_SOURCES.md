# Data sources and licences

Every piece of data in this project is free, public and usable without an account or API key.
Checked on November 12, 2026, before any data was downloaded.

| What | Source | Licence / terms | How we use it | Do we redistribute it? |
|---|---|---|---|---|
| Aerial photos of roofs | Maine GeoLibrary, *Maine Orthoimagery – South Portland 2022* | State public data; the statewide orthoimagery metadata says **"No restrictions apply to this data."** | Cut small roof pictures for the training pack | Yes, as small roof pictures, with attribution |
| Building outlines | *USA Structures*, Oak Ridge National Laboratory and FEMA Geospatial Response Office | CC BY 4.0 on the Esri Living Atlas copy; listed on data.gov as a U.S. Government Work | Find each home, turn it straight, black out the yard | **No.** Outlines and addresses stay out of the pack |
| Fallback photos (comparison only) | USDA NAIP, served by USGS *The National Map* | Public domain | Side-by-side quality comparison in the data test | Only in the private data-test pack |
| Address to map position (lecture demo only) | U.S. Census Bureau Geocoding Services | Free public API, no key | Show the first step of an insurer's pipeline for one public building | No |

## 1. Aerial photos: Maine GeoLibrary, South Portland 2022

- **Service:** https://gis.maine.gov/image/rest/services/Municipal/orthoMunicipalSouthPortland2022/ImageServer
- **What it is:** 4-band (red, green, blue, near-infrared) orthoimagery at 7.5 cm per pixel, flown in the
  2022 spring leaf-off season under the Maine GeoLibrary's statewide orthoimagery program. Coordinate system
  NAD83 / UTM zone 19N.
- **Terms:**
  - The service's copyright line reads "Maine GeoLibrary".
  - The service allows image export and download.
  - The GeoLibrary's statewide orthoimagery metadata (for example the 2012, 2013 and 2014 project records)
    states "No restrictions apply to this data."
  - The South Portland 2022 service page itself carries no separate licence text.
- **Attribution we print:** *Imagery: Maine GeoLibrary, Maine Orthoimagery – South Portland 2022.*
- **Optional:** a courtesy note to the Maine GeoLibrary saying the imagery is used for a free university
  lecture. It is not required by the terms above.

## 2. Building outlines: USA Structures (ORNL / FEMA)

- **Service:** https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0
- **What it is:** outlines of every U.S. building larger than about 450 square feet, with a building-use class
  (for example *Residential*, *Single Family Dwelling*).
- **Terms:**
  - The Esri Living Atlas item states a Creative Commons Attribution 4.0 (CC BY 4.0) licence.
  - data.gov lists the dataset (FEMA-0209) under U.S. Government Works.
  - We follow the stricter of the two: CC BY 4.0, with attribution.
- **Privacy by design:**
  - The service also holds street addresses.
  - Our build script requests only three fields: `OCC_CLS`, `PRIM_OCC` and `SQMETERS`.
  - It never asks for addresses, owner data or building IDs.
- **Attribution we print:** *Building outlines: USA Structures – Oak Ridge National Laboratory (ORNL) and FEMA
  Geospatial Response Office, CC BY 4.0.*

## 3. Fallback photos: NAIP via USGS

- **Service:** https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer
- **What it is:** the USDA National Agriculture Imagery Program mosaic, served free by USGS.
- **Terms:** public domain.
- **Use:** comparison only. It covers every U.S. address, but at about 0.6 m per pixel roofs are blurry, and
  the server can be slow.

## 4. Address lookup (lecture demo only): U.S. Census Bureau Geocoder

- **Service:** https://geocoding.geo.census.gov/geocoder/ (documentation:
  https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html)
- **Terms:** free U.S. government service; no key or account.
- **Use:** in class we geocode **one public building**, never a private home, to show the first step of the
  pipeline. Nothing from it is stored in the pack.

## What the roof pack contains, and what it never contains

| Contains | Never contains |
|---|---|
| Small roof pictures (24 m squares) | Street addresses, owner names |
| Roof area, long and short side, angle | Coordinates, parcel or building IDs |
| Building-use class (for example *Single Family Dwelling*) | Anything that locates a specific home |
| An `ATTRIBUTION.txt` file | |

## Deliberately not used

| Not used | Why |
|---|---|
| Any company's private data, images, vendor names, project names or figures | Proprietary |
| Commercial aerial or satellite imagery | Not free for students |
| Online map tiles from consumer map apps | Their terms forbid this kind of reuse |
| AIRS (Aerial Imagery for Roof Segmentation) | Its licence is listed as unknown |
