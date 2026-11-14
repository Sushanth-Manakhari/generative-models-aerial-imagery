# Data sources and licences

Every piece of data in this project is free, public and usable without an account or API key.
Licences and services were checked on November 12, 2026, before any pack was built. The machine-readable
list the build script uses is [`scripts/roof_sources.json`](../scripts/roof_sources.json).

## The two roof packs

| Pack | What it is | Roofs | Used for |
|---|---|---|---|
| `roofs_maine_2022` | Eight southern-Maine towns, spring 2022 (leaf-off), one photo program | about 22,000 | Training the autoencoder, VAE and GAN |
| `roofs_us_variety` | Five U.S. regions with different climates and building styles | about 3,000 | Testing: "is this roof unusual for a Maine-trained model?" |

Each roof is a 24 m × 24 m square cut around one home and turned so the long side of the roof is horizontal.
The yard around the roof is kept. Images are 128 × 128 JPEG (the lecture trains at 64 × 64).

**Where to get them:** release `data-v1` of this repository has `roofs_maine_2022.zip` (about 213 MB) and
`roofs_us_variety.zip` (about 29 MB). The size and SHA-256 checksum of each zip are in
[`packs/`](../packs/) and on the release; the notebooks download the zips and check them automatically.

### Known limits (checked by eye on random samples)

- **Training pack:** in 200 random roofs, about **4% show no house** (a wooded or empty lot, a pool area) and
  about **3% are unclear** (deep shadow, a house under construction). The public building outlines and the
  spring 2022 photos were not made at the same time, and trees confuse the automatic roof check. We kept these
  images on purpose. In the autoencoder exercise (ROOF-301, "find the strangest roofs") they turn up at the
  *most ordinary* end of the ranking, not the strangest: lawns, woods and empty lots are easy to copy, so a
  reconstruction score misses "no house here". The exercise shows both ends so the class can discuss that limit.
- **Variety pack:** 75 random roofs (15 per region) showed no empty lots. Larger homes, common in Florida
  and Arizona, can fill most of the 24 m square.
- Every roof was checked automatically for missing-photo areas, including the black-and-white checkerboard
  the Texas service draws where it has no photo; none remain.

## Aerial photos

| Source | Area used | Licence / terms | Attribution printed |
|---|---|---|---|
| Maine GeoLibrary, municipal orthoimagery 2022 (3-inch and 6-inch, leaf-off) | Gorham, Windham, Sanford, York, Kennebunk, Old Orchard Beach | Each service's licence field: **"Use constraints: None"**; acknowledgement appreciated | *Imagery: Maine GeoLibrary, Maine Orthoimagery – Municipal ⟨town⟩ 2022* |
| Maine GeoLibrary, South Portland 2022 (3-inch) | South Portland | The service carries no licence text of its own. The GeoLibrary's statewide orthoimagery metadata says **"No restrictions apply to this data."** (caveat accepted by the instructor) | *Imagery: Maine GeoLibrary, Maine Orthoimagery – South Portland 2022* |
| Maine GeoLibrary, Wells 2022 (6-inch, leaf-off) | Wells | The service's licence field holds a description of the 2022 statewide program instead of terms, so no use constraints are stated. Same Maine GeoLibrary program as the other towns. (Included at the instructor's decision.) | *Imagery: Maine GeoLibrary, Maine Orthoimagery – Municipal Wells 2022* |
| State of Connecticut, 2023 Statewide Orthoimagery (3-inch), served by CT ECO | West Hartford neighbourhoods | Metadata: **"Use constraints: None"**; acknowledgement of the providing organization appreciated | *Imagery: State of Connecticut, 2023 Statewide Orthoimagery, served by CT ECO (UConn CLEAR and CT DEEP)* |
| Indiana Geographic Information Office (IGIO), Indiana Orthoimagery | Fort Wayne neighbourhoods (Allen County, 2024) | **CC0 1.0**, stated on the service | *Imagery: Indiana Geographic Information Office (IGIO)* |
| Texas Geographic Information Office (TxGIO), StratMap 2021 orthoimagery (6-inch, leaf-off) | Round Rock neighbourhoods | **Public domain** under the Texas Strategic Mapping Program (StratMap). This is not the separately licensed Texas Imagery Service, which we do not use. | *Imagery: TxGIO, StratMap 2021, with the Capital Area Council of Governments and local partners* |
| Miami-Dade County GIS, 2025 aerial imagery (public map tiles, about 7-inch) | Southwest Miami-Dade neighbourhoods (Kendall area) | The tile service carries **no licence text**. Basis: Florida's Public Records Law (Florida Statutes, Chapter 119); a Florida appeals court held in *Microdecisions, Inc. v. Skinner* (2004) that a county office cannot claim copyright in its GIS public records. (Caveat accepted by the instructor; this is not legal advice.) | *Imagery: Miami-Dade County GIS, 2025 aerial imagery* |
| City of Tempe, 2024 Aerial Imagery (public map tiles, about 5-inch) | South Tempe neighbourhoods | **CC BY 4.0** | *Imagery: City of Tempe, 2024 Aerial Imagery (CC BY 4.0)* |

Service addresses are in `scripts/roof_sources.json` and in each pack's `ATTRIBUTION.txt`.

## Building outlines: USA Structures (ORNL / FEMA)

- **Service:** https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0
- **What it is:** outlines of every U.S. building larger than about 450 square feet, with a building-use class
  (for example *Residential*, *Single Family Dwelling*).
- **Terms:**
  - The Esri Living Atlas item states a Creative Commons Attribution 4.0 (CC BY 4.0) licence.
  - data.gov lists the dataset (FEMA-0209) under U.S. Government Works.
  - We follow the stricter of the two: CC BY 4.0, with attribution.
- **How we use it:** to find each home, measure it, centre it and turn it straight. Public outlines are often a
  few metres off the photo, mostly by the same amount across a neighbourhood, so the build script measures that
  shift once per neighbourhood (the median over many houses) and corrects it. Each house is then nudged up to
  2.5 m onto its own roof edges when that position is clearly best. A house is left out when its outline does not
  sit on roof edges (an empty lot, trees, or a home built after the photo was taken), and a whole neighbourhood
  block is skipped when most of its outlines miss.
- **Privacy by design:**
  - The service also holds street addresses.
  - The build script requests only three fields: `OCC_CLS`, `PRIM_OCC` and `SQMETERS`.
  - It never asks for addresses, owner data or building IDs, and the outlines are not included in the packs.
- **Attribution we print:** *Building outlines: USA Structures – Oak Ridge National Laboratory (ORNL) and FEMA
  Geospatial Response Office, CC BY 4.0.*

## Address lookup (lecture demo only): two free geocoders, one public building

The pipeline demo (`genroofs.pipeline`) looks up **one public building**, South Portland City Hall at
25 Cottage Road, never a private home. It asks two free geocoders on purpose, because they disagree: the
Census point lands on the street about 36 m from the roof, while OpenStreetMap's lands on the roof. The class
sees both photo squares and discusses which "roof" a pipeline would score. Nothing from either service is
stored in the packs or written to disk.

**U.S. Census Bureau Geocoder**

- **Service:** https://geocoding.geo.census.gov/geocoder/ (documentation:
  https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html)
- **Terms:** free U.S. government service; no key or account.
- **How it finds the point:** it matches the address to a street segment and interpolates along the segment's
  address range, so the point sits on the street, not on a building.

**OpenStreetMap Nominatim** (added November 13, 2026)

- **Service:** https://nominatim.openstreetmap.org/ (usage policy:
  https://operations.osmfoundation.org/policies/nominatim/)
- **Terms:** free, no key or account. The data is © OpenStreetMap contributors under the Open Database Licence
  (ODbL 1.0), so the demo prints that credit.
- **Usage policy:** at most one request per second for all users of an application together, a User-Agent
  that names the application, and no bulk lookups. `genroofs` names itself in every request, and by default the
  demo uses the answer Nominatim gave on November 13, 2026 (a public building does not move). Only the
  instructor asks again, once, with `osm="live"`, so a full class never sends a burst of requests.
- **How it finds the point:** it finds the building by its name (*South Portland City Hall*) and returns the
  centre of its outline, so the point sits on the roof.

**Photo for the demo:** the same Maine GeoLibrary South Portland 2022 service listed above, a 24 m square
around each point. **Credit printed by the demo:** *Geocoding: U.S. Census Bureau Geocoder (public domain) and
OpenStreetMap Nominatim (data © OpenStreetMap contributors, ODbL). Photo: Maine GeoLibrary, Maine
Orthoimagery – South Portland 2022.*

## What the roof packs contain, and what they never contain

| Contains | Never contains |
|---|---|
| Roof pictures (24 m squares, yard kept, turned straight) | Street addresses, owner names |
| Roof area, long and short side, turn angle | Coordinates, parcel or building IDs |
| Building-use class (for example *Single Family Dwelling*) | Building outlines |
| Town, state and photo year | Anything that locates a specific home |
| `ATTRIBUTION.txt`, `README.txt`, `SHA256SUMS.txt`, `build_report.json` | |

The build script refuses to zip a pack if its privacy check finds a location-like field or number.

## Deliberately not used

| Not used | Why |
|---|---|
| Any company's private data, images, vendor names, project names or figures | Proprietary |
| Commercial aerial or satellite imagery | Not free for students |
| Online map tiles from consumer map apps | Their terms forbid this kind of reuse |
| Commercial geocoding APIs | They need a key or a paid account |
| OpenStreetMap map tiles and bulk geocoding | Their usage policies forbid heavy use; the demo makes one lookup at most |
| South Florida Water Management District county imagery (including Miami-Dade 2023) | Now requires a login |
| Hillsborough County (Tampa) imagery | Restricted vendor licence |
| Maricopa County imagery | Sold, not free |
| Texas Imagery Service | Licensed to subscribing agencies only |
| USDA NAIP (used only in the first data test) | Free, but about 0.6 m per pixel is too blurry for roofs |
| AIRS (Aerial Imagery for Roof Segmentation) | Its licence is listed as unknown |
