# Generative Models on Aerial Roof Images

**Guest lecture kit · The Roux Institute at Northeastern University · November 19, 2026, 4:00–5:30 PM**
Dr. Sushanth Manakhari · hosted by Dr. Umesh Hodeghatta Rao · AAI6640, Module 13: Generative Models

> **Notebooks:** [`notebooks/`](notebooks/): a five-minute pre-flight check to run before class, and the
> hands-on lab. Each opens in Google Colab.

## What you will build

You join the property team of a fictional home insurer. Every roof in this lab comes from free, public
aerial photographs. Four tickets follow the parts of Module 13, and a fifth puts the models to work on a real
hurricane:

| Ticket | Model | What it does for the team |
|---|---|---|
| **ROOF-301** | Autoencoder (AE) | Squeezes a roof photo through a small bottleneck and rebuilds it. Roofs it rebuilds badly are the unusual ones worth a second look. |
| **ROOF-302** | Variational autoencoder (VAE) | Learns a smooth "roof space": sample brand-new roofs and morph one roof into another. |
| **ROOF-303** | Generative adversarial network (GAN) | A generator and a critic compete until the fakes look real. You will also break it on purpose (mode collapse). |
| **ROOF-304** | Ethics | Synthetic images, bias across regions, deepfakes and responsible deployment. |
| **ROOF-305** | After the storm | Real photos from the day after Hurricane Michael (2018) with FEMA's damage labels: see why the unusual score fails, teach a small damage model with the critic's loss, and rank homes for adjusters as a property and casualty insurer would, with a person deciding every claim. |

**Then the modern fixes.** Autoencoders and VAEs blur; GANs are sharper but draw strange shapes. The last part
runs two free, open enlargers, **Real-ESRGAN** (a GAN) and **SwinIR** (a Swin Transformer), on the same roofs
and on our models' pictures, and shows what they can and cannot fix: they sharpen detail that is still in a
picture, and what they add is believable, not necessarily true. Then it runs a better autoencoder, **Stable
Diffusion's own** (MIT licence), whose copies of the same roofs are sharp, to show that the real fix for blur is a
better model. Credits and licences: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md#models-used-in-the-lecture).

## Data, in one paragraph

Three roof packs, all cut from free, public aerial photographs with free, public building outlines (USA
Structures, from Oak Ridge National Laboratory and FEMA): a **training pack** of about 22,000 roofs from eight
southern-Maine towns (spring 2022), a **variety pack** of about 3,000 roofs from Connecticut, Indiana, Texas,
Florida and Arizona, and a **storm pack** of about 2,000 roofs from five Florida towns photographed by NOAA the day
after Hurricane Michael (October 2018), each with FEMA's public damage level for the home. Each image is a 24 m
square around one roof, turned straight. The packs contain **no
addresses, coordinates, building IDs or owner details**. One demo looks up a single public building (South
Portland City Hall) with two free geocoders, the U.S. Census Bureau Geocoder and OpenStreetMap Nominatim
(© OpenStreetMap contributors, ODbL), to show how an address becomes a score. Sources, licences and what is
deliberately left out: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Setup

**In class, everyone uses Google Colab**, which needs no setup beyond one install cell in each notebook.
To run the code on your own Mac or Windows laptop afterwards, follow [docs/SETUP.md](docs/SETUP.md): one script
creates a `.venv` inside the project folder, installs everything there, and runs `python -m genroofs.check`.
The Mac setup is tested; the Windows setup is provided but not yet tested on a real Windows PC.

## Repository layout

```
src/genroofs/                course package: data, models, training, metrics, pictures, exercises,
                             the storm triage, the pipeline demo, an in-notebook explorer panel and a
                             setup self-check
tests/                       offline tests for the package and the pack builder (no downloads)
scripts/setup_mac.sh         laptop setup for macOS
scripts/setup_windows.ps1    laptop setup for Windows
scripts/build_roof_pack.py   builds the roof packs from public map services (instructor tool)
scripts/roof_sources.json    every photo source, its licence and attribution
packs/                       size and SHA-256 of each roof pack (the zips are on release data-v1)
                             (ready-made trained models: release models-v1)
docs/SETUP.md                Colab, Mac and Windows setup, step by step
docs/DATA_SOURCES.md         sources, licences, privacy rules and known limits of the data
docs/diagrams/               the lecture's diagrams as draw.io files (open at app.diagrams.net)
notebooks/                   the pre-flight check and the hands-on lab (open in Colab)
pyproject.toml               package metadata (pip install from this repository)
```

## Licence

Code: MIT ([LICENSE](LICENSE)). Imagery, building outlines and the two enlargers' weights stay under their own
terms, listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md); every roof pack ships with an `ATTRIBUTION.txt`.
