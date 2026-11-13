# Generative Models on Aerial Roof Images

**Guest lecture kit · The Roux Institute at Northeastern University · November 19, 2026, 4:00–5:30 PM**
Dr. Sushanth Manakhari · hosted by Dr. Umesh Hodeghatta Rao · AAI6640, Module 13: Generative Models

> 🚧 **Work in progress.** The notebooks, the roof image packs and the Mac, Windows and Colab setup guides
> are added before the lecture. Students receive the final link from the course instructor.

## What you will build

You join the property team of a fictional home insurer. Every roof in this lab comes from free, public
aerial photographs. Four tickets, one per part of Module 13:

| Ticket | Model | What it does for the team |
|---|---|---|
| **ROOF-301** | Autoencoder (AE) | Squeezes a roof photo through a small bottleneck and rebuilds it. Roofs it rebuilds badly are the unusual ones worth a second look. |
| **ROOF-302** | Variational autoencoder (VAE) | Learns a smooth "roof space": sample brand-new roofs and morph one roof into another. |
| **ROOF-303** | Generative adversarial network (GAN) | A generator and a critic compete until the fakes look real. You will also break it on purpose (mode collapse). |
| **ROOF-304** | Ethics | Synthetic images, bias across regions, deepfakes and responsible deployment. |

## Data, in one paragraph

Roof images are cut from public aerial photographs (Maine GeoLibrary town imagery, with other free U.S.
sources for variety) using public building outlines (USA Structures, from Oak Ridge National Laboratory
and FEMA). Each image is a small, turned crop of one roof. The packs contain **no addresses, coordinates,
building IDs or owner details**. Sources, licences and what is deliberately left out:
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Repository layout (so far)

```
scripts/build_roof_pack.py   builds a roof image pack from public map services (instructor tool)
tests/                       offline tests for the pack builder (no downloads)
docs/DATA_SOURCES.md         sources, licences and privacy rules for the data
src/genroofs/                course package: data loading, models, metrics (in progress)
notebooks/                   lecture notebooks (coming)
```

## Licence

Code: MIT ([LICENSE](LICENSE)). Imagery and building outlines stay under their own terms, listed in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md); every roof pack ships with an `ATTRIBUTION.txt`.
