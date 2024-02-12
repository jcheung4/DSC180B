# ODDS - Overhead Structure Detection and Data Science

The goal of this project is to ensure up-to-date data quality control with SDG&E's databases of electric pole assets. Our project seeks to create a more efficient way of ensuring that the databases are up-to-data by utilizing the Google Street View API along with a pole-detection model trained using [DETR](https://github.com/facebookresearch/detr/). Our model will traverse the streets between 2 given longitude and latitude coordinate pairs and count the number of wooden and metal electric poles it identifies. The counted poles are compared to the number on record in SDG&E's database which would be a good indicator of whether an update is necessary.

## Data Sources:

# Necessary?

**Training images** are obtained using the [Google Street View Static API](https://developers.google.com/maps/documentation/streetview/overview).

1. The [streetwatch](https://github.com/pdashk/streetwatch) repository outlines how to download all the images into an output folder: `images`.
2. After running `python scripts/collect_images.py` in streetwatch, move the `images` output folder into the `data` directory of this repository.


**Image annotations** in COCO json format will be needed for the model to train on. Training and validation annotations should be placed within the `data/annotations` directory and be named `custom_train.json` and `custom_val.json` respectively.
> [!NOTE]
> For the streetwatch data specifically, the annotations have already been added in the `data/annotations` directory as the original source is a private Sharepoint within our subject domain.

## Setup

### Conda Environment
After cloning repository, navigate to root level and run:
```
conda env create -f environment.yml
```

### PostgreSQL Docker Container
1. Run `lsof -i :5432` to see if anything is currently occupying that port.
    - If so, run `sudo kill [pid]` to get rid of it.
2. Run `docker build -t dsc180b-image .`
3. Run `docker run -p 5432:5432 --name dsc180b-container -d dsc180b-image`

## Running the Traversal Model Script

In order to run the model to identify wooden and metal poles along the streets of 2 coordiante pairs, run:
```
python pole_workflow.py [coordinate pair 1] [coordinate pair 2]
```
> [!NOTE]
> Example coordinate pair: `32.8209644,-117.1861909`
