# ODDS - Overhead Structure Detection and Data Science

The goal of this project is to ensure up-to-date data quality control with SDG&E's databases of electric pole assets. Our project seeks to create a more efficient way of ensuring that the databases are up-to-data by utilizing the Google Street View API along with a pole-detection model trained using [DETR](https://github.com/facebookresearch/detr/). Our model will traverse the streets between 2 given longitude and latitude coordinate pairs and count the number of wooden and metal electric poles it identifies. The counted poles are compared to the number on record in SDG&E's database which would be a good indicator of whether an update is necessary.

## Data Sources:

**Training images** are obtained using the [Google Street View Static API](https://developers.google.com/maps/documentation/streetview/overview).

Running `python scripts/image_collection/collect_images.py` will download all the training images into the `images` folder into the `data` directory of this repository.


## Setup
Create a .env file with variables from Google Cloud:
```
API_KEY='your_api_key'
SECRET='your_secret_key'
```

### Conda Environment
After cloning repository, navigate to root level and run:
```
conda env create -f environment.yml
```

### DETR Model
You must clone the DETR repository in order to train the model:
```
git clone https://github.com/woctezuma/detr.git
cd detr
git checkout finetune
cd ..
```

### Create Training/Validation Datasets and Prepare files to train model
After downloading `images` folder into the `data` directory, split the data into training/validation set by running:
```
python scripts/cocosplit.py --having-annotations --multi-class -s 0.8 annotations/combined-everyone.json data/custom/annotations/custom_train.json data/custom/annotations/custom_val.json
```
[Cocosplit Repo](https://github.com/akarazniewicz/cocosplit)
This will download the model's "base" and split the data into training/validation sets based on the COCO json annotations.

To split the images into its train and validation set, run:
```
python scripts/train-val-split.py
```

### Train the Model
In order to train the model, run the following:
```
python detr/main.py \
  --dataset_file "custom" \
  --coco_path "data" \
  --output_dir "entire_workflow/models" \
  --resume "detr/detr-r50_no-class-head.pth" \
  --num_classes 2 \
  --epochs 10 \
  --device cuda
```
The parameters preceded by "--" may be modified accordingly such as the number of epochs to train for.
> [!IMPORTANT]
> A GPU is required to timely train the model.

After the model is finished training, output files will be saved into `detr/outputs`, and from there, move the model into the `entire_workflow/models` folder.

### PostgreSQL Docker Container
1. Run `lsof -i :5432` to see if anything is currently occupying that port.
    - If so, run `sudo kill [pid]` to get rid of it.
2. Run `docker build -t dsc180b-image .`
3. Run `docker run -p 5432:5432 --name dsc180b-container -d dsc180b-image`

## Running the Traversal Model Script

In order to run the model to identify wooden and metal poles along the streets of 2 coordiante pairs, run:
```
python entire_workflow/pole_workflow.py [coordinate pair 1] [coordinate pair 2]
```
> [!NOTE]
> Example:
> ```
> python entire_workflow/pole_workflow.py '32.8209644,-117.1861909' '32.8195283,-117.1861259'
> ```
