# STERN Video module

STERN Video module repository contains the software, testing pipeline, and sub-modules such as preprocessing, training, and inference. 

- [Preprocessing](#preprocessing) module extracts frames from video and preprocess the frames. 
- [Training](#training) module provides an API to train the models on the extract features. 
- [Inference](#inference) module analyses the emotion based on the frame(s). 
- [Testing pipeline](#testing-pipeline) pipeline that assess the models and tests the other system modules such as the different preprocessing steps. 

## Repository structure

- data: this directory is used for the original data related to the solution. Treat the data in this directory as immutable.
- docs: contains the readme.md files and relevant images for documentation.
- models: contains the video models such as ANNs that are used by the inference module.
- src : contains the STERN video module and configuration files. 
  - inference: contains script that is able to detect emotion from camera stream or input video.
  - output: contains scripts that receive inference result as an input from inference module and show the result to the GUI of the live streaming or log the result into a json log file. 
  - preprocessing: contains scripts that extract frames from video and perform data preprocessing steps.
  - training: contains model building and optimization and model training scripts with configuration files. 
- test: contains test scripts with configuration file.
- requirements.txt: contains all the dependencies which need to be installed in order to run the STERN video software.

## Requirements

` pip ` is already installed with `Python` downloaded from [python.org](https://www.python.org/). 

The software requires `Python3.7+`. 

All the Python dependencies needed to run the software are listed in [requirements.txt](requirements.txt). 

The following command installs the packages according to the requirements file. 
```
pip install -r requirements.txt 
```   

**NOTE:** On the Raspberry Pi, `python2` and `python3` both are installed. 

In order to prevent cluttering the Raspberry Pi's system libraries, a `Python Virtual Environment` is recommended. The following command is used to setup the environment. 

```
pip install virtualenv

virtualenv [name]
```
To activate the python virtual environment by running the following command:

```
source [name]/bin/activate
```

To deactivate the python virtual environment by running the following command:

```
deactivate
```

## Tutorials (Use cases)

### Preprocessing 
Video preprocessing aims to process data and convert them to proper form for training and inference module. 
There are four video preprocessing techniques, including frame generator, face detector, face alignment, spatial normalizer. 
For more details on its usage, please read [preprocessing module](./docs/Preprocessing.md)

### Training
The model is trained and optimized with the preprocessed data by the training module. 
The training phase includes three steps that are data preparation, model training, and model optimization. 
Every step is configurable since each step has its configuration file with various parameters. 
For more information about training the model, please read [training module](./docs/Training.md) 

### Inference
Inference module can capture frame from camera stream or input video, and detect facial emotion. 
It uses preprocessing module to prepare date and output module to display the results. 
It is configurable with configuration yaml file. For more details on inference, please read [Inference module](./docs/Inference.md)

### Testing pipeline
[Testing pipeline](./docs/Testing.md)
