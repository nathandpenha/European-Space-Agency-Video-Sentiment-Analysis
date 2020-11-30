# Training Module 
This training module is a sub module of the Video Sentimental Analysis that has dataset preparation, model training, and
model optimization parts. These are explained one by one as follows.  
### 1. Data preparation script
#### General info:
 *  Used to prepare the training data.
 *  Uses a configuration file `config/data_preparation_config.yaml`
 *  Generates different training dataset versions based on he value of dataset_type in the config file 
    * Set `dataset_type` to `actor` to split the dataset  based on actors
    * Set `dataset_type` to `percentage` to split the dataset based on percentage
 * Executes the preprocessing modules (face_detector, face_alignment, spatial_normalization) according 
   to their order and values in the config file
	* Set the values of these modules  to `True` or `False`
	* The order of these modules can be changed by reordering the list 
 * Other values can be configured as required in the config file.
#### Dependency installation
 * Install `requirements.txt`: run `pip install -r requirements.txt`
#### How to run this script 
 * Clone/pull this  branch
 * Go to the `src/training` directory
 * Download `shape_predictor_68_face_landmarks.dat` from this path on 
 sharepoint: ESA PDEng ST Project/ModlesAndData/Video/Models/Dvelopment 
 or this [link](https://tuenl.sharepoint.com/:u:/r/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/ModelsAndData/Video/Models/Development/shape_predictor_68_face_landmarks.dat)
 * Set the required parameters values in the configuration file in `config/data_preparation_config.yaml`
 * Run  `python data_preparation.py`
### 2. Training script
#### General info:
 * Used to build, train, and evaluate a (3d) cnn type neural networks.
 * Uses a configuration file `config/training_config.yaml`
 * Loads  a preprocessd training, validation, and testing  dataset processed by the data preparation script.
 * Saves the trained model in `.pb` format into a directory.
 * Saves the model training information into a directory for later reference.  
#### Dependency installation
 * Install `requirements.txt`: run `pip install -r requirements.txt`
#### How to run this script 
 * Clone/pull this  branch
 * Go to the `src/training` directory
 * Set the required training parameters and directory path  values in the configuration file in `config/training_config.yaml`
 * Run  `python training.py`
### 3. Model optimization script
#### General info:
 * Used to optimize a trained model using openvino API.
 * Uses a configuration file `config/training_config.yaml`
 * Loads  a trained model to optimize that is saved in `.pb` format .
 * Saves the optimized model in `.xml` and `.bin` format into a directory.  
#### Dependency installation
 * Install `requirements.txt`: run `pip install -r requirements.txt`
 * Install OpenVINO version 2020.4 or higher by following this article: [openvino](https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_windows.html )
#### How to run this script 
 * Clone/pull this  branch
 * Go to the `src/training` directory
 * Set the required optimization parameters and directory path  values in the configuration file in `config/training_config.yaml`
     * Set `openvino_inst_dir` to bin folder path of openvino installation
     * Set `openvino_optimizer_dir` to model_optimizer folder path of openvino installation
     * Set `model_input_path` to folder path of a model to optimize
     * Set `optimized_output_path` to folder path where the optimized model will be saved
 * Run  `python model_optimization.py`