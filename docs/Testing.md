## Testing pipeline

The testing pipeline consists of unit tests for a preprocessing module, and a module responsible for reading a configuration file.

Test cases are divided into three blocks:
- checking the configuration file
- testing face detection module
- testing the frame generation module

### Usage

Run the files found in ```tests/preprocessing``` with the command:

```
python test_configuration_file_verification.py 
& python test_face_detection.py
& python test_frame_generation.py
```

### Test data

The set of files for testing can be found in ```data/tests``` folder.

### Description of the tests

```test_configuration_file_verification``` checks whether the types and values of the fields of the generated configuration file comply with the constraints that were set in the configuration file schema. The YAML schema shown below allows us to easily and accurately check the correctness of filling the original configuration file:
```
video:
  prediction:
    input_directory: str(required=False)
    input_type: str()
    frame_per_second: int(min=1)
    is_rpi: bool()
    model_type: str()
    model_format: str()
    model_directory: str(required=False)
    output_type: str(required=False)
    log_directory_path: str(required=False)
    camera:
      CAP_PROP_FRAME_WIDTH: int(min=1)
      CAP_PROP_FRAME_HEIGHT: int(min=1)
    preprocessing:
      face_detector: bool()
      face_alignment: bool()
      spatial_normalization: bool()
    model_input_shape:
      - channels: int(min=1)
      - height: int(min=1)
      - width: int(min=1)
    gray_color: bool()
    ir_run:
      model: str(required=False)
      cpu_extension: str(required=False)
      device: str(required=False)
```

```test_face_detection``` checks whether the module is capable of recognizing faces in images and saving them to disk. Testing is carried out for various input data, such as: 
- a set of images with a human face
- a set of images without a human face  
- invalid input data

```test_frame_generation``` checks if the module is capable of generating and saving images from the provided video file at a given frame rate. Testing is carried out for various input data, such as: 
- video containing a frame with a human face
- video that does not contain a frame with a human face
- invalid input data.