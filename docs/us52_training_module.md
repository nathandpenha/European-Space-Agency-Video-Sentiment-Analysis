# Training module 
## Initial Setup 

- Open the `trainer_config.yaml` file from the `src/training` directory 
- In the config file set the `data_dir` to directory where the training data are stored.
    - e.g data_dir: `path/to/videos/`
    - The training data structure should look like as the following structure:<br/>
   ```
    ../videos
     |___Actor1
     |   | filename01.mp4
     |   | filename02.mpd
     |   | ...
     |___Actor2
     |   | filename01.mp4
     |   | filename02.mp4
     |   | ...
     |   .
     |   .
     |___Actor24
     |   | filename01.mp4
     |   | filename02.mpd
     |   | ...
   ```
  
- In the config file set the `model_dir` to directory where the trained model will be saved. By default 
   it will be saved to `models` directory of this repository.
- Other training parameters can be modified in the config file as well, if needed.


## Dependency Installation

 - To Install dependencies,  run `pip install -r requirements.txt`

## How to run 

 - Run the script using `python training.py` 
#### Note: 
- To test the functionality of the script, only three actor videos are used.
