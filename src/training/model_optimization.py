"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Description: This script optimizes a trained model using OpenVino API.
@Last modified date: 27-11-2020
"""

import os
import yaml


class ModelOptimization:
    """
    This class is used to optimize a trained model
    """
    def optimize_model(self, model_type, parameters):
        """
        This function optimizes a given cnn model
        :param model_type: type of model architecture
        :param parameters: optimizer  including paths
        :type parameters: dict
        """

        # set openvino installation path
        env_setup_path = parameters["openvino_inst_dir"]
        open_vino_opt = parameters["openvino_optimizer_dir"]
        model_input_path = parameters["model_input_path"] + '/' + parameters["trained_model_name"]
        optimized_out_path = parameters["optimized_output_path"] + '/' + parameters["trained_model_name"]
        input_shape = None
        model_input_shape = parameters["model_input_shape"]
        depth = parameters["depth"]

        if model_type == "cnn":
            input_shape = (1, model_input_shape[0], model_input_shape[1], model_input_path[2])
        if model_type == "3d_cnn":
            input_shape = (1, model_input_shape[0], model_input_shape[1], depth, model_input_shape[2])

        if not os.path.exists(optimized_out_path):
            os.makedirs(optimized_out_path)
        os.chdir(env_setup_path)
        os.system('setupvars.bat')
        os.chdir(open_vino_opt)
        os.system('python mo_tf.py --saved_model_dir' + ' ' + str(model_input_path) + ' ' +
                  '--input_shape' + ' ' + str(input_shape).replace(" ", "") + ' ' + '--output_dir' + ' ' + str(
            optimized_out_path)
                  + ' ' + '--model_name' + ' ' + str(parameters["trained_model_name"]))


if __name__ == '__main__':
    with open("config/training_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    parameters = config["train_params"]
    model_type = config["model_type"]
    optimizer = ModelOptimization()
    optimizer.optimize_model(model_type, parameters)
