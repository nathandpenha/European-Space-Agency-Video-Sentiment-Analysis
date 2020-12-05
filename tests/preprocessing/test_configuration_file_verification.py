"""
Copyright (c) 2020 TU/e -PDEng Software Technology c2019. All rights reserved.
@Author: Vladimir Romashov v.romashov, Georgios Azis g.azis@tue.nl
Description:
This is a test configuration file script that is able to validate the schema of the
configuration file.

Configuration and configuration schema yaml files; os and yamale libraries were used in
this script.

Last modified date: 02-12-2020
"""

import os
import yamale


def test_config_file_against_schema():
    """
    This method validates the configuration file based on the yaml schema.
    :return The test is complete if the error is not raised.
    """
    project_path = os.path.abspath(os.path.join(__file__, "../../.."))
    schema_path = project_path + '/prod_data/tests/test_config_files/schema_configuration.yml'
    schema = yamale.make_schema(schema_path)
    file_path_configuration = project_path + '/src/configuration.yml'
    data = yamale.make_data(file_path_configuration)
    raised = True
    try:
        _ = yamale.validate(schema, data)
        raised = False
    finally:
        assert raised == False
