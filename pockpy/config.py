""" Loads the configuration settings to be accessed by other modules."""

import os
import yaml
import warnings

# Get path of current file
# NOTE: Realpath is included to first resolve symlinks, probably convoluted
# and might probably be dropped with no impact.
POCKPY_PATH = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

# We want the POCKPY path to be relative to the root of the project
POCKPY_PATH = os.path.dirname(POCKPY_PATH)

# Working directory
WORKING_DIRECTORY_PATH = os.getcwd()

# Check if the current working directory has a .pockpy_config.yml file.
# If it does not, open the local config.yml
try:
    file = open(os.path.join(WORKING_DIRECTORY_PATH, '.pockpy_config.yml'))
    CONFIG_PATH = WORKING_DIRECTORY_PATH
except FileNotFoundError:
    file = open(os.path.join(POCKPY_PATH, 'config.yml'))
    CONFIG_PATH = POCKPY_PATH

__settings = yaml.load(file, Loader=yaml.FullLoader)
file.close()

# Define the settings
KNOB_DEFINITIONS_PATH = __settings['KNOB_DEFINITIONS_PATH']
if __settings['KNOB_DEFINITIONS_PATH_IS_RELATIVE']:
    KNOB_DEFINITIONS_PATH = os.path.join(CONFIG_PATH, KNOB_DEFINITIONS_PATH)

MADX_FILENAME_DEFAULT = __settings['MADX_FILENAME_DEFAULT']

SEQUENCE_NAMES = __settings['SEQUENCE_NAMES']

MINIMUM_TWISS_COLUMNS = __settings['MINIMUM_TWISS_COLUMNS']

CONNECTED_ELEMENTS = __settings['CONNECTED_ELEMENTS']

PICKLE_JAR_PATH = __settings['PICKLE_JAR_PATH']
if __settings['PICKLE_JAR_PATH_IS_RELATIVE']:
    PICKLE_JAR_PATH = os.path.join(CONFIG_PATH, PICKLE_JAR_PATH)

APERTURE_OFFSET_DEFAULTS = __settings['APERTURE_OFFSET_DEFAULTS']

def get_knob_definition():
    """ Returns the definition of all knobs from the provided .yml file. """
    try:
        with open(KNOB_DEFINITIONS_PATH, 'r') as f:
            knob_definition = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        warnings.warn(
            f'No knob definition found in "{KNOB_DEFINITIONS_PATH}".',
            UserWarning,
            stacklevel=1)
        knob_definition = {}
    return knob_definition
