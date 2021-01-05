from enum import Enum

class Paths(Enum):
    RESOURCES_BASE_PATH = "server/resources"
    MODELS_BASE_PATH = "server/resources/models"

    DATASETS_PATH = f"{RESOURCES_BASE_PATH}/datasets"
    ENCODERS_PATH = f"{MODELS_BASE_PATH}/encoders"
    CLASSIFIERS_PATH = f"{MODELS_BASE_PATH}/classifiers"
    TEAM_DATA_PATH = f"{MODELS_BASE_PATH}/team_data"