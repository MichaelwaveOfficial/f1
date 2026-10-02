from pathlib import Path
import os
from dotenv import load_dotenv

# Force reload and overwrite existing environment variables
load_dotenv(override=True)


class Settings(object):

    # API STUFF.
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    WORKSPACE_ID = os.getenv("WORKSPACE_ID")
    CAR_DETECTOR_ID = os.getenv("CAR_DETECTOR_ID")
    TEAM_CLASSIFIER_ID = os.getenv("TEAM_CLASSIFIER_ID")

    # CURRENT WORKING DIRECTORY
    WORKING_DIR = Path.cwd()

    # DATA DIRECTORIES
    DATA_DIR = WORKING_DIR / "data"
    INPUTS_DIR = WORKING_DIR / "inputs"
    OUTPUTS_DIR = WORKING_DIR / "outputs"

    # Hardcoded for MVP.
    test_image = INPUTS_DIR / "three-cars-1.jpg"

    # MOCK DBs or whatever.
    DRIVER_JSON = DATA_DIR / "drivers-db.json"
    TEAM_ALIASES_JSON = DATA_DIR / "team-aliases.json"
    TEAM_COLOURS_JSON_PATH = DATA_DIR / "team-colours.json"

    # Image processing Vars
    crop_padding = 15


settings = Settings()
