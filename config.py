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

    # PATHS.
    WORKING_DIR = Path.cwd()
    DATA_DIR = WORKING_DIR / "data"
    TEST_IMAGES_DIR = DATA_DIR / "pics"
    DRIVER_JSON = DATA_DIR / "drivers-db.json"

    # Vars
    crop_padding = 15


settings = Settings()
