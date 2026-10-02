from nlp import normalise_text
from config import settings.DRIVER_JSON as driver_json_db
from utils import load_json

driver_db = load_json(driver_json_db)


def lookup_driver(team, tcam_color):
  
    if team is None:
        team = 'Unknown Team'

    if tcam_color is None:
        tcam = 'unknown'

    tcam_normlised = normalise_text(tcam_color)

    driver_record = (
        driver_db
        .get(team, {})
        .get(tcam_normlised)
    )

    return {
        "driver": driver_record['driver'],
        "number": driver_record['number'],
    }
    