"""Track-level team and driver identification for annotated race videos."""

import cv2

from classify import classify_team, classify_tcam_colour
from config import settings
from driver_lookup import canonical_team_name, lookup_driver
from roboflow_client import run_rf_detr


def _clip_box(frame, bbox, padding=0):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(frame_width, x1 - padding))
    y1 = max(0, min(frame_height, y1 - padding))
    x2 = max(0, min(frame_width, x2 + padding))
    y2 = max(0, min(frame_height, y2 + padding))
    return x1, y1, x2, y2


def _crop_car(frame, bbox, padding=15):
    x1, y1, x2, y2 = _clip_box(frame, bbox, padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _crop_tcam(frame, bbox):
    """Crop the small T-cam area near the rear half of this clip's cars."""
    x1, y1, x2, y2 = _clip_box(frame, bbox)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 3 or box_height < 3:
        return None

    # In this clip the T-cam colour strip sits around 63–67% across the car,
    # roughly 4–6% below the box top. This avoids the surrounding livery/grass.
    crop_x1 = x1 + round(box_width * 0.63)
    crop_x2 = x1 + round(box_width * 0.67)
    crop_y1 = y1 + round(box_height * 0.04)
    crop_y2 = min(y2, y1 + max(1, round(box_height * 0.06)))
    crop_x1 = max(0, min(frame.shape[1], crop_x1))
    crop_x2 = max(crop_x1 + 1, min(frame.shape[1], crop_x2))
    crop_y2 = max(crop_y1 + 1, crop_y2)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def identify_car(frame, bbox):
    """Identify one representative tracked car crop.

    The team classifier is called once per track. T-cam colour is read locally,
    then combined with the team to look up the driver.
    """
    tcam_crop = _crop_tcam(frame, bbox)
    tcam_colour = classify_tcam_colour(tcam_crop)

    car_crop = _crop_car(frame, bbox, padding=15)
    if car_crop is None or car_crop.size == 0:
        return {
            "team": "Unknown Team",
            "team_confidence": "0.0%",
            "tcam_colour": tcam_colour,
            "driver": "Unknown Driver",
            "number": None,
        }

    credentials = (
        settings.ROBOFLOW_API_KEY,
        settings.WORKSPACE_ID,
        settings.TEAM_CLASSIFIER_ID,
    )
    if not all(credentials):
        team_name = "Unknown Team"
        team_confidence = "0.0%"
    else:
        rgb_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
        response = run_rf_detr(
            frame=rgb_crop,
            workspace_id=settings.WORKSPACE_ID,
            model_id=settings.TEAM_CLASSIFIER_ID,
            api_key=settings.ROBOFLOW_API_KEY,
        )
        team_result = classify_team(response)
        team_name = canonical_team_name(team_result["name"]) or "Unknown Team"
        team_confidence = team_result["confidence"]

    driver_result = lookup_driver(team_name, tcam_colour)
    return {
        "team": team_name,
        "team_confidence": team_confidence,
        "tcam_colour": tcam_colour,
        "driver": driver_result["driver"],
        "number": driver_result["number"],
    }
