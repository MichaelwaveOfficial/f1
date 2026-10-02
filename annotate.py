from utils import load_json 
from config import settings

import cv2 

team_colours_json = load_json(settings.TEAM_COLOURS_JSON_PATH)


def annotate_car_bbox(raw_frame, bbox, colour=(0, 255, 0)):

    ''' Annotate the detected F1 car within the frame. '''

    annotated_frame = raw_frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]

    x1, y1, x2, y2 = bbox

    # Keep the box within the image.
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))


    # Draw a rectangle around the detected F1 car.
    cv2.rectangle(
        annotated_frame,
        (x1, y1),
        (x2, y2),
        colour,
        2
    )

    return annotated_frame


def annotate_team_colour(team_name):

    ''' Get the team's BGR colour from team-colours.json. '''

    team_data = team_colours_json.get(team_name, {})

    return tuple(team_data.get('bgr', [0, 255, 0]))


def annotate_team_label(raw_frame, bbox, team_name):

    ''' Add the team name above the car. '''

    annotated_frame = raw_frame.copy()
    x1, y1, _, _ = bbox

    team_colour = annotate_team_colour(team_name)

    cv2.putText(
        annotated_frame,
        team_name.upper(),
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        team_colour,
        1,
        cv2.LINE_AA,
    )

    return annotated_frame


def annotate_driver_label(raw_frame, bbox, driver_name):

    ''' Add the driver name inside the car box. '''

    annotated_frame = raw_frame.copy()
    x1, y1, _, _ = bbox

    # Dark outline keeps the name readable over the car.
    cv2.putText(
        annotated_frame,
        driver_name.upper(),
        (x1 + 4, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        driver_name.upper(),
        (x1 + 4, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated_frame