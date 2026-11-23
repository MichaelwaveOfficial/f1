from utils import load_json 
from config import settings

import cv2 

team_colours_json = load_json(settings.TEAM_COLOURS_JSON_PATH)
BBOX_PADDING_MIN = 50
BBOX_PADDING_MAX = 125
BBOX_PADDING_RATIO = 0.035


def _bbox_visual_padding(bbox):
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(BBOX_PADDING_MIN, min(BBOX_PADDING_MAX, round(width * BBOX_PADDING_RATIO)))
    pad_y = max(BBOX_PADDING_MIN, min(BBOX_PADDING_MAX, round(height * BBOX_PADDING_RATIO)))
    return pad_x, pad_y


def annotate_car_bbox(
    raw_frame,
    bbox,
    colour=(255, 190, 0),
    min_padding=BBOX_PADDING_MIN,
    max_padding=BBOX_PADDING_MAX,
):

    ''' Annotate the detected F1 car within the frame. '''

    annotated_frame = raw_frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]

    x1, y1, x2, y2 = (int(value) for value in bbox)

    # Add a small visual-only margin; the source bbox and CSV remain unchanged.
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = max(min_padding, min(max_padding, round(box_width * BBOX_PADDING_RATIO)))
    pad_y = max(min_padding, min(max_padding, round(box_height * BBOX_PADDING_RATIO)))
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    # Keep the padded drawing coordinates within the image.
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

    # Fine outline with heavier corner brackets; no ground marker.
    cv2.rectangle(
        annotated_frame,
        (x1, y1),
        (x2, y2),
        colour,
        1,
        cv2.LINE_AA,
    )
    corner_length = max(28, min(72, int(min(x2 - x1, y2 - y1) * 0.2)))
    corner_thickness = 8
    corners = (
        ((x1, y1), (x1 + corner_length, y1), (x1, y1 + corner_length)),
        ((x2, y1), (x2 - corner_length, y1), (x2, y1 + corner_length)),
        ((x1, y2), (x1 + corner_length, y2), (x1, y2 - corner_length)),
        ((x2, y2), (x2 - corner_length, y2), (x2, y2 - corner_length)),
    )
    for corner, horizontal_end, vertical_end in corners:
        cv2.line(annotated_frame, corner, horizontal_end, colour,
                 corner_thickness, cv2.LINE_AA)
        cv2.line(annotated_frame, corner, vertical_end, colour,
                 corner_thickness, cv2.LINE_AA)

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


def annotate_driver_card(
    frame,
    bbox,
    driver_name,
    team_name,
    team_colour=(255, 190, 0),
    occupied=(),
    preferred_position=None,
):
    """Draw a compact broadcast-style driver/team card and return its layout."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 12
    accent_width = 6
    name_scale, team_scale = 1.2, 0.8
    name = str(driver_name).upper()
    team = str(team_name).upper()

    name_size, _ = cv2.getTextSize(name, font, name_scale, 2)
    team_size, _ = cv2.getTextSize(team, font, team_scale, 1)
    card_width = max(name_size[0], team_size[0]) + padding * 2 + accent_width
    card_height = 88

    # Prefer placement above the car, then below, then beside it. If cards
    # collide, try the remaining positions and finally allow the least crowded.
    center_x = (x1 + x2) // 2
    pad_x, pad_y = _bbox_visual_padding(bbox)
    candidates = {
        "above": (center_x - card_width // 2, y1 - pad_y - card_height - 14),
        "below": (center_x - card_width // 2, y2 + pad_y + 14),
        "left": (x1 - pad_x - card_width - 14, (y1 + y2 - card_height) // 2),
        "right": (x2 + pad_x + 14, (y1 + y2 - card_height) // 2),
    }
    order = ["above", "below", "left", "right"]
    if preferred_position in candidates:
        order.remove(preferred_position)
        order.insert(0, preferred_position)

    def bounded_rect(position):
        left, top = candidates[position]
        left = max(4, min(left, width - card_width - 4))
        top = max(4, min(top, height - card_height - 4))
        return (left, top, left + card_width, top + card_height)

    def overlap_area(rect, other):
        ox1, oy1, ox2, oy2 = other
        dx = max(0, min(rect[2], ox2) - max(rect[0], ox1))
        dy = max(0, min(rect[3], oy2) - max(rect[1], oy1))
        return dx * dy

    options = [(position, bounded_rect(position)) for position in order]
    position, card = next(
        ((pos, rect) for pos, rect in options
         if all(overlap_area(rect, other) == 0 for other in occupied)),
        min(options, key=lambda item: sum(overlap_area(item[1], other) for other in occupied)),
    )
    left, top, right, bottom = card

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    colour = tuple(int(channel) for channel in team_colour)
    cv2.rectangle(frame, (left, top), (left + accent_width, bottom), colour, -1)
    cv2.line(frame, (left + 14, top + 47), (right - 10, top + 47),
             (72, 76, 82), 1, cv2.LINE_AA)

    cv2.putText(frame, name, (left + 16, top + 34), font, name_scale,
                (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, team, (left + 16, top + 75), font, team_scale,
                (205, 208, 212), 1, cv2.LINE_AA)

    # Connect the nearest edge of the card to the car.
    car_anchor = (center_x, y1 - pad_y if position == "above" else y2 + pad_y)
    if position in ("left", "right"):
        car_anchor = (
            x1 - pad_x if position == "left" else x2 + pad_x,
            (y1 + y2) // 2,
        )
    card_anchor = (
        max(left + 10, min(car_anchor[0], right - 10)),
        bottom if position == "above" else top if position == "below"
        else (top + bottom) // 2,
    )
    cv2.line(frame, card_anchor, car_anchor, colour, 1, cv2.LINE_AA)
    return frame, card, position
