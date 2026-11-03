from utils import crop_frame


def detect_tcam(detections_bbox, raw_frame):

    '''
        Takes the raw frame, applies the detection bboxes, crops the top of the car bbox and returns the tcam co-ords.
    '''

    car_x1, car_y1, car_x2, car_y2 = detections_bbox

    # T-cam search region.
    y1 = 0
    y2 = max(1, round((car_y2 - car_y1) * 0.075))
    x1 = round((car_x2 - car_x1) * 0.4)
    x2 = max(x1 + 1, round((car_x2 - car_x1) * 0.6))

    return {
        'x1' : x1,
        'y1' : y1,
        'x2' : x2,
        'y2' : y2
    }


''' Generic helper methods to unpack response JSON from ROBOFLOW API. '''


def get_inference_time(response_json : dict) -> float:
    '''
        Get the inference time from the response JSON.
    '''
    return response_json.get("time", 0.0)


def get_detections(response_json : dict) -> list:
    '''
        Get the bounding boxes from the response JSON.
    '''
    return response_json.get("predictions", [])


def get_detection_info(car_id : int, response_json : dict) -> dict:
    '''
        Get the bounding box coordinates and class label from a single detection.
    '''

    detections = get_detections(response_json)
    index = car_id - 1

    if index < 0 or index >= len(detections):
        raise IndexError(f"Detection {car_id} is not in this response")

    detection = detections[index]
    x = float(detection.get("x", 0))
    y = float(detection.get("y", 0))
    width = float(detection.get("width", 0))
    height = float(detection.get("height", 0))

    return {
        "id": car_id,
        "bbox": {
            "x1": round(x - width / 2),
            "y1": round(y - height / 2),
            "x2": round(x + width / 2),
            "y2": round(y + height / 2),
        },
        "class_label": detection.get("class", ""),
        "class_id": detection.get("class_id", -1),
        "confidence": detection.get("confidence", 0.0),
    }


def fetch_no_of_detections(response_json : dict) -> int:
    '''
        Get the number of detections from the response JSON.
    '''
    return len(get_detections(response_json))
