from utils import crop_frame


def detect_tcam(detections_bboxes, raw_frame):

    '''
        Takes the raw frame, applies the detection bboxes, crops the top of the car bbox and returns the tcam co-ords.
    '''

    car_x1, car_y1, car_x2, car_y2 = detections_bboxes

    # Your chosen T-cam search region.
    y1 = 0
    y2 = max(1, round((car_y2 - car_y1) * 0.075))
    x1 = round((car_x2 - car_x1) * 0.4)
    x2 = max(x1 + 1, round((car_x2 - car_x1) * 0.6))

    tcam_crop = crop_frame(
        frame=raw_frame,
        bbox=(car_x1 + x1, car_y1 + y1, car_x1 + x2, car_y1 + y2),
        padding=5
    )

    return tcam_crop


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


def get_detection_info(detection : dict) -> tuple:
    '''
        Get the bounding box coordinates and class label from a single detection.
    '''

    return {
        'x1' : detection.get("x", 0),
        'y1' : detection.get("y", 0),
        'x2' : (
            detection.get("x", 0) + detection.get("width", 0)
            ),
        'y2' : (
            detection.get("y", 0) + detection.get("height", 0)
            ),
        'class_label' : detection.get("class", ""),
        'class_id' : detection.get("class_id", -1)
    }


def fetch_no_of_detections(response_json : dict) -> int:
    '''
        Get the number of detections from the response JSON.
    '''
    return len(get_detections(response_json))
