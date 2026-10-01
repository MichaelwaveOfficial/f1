import cv2 


def load_json(json_path):

    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def crop_frame(image, bbox, padding=0):

    '''
    Crop the image using the bounding box coordinates.
    '''

    x1, y1, x2, y2 = bbox

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image
