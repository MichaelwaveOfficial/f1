from roboflow_client import run_rf_detr
from detection import get_detection_info, get_no_of_detections, crop_frame
from classify import classify_team, classify_tcam_colour
from config import settings
from utils import crop_frame


def run_pipeline():

    ''' Read image from inputs dir. ''' 

    # Load test image into memory as a raw frame.
    image_path = settings.test_image
    raw_frame = cv2.imread(image_path)

    ''' Detect the F1 car. '''

    # Store data in memory for later use.
    cars = []
    car_frame_crops = []
    tcam_frame_crops = []

    # Transformer to detect F1 cars in the frame using ROBOFLOW API.
    car_detections = run_rf_detr(
        frame=raw_frame,
        workspace_id=settings.WORKSPACE_ID,
        model_id=settings.CAR_DETECTOR_ID,
        api_key=settings.ROBOFLOW_API_KEY,
    )
    # Later eval? 
    no_of_cars_detected = get_no_of_detections(car_detections)
    inference_time = get_inference_time(car_detections)

    # Iterate over each detected F1 car and process it.
    for car_id, car in enumerate(car_detections['predictions'], start=1):
        
        # Get the bounding box coordinates for the detected F1 car.
        detection_data = get_detection_info(car_id, car_detections)
        cars.append(detection_data)

    ''' Crop the detected F1 cars within the frame. '''

    car_frame_crop_padding = 15

    for car in cars:

        car_frame_crop = crop_frame(
            raw_frame,
            detection_data['bbox'],
            padding=car_frame_crop_padding
        )
        car_frame_crops.append(car_frame_crop)

    ''' Classify the F1 car's team from cropped F1 car. '''

    for car_id, (car, cropped_frame) in enumerate(zip(cars, car_frame_crops), start=1):

        # Call the ROBOFLOW API to classify the team of the F1 car.
        team_result = run_rf_detr(
            frame=cropped_frame,
            workspace_id=settings.WORKSPACE_ID,
            model_id=settings.TEAM_CLASSIFIER_ID,
            api_key=settings.ROBOFLOW_API_KEY,
        )

        # Classify the team based on the API response.
        classified_team = classify_team(team_result)
        car['team'] = classified_team

    ''' ID T-Cam Colour from cropped F1 car. '''

    for car in cars:

        tcam_crop_padding = 5

        tcam_bbox = tcam_detect_tcam(car['bbox'], raw_frame)
        car['tcam']['bbox'] = tcam_bbox

        tcam_frame_crop = crop_frame(
            raw_frame,
            tcam_bbox,
            padding=tcam_crop_padding
        )
        tcam_frame_crops.append(tcam_frame_crop)

        tcam_colour = classify_tcam_colour(tcam_frame_crop)
        car['tcam']['colour'] = tcam_colour

        ''' ID the driver from given T-Cam colour and team. '''

        driver_result = lookup_driver(
            team=car['team']['name'],
            tcam_colour=car['tcam']['colour']
        )
        # TODO: Lookup driver func // module. Once driver is identified, add to car dict.
        # TODO: Figure out way to output.

    ''' Output the results to a outputs dir with JSON file and annotated image. '''


if __name__ == "__main__":
    print(f'Running pipeline with test image: {settings.test_image}')
    run_pipeline()