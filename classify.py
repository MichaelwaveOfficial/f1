from detection import run_rf_detr, get_inference_time, get_detection_info, get_no_of_detections


def classify_team(crop):

    '''
        Feeds crop to ROBOFLOW API for team classification. Returns a dictionary containing the car_id, crop, team, and confidence.
    '''

    car = {}

    # One team-classifier API call per crop.
    results = run_rf_detr(
        image_path=output_path,
        workspace_id=WORKSPACE_ID,
        model_id=TEAM_CLASSIFIER_ID,
        api_key=ROBOFLOW_API_KEY,
    )

    team_predictions = team_result.get("predictions", [])

    if team_predictions:

        # Pick highest confidence prediction for the team classification.
        best_prediction = max(
            team_predictions,
            key=lambda prediction: prediction["confidence"],
        )

        team = best_prediction["class"]
        team_confidence = best_prediction["confidence"]

    else:

        # Othwewise, if no predictions were made, set the team to "Unknown Team" and confidence to 0.0.
        best_prediction = None
        team = "Unknown Team"
        team_confidence = 0.0

    car = {
        "car_id": car_id,
        "crop": crop,
        "team": team,
        "confidence": f"{team_confidence:.1%}",
    }

    return car 


def classify_tcam_colour(tcam_crop, yellow_ratio_threshold=0.004):

    '''
        Fortunately, the T-cam is always either black or yellow, so we can use a simple HSV thresholding method 
        to determine the colour; In turn, helping us find our driver!
    '''

    if tcam_crop is None or tcam_crop.size == 0:
        result = "unknown"

    # BGR -> HSV conversion. Thresholding for bright yellow.
    hsv = cv2.cvtColor(tcam_crop, cv2.COLOR_BGR2HSV)

    # Fluorescent yellow range. The lower hue excludes most orange liveries. (Looking at you McLaren and Ferrari.)
    lower_yellow = np.array([20, 90, 100], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    # Remove isolated noise and connect nearby yellow pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Open & closing ops. To remove small noise and connect nearby yellow pixels.
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel,
    )
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    crop_area = tcam_crop.shape[0] * tcam_crop.shape[1]
    yellow_pixels = cv2.countNonZero(yellow_mask)
    yellow_ratio = yellow_pixels / crop_area

    # Require a coherent yellow region, not scattered livery pixels.
    number_of_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(yellow_mask)
    )

    # If there are multiple connected components, find the largest one. Otherwise, set the largest yellow region to 0.
    if number_of_labels > 1:
        largest_yellow_region = int(
            stats[1:, cv2.CC_STAT_AREA].max()
        )
    else:
        largest_yellow_region = 0

    minimum_region_area = max(8, round(crop_area * 0.001))

    # Binary sorting conditional.
    is_yellow = (
        yellow_ratio >= yellow_ratio_threshold
        and largest_yellow_region >= minimum_region_area
    )

    return "yellow" if is_yellow else "black"
