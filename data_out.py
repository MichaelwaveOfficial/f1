import json
from pathlib import Path
from datetime import datetime, timezone 

import cv2 


def save_frame(raw_frame, annotated_frame, output_dir):

    ''' Saves original & annotated frame '''

    now_utc = datetime.now(timezone.utc)

    frame_name = f'frame_{now_utc.timestamp()}'

    frames_dir_path = output_dir / "frames"
    frames_dir_path.mkdir(parents=True, exist_ok=True)

    raw_path = frames_dir_path / f'{frame_name}_raw.jpg'
    annotated_path = frames_dir_path / f'{frame_name}_annotated.jpg'

    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(annotated_path), annotated_frame)

    return {
        'raw_frame': raw_path.relative_to(output_dir).as_posix(),
        'annotated_frame': annotated_path.relative_to(output_dir).as_posix(),
    }


def create_output_dirs(output_dir, frame_name, car):  

    ''' Create the team, driver, and car record folder. '''

    team = car.get('team') or {}
    team_name = team.get('name', 'Unknown_Team')

    driver = car.get('driver') or {}
    driver_name = driver.get('name', 'Unknown_Driver')
    number = driver.get('number', 0)

    return (
        output_dir
        / 'teams'
        / team_name.replace(' ', '_')
        / driver_name.replace(' ', '_')
        / f'{frame_name}_car_{number}'
    )


def save_crops(car, car_dir):

    ''' Save the car crop and T-Cam crop. '''

    car_crop_path = car_dir / 'car_crop.jpg'
    cv2.imwrite(str(car_crop_path), car['car_crop'])

    tcam_crop_path = None

    if car.get('tcam_crop') is not None:
        tcam_crop_path = car_dir / 'tcam_crop.jpg'
        cv2.imwrite(str(tcam_crop_path), car['tcam_crop'])

    return {
        'car_crop': car_crop_path.name,
        'tcam_crop': tcam_crop_path.name if tcam_crop_path else None,
    }


def create_data_outputs(
    car,
    car_dir,
    frame_assets,
    crop_assets,
    frame_id,
    timestamp_s,
    car_number,
):
    
    ''' Save the car's processed data as JSON. '''

    data = {
        key: value
        for key, value in car.items()
        if key not in ('car_crop', 'tcam_crop')
    }

    data.update({
        'frame_id': frame_id,
        'timestamp_s': timestamp_s,
        'detection_index': car_number,
        'track_id': car.get('track_id'),
        'images': {
            **frame_assets,
            **crop_assets,
        },
    })

    json_path = car_dir / 'data.json'

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

    return json_path


def save_outputs(raw_frame, annotated_frame, cars, output_dir, frame_id=0, timestamp_s=None):

    ''' Save images and processed data for one frame. '''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_name = f'frame_{frame_id:06d}'
    frame_assets = save_frame(
        raw_frame,
        annotated_frame,
        output_dir,
    )

    saved_records = []

    for car_number, car in enumerate(cars, start=1):
        
        car_dir = create_output_dirs(
            output_dir,
            frame_name,
            car,
        )
        car_dir.mkdir(parents=True, exist_ok=True)

        crop_assets = save_crops(car, car_dir)

        json_path = create_data_outputs(
            car,
            car_dir,
            frame_assets,
            crop_assets,
            frame_id,
            timestamp_s,
            car_number,
        )
        saved_records.append(json_path)

    return saved_records