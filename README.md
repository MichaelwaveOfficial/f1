F1 Computer Vision MVP

> **Work in progress** — an experimental computer vision project exploring how Formula 1 cars in race footage can be detected and identified.

This repo is a place to share the build as it develops. The goal is to turn race images into an annotated view of the cars, with computer vision doing the detection and identification rather than claiming perfect results.

This is currently hashed into a notebook to visualise each step. Trying to work on developing an actual full in one pipeline or perhaps a more fluid example that can
run on video.

## MVP idea

The current prototype direction is a small, staged vision pipeline:

1. Detect cars in an image using a Roboflow-hosted transformer for object detection.
2. Crop each detected car from the frame using the detections bounding boxes.
3. Classify the team from the crop. Again, using a Roboflow-hosted transfomer (this time for classification).
4. Explore additional visual identification, such as T-cam colour by cropping the top half of the car bbox with extra padding outside of the bbox.
5. Leveraging team -> t-cam colour -> perform driver lookup.
6. Draw the results onto the image with OpenCV.

The first milestone is image-based processing. Video tracking, persistent car identities across frames, and race-session statistics are later ideas, not features this MVP claims to provide.

## Status

This is an early MVP and parts of the pipeline are still being built and validated. Detection and classification can be imperfect, especially with small, overlapping, blurred, or partially obscured cars. Results depend on the footage and the model.

A short demo, sample output, and setup instructions will be added here as they are ready. No API keys or private credentials should be committed to the repository.

## Tech

- Python
- Roboflow models for car detection and image classification
- OpenCV for image processing and annotations

## Disclaimer

This is an independent, non-commercial fan project. It is not affiliated with, endorsed by, or associated with Formula 1, the FIA, any team, or any driver. F1 and related names and marks belong to their respective owners.
