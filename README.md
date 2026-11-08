# Hand Gesture (Mudra) Recognition Project

This repository contains a hand gesture recognition workflow with:

- model training data folders
- a trained Keras model file
- detector scripts for live and upload-based prediction
- a classical Flask web interface in a separate folder

## Project Structure

- livedetector.py: Live camera gesture detection script
- uploaddetector.py: Gesture detection from uploaded media
- mudra_model.keras: Trained gesture classification model
- train_model.ipynb: Notebook used for model training/experiments
- train/: Training dataset organized by gesture class
- val/: Validation dataset organized by gesture class
- classical/: Flask-based web application
  - app.py: Main Flask app entry
  - templates/: HTML templates
  - static/css/style.css: App styling

## Gesture Classes

Current dataset/model classes:

- Alapadmam
- Chandrakala
- Kapitham
- Shikaram

## Quick Start

## 1) Python environment

Create and activate a virtual environment, then install required packages.

Typical packages for this project may include:

- tensorflow
- opencv-python
- numpy
- flask

If you already have an existing environment in .venv, activate it and install any missing dependencies.

## 2) Run live detector

Run livedetector.py to predict gestures from webcam input.

## 3) Run upload detector

Run uploaddetector.py to predict gestures from uploaded images/videos (based on script behavior).

## 4) Run web app

Inside classical/, start app.py to launch the Flask interface.

## Notes

- Keep train/ and val/ class folder names consistent with model output labels.
- If you retrain the model, replace mudra_model.keras and ensure detector scripts use the updated label mapping.
- The classical/ app can be customized independently from core detector/training scripts.

## Suggested Next Improvements

- Add a requirements.txt for reproducible setup
- Add model versioning notes
- Add sample input/output screenshots to this README
