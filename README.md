# My Vision Project

A small Flask application for computer-vision experiments.

## Project structure

```text
my_vision_project/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── css/
        └── style.css
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. The `/health` endpoint returns the application status.

## Assistive features

The app now includes:

- Open-vocabulary prompts for sharp objects, fire, smoke, doors, people, stairs, curbs, potholes, and common obstacles.
- Spoken warnings with left, ahead, or right positional guidance in the browser.
- Mobile GPS location and a Google Maps panel.

Set `GOOGLE_MAPS_API_KEY` before starting the app to enable the embedded Google map. Restrict the key to the Maps JavaScript API in Google Cloud.

YOLO-World needs the `clip` Python module. If the model stream reports `ModuleNotFoundError: No module named 'clip'`, install Git and run `python -m pip install git+https://github.com/ultralytics/CLIP.git` inside the virtual environment, then restart the app.

This is assistive software and is not a replacement for a cane, guide, caregiver, or independent safety judgment. Object detection can miss hazards and cannot reliably identify a person's identity or guarantee that a route is safe.
