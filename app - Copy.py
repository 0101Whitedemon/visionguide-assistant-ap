from flask import Flask, Response, jsonify, render_template, request

import cv2
from threading import Lock

app = Flask(__name__)
camera = cv2.VideoCapture(0)
model = None
detection_state = {"hazards": [], "message": "Monitoring active"}
detection_lock = Lock()
settings = {"confidence": 0.45, "profile": "all"}

DETECTION_CLASSES = [
    "knife",
    "scissors",
    "gun",
    "fire",
    "smoke",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "dog",
]
HAZARD_CLASSES = set(DETECTION_CLASSES)
PROFILES = {
    "all": HAZARD_CLASSES,
    "sharp": {"knife", "scissors", "gun", "fire", "smoke"},
    "traffic": {"car", "motorcycle", "bus", "truck", "bicycle"},
}
URGENT_HAZARDS = {"fire", "smoke", "gun"}


def get_model():
    global model
    if model is None:
        from ultralytics import YOLO

        model = YOLO("yolov8s-worldv2.pt")
        model.set_classes(DETECTION_CLASSES)
    return model


def generate_frames():
    global detection_state
    detector = get_model()

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = detector(frame, verbose=False)[0]
        detected_hazards = []
        active_settings = settings.copy()
        active_classes = PROFILES.get(active_settings["profile"], HAZARD_CLASSES)

        for box in results.boxes:
            confidence = float(box.conf[0])
            class_name = results.names[int(box.cls[0])]
            required_confidence = (
                min(active_settings["confidence"], 0.30)
                if class_name in URGENT_HAZARDS
                else active_settings["confidence"]
            )
            if confidence < required_confidence or class_name not in active_classes:
                continue

            detected_hazards.append(
                {"name": class_name, "confidence": round(confidence, 2)}
            )
            left, top, right, bottom = map(int, box.xyxy[0])
            label = f"WARNING: {class_name} {confidence:.0%}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(
                frame,
                label,
                (left, max(top - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        urgent_hazards = [
            item for item in detected_hazards if item["name"] in URGENT_HAZARDS
        ]
        status = (
            f"URGENT WARNING: {', '.join(item['name'] for item in urgent_hazards)}"
            if urgent_hazards
            else f"WARNING: {len(detected_hazards)} possible hazard(s)"
            if detected_hazards
            else "No configured hazards detected"
        )
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 180) if urgent_hazards else (0, 0, 255) if detected_hazards else (0, 255, 204),
            2,
        )

        with detection_lock:
            detection_state = {
                "hazards": detected_hazards,
                "message": (
                    "Urgent warning: " + ", ".join(item["name"] for item in urgent_hazards)
                    if urgent_hazards
                    else "Warning: " + ", ".join(item["name"] for item in detected_hazards)
                    if detected_hazards
                    else "Path appears clear"
                ),
                "urgent": bool(urgent_hazards),
            }

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return jsonify(
        {
            "camera": camera.isOpened(),
            "detector": model is not None,
            "mode": "Hazard watch",
            "settings": settings,
        }
    )


@app.get("/detections")
def detections():
    with detection_lock:
        return jsonify(detection_state)


@app.route("/settings", methods=["GET", "POST"])
def update_settings():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        if "confidence" in payload:
            settings["confidence"] = min(0.8, max(0.25, float(payload["confidence"])))
        if payload.get("profile") in PROFILES:
            settings["profile"] = payload["profile"]
    return jsonify(settings)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
