from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN
import os
import datetime
import pickle
import face_recognition
import json
import time

app = Flask(__name__)

# For formatting timestamps
@app.template_filter('datetimeformat')
def datetimeformat(value):
    from datetime import datetime
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')

# Serve profile images
@app.route('/dataset/<path:filename>')
def serve_dataset_file(filename):
    return send_from_directory('dataset', filename)

# Load the mask detection model
model = tf.keras.models.load_model("RESNET.h5")
IMG_SIZE = 224
detector = MTCNN()
camera = cv2.VideoCapture(1)

# Ensure directories exist
os.makedirs("static/snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Load known face encodings
try:
    with open("face_encodings.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
except FileNotFoundError:
    known_face_encodings, known_face_names = [], []

registration_mode = False
current_person = ""
person_meta = {}
count = 0
max_images = 20
last_snapshot_time = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/snapshots')
def snapshots():
    user_cards = []
    snapshot_root = "static/snapshots"

    for person_name in os.listdir(snapshot_root):
        person_snap_dir = os.path.join(snapshot_root, person_name)
        if not os.path.isdir(person_snap_dir):
            continue

        profile_path = os.path.join("dataset", person_name, "profile.json")
        profile_img = os.path.join("dataset", person_name, "profile.jpg")

        if not (os.path.exists(profile_path) and os.path.exists(profile_img)):
            continue

        with open(profile_path, "r") as f:
            meta = json.load(f)

        snapshots = []
        for file in os.listdir(person_snap_dir):
            if file.endswith(".jpg"):
                snapshots.append({
                    "path": f"/{person_snap_dir}/{file}".replace("\\", "/"),
                    "timestamp": os.path.getmtime(os.path.join(person_snap_dir, file))
                })

        user_cards.append({
            "name": meta.get("name"),
            "department": meta.get("department"),
            "job": meta.get("job"),
            "profile_img": f"/dataset/{person_name}/profile.jpg",
            "snapshots": sorted(snapshots, key=lambda x: x['timestamp'], reverse=True)
        })

    return render_template("snapshots.html", users=user_cards)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global last_snapshot_time
    while True:
        success, frame = camera.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)
        for result in detections:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + w, y + h
            face = frame[y:y2, x:x2]
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue
            label, confidence = predict_mask(face)
            name = "Unknown"
            try:
                face_encoding = face_recognition.face_encodings(rgb, [(y, x2, y2, x)])[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
            except:
                pass
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            label_text = f"{name}: {'Mask' if label == 0 else 'No Mask'} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            current_time = time.time()
            if name != "Unknown" and label == 1 and confidence >= 99 and (current_time - last_snapshot_time) >= 10:
                person_folder = os.path.join("static/snapshots", name)
                os.makedirs(person_folder, exist_ok=True)
                snapshot_path = os.path.join(person_folder, f"{int(current_time)}.jpg")
                cv2.imwrite(snapshot_path, frame)
                last_snapshot_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def predict_mask(face):
    try:
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face, verbose=0)
        class_id = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        return class_id, confidence
    except:
        return -1, 0

@app.route('/start_register', methods=['POST'])
def start_register():
    global registration_mode, current_person, count, person_meta
    name = request.form.get('name')
    department = request.form.get('department')
    job = request.form.get('job')
    gender = request.form.get('gender')
    phone = request.form.get('phone')
    profile_img = request.files.get('profileImage')

    if not name:
        return "Missing name", 400

    current_person = name
    count = 0
    registration_mode = True
    person_meta = {"name": name, "department": department, "job": job, "gender": gender, "phone": phone}

    person_dir = os.path.join("dataset", name)
    os.makedirs(person_dir, exist_ok=True)

    with open(os.path.join(person_dir, "profile.json"), "w") as f:
        json.dump(person_meta, f)

    if profile_img:
        profile_img.save(os.path.join(person_dir, "profile.jpg"))

    return "Registration started"

@app.route('/capture_faces')
def capture_faces():
    return Response(generate_capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_capture_frames():
    global count, current_person, registration_mode
    cap = cv2.VideoCapture(0)
    while registration_mode and count < max_images:
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)
        for result in detections:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + w, y + h
            face = frame[y:y2, x:x2]
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue
            face = cv2.resize(face, (224, 224))
            person_dir = os.path.join("dataset", current_person)
            file_path = os.path.join(person_dir, f"{current_person}_{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    registration_mode = False

    person_dir = os.path.join("dataset", current_person)
    for file in os.listdir(person_dir):
        if file.endswith(".jpg") and not file.startswith("profile"):
            path = os.path.join(person_dir, file)
            img = cv2.imread(path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='hog')
            encs = face_recognition.face_encodings(rgb, boxes)
            for enc in encs:
                known_face_encodings.append(enc)
                known_face_names.append(current_person)
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
@app.route('/delete_snapshot', methods=['POST'])
def delete_snapshot():
    data = request.get_json()
    path = data.get("path")

    if path and path.startswith("/static/snapshots/"):
        abs_path = os.path.join(os.getcwd(), path.lstrip("/"))  # convert to full disk path
        try:
            os.remove(abs_path)
            print(f"[INFO] Deleted file: {abs_path}")
            return jsonify({"status": "deleted"})
        except Exception as e:
            print(f"[ERROR] Failed to delete {abs_path}: {e}")
            return jsonify({"status": "error", "reason": str(e)}), 500
    return jsonify({"status": "invalid"}), 400

if __name__ == '__main__':
    app.run(debug=True)