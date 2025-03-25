from flask import Flask, request, jsonify
from flask_cors import cross_origin
from flask_cors import CORS
import cv2
import os
import time
import csv
import datetime
import numpy as np


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



# Base directory to store user images
BASE_DIR = "captured_faces"
attendance_file = "Attendance/Attendance.csv"
#########################################
## Attendance Records Endpoint   #
#########################################
@app.route("/", methods=["GET"])
@cross_origin()
def get_attendance():
    # Use an absolute path to ensure the file is found
    attendance_filename = os.path.join(os.getcwd(), attendance_file)
    print("Looking for attendance file at:", attendance_filename)
    records = []
    if os.path.isfile(attendance_filename):
        with open(attendance_filename, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # skip header
            for row in csv_reader:
                if len(row) >= 3:
                    records.append({
                        "id": row[0],
                        "name": row[1],
                        "time": row[2]
                    })
    else:
        print("Attendance file not found")

    return jsonify({"attendance": records})

#########################################
#           Registration Endpoint       #
#########################################
@app.route("/register", methods=["POST"])
@cross_origin()
def register():
    # Get username from form-data
    username = request.form.get("username", "").strip()
    if not username:
        return jsonify({"error": "Username is required."}), 400

    # Check if user is already registered (folder exists and has images)
    folder_path = os.path.join(BASE_DIR, username)
    if os.path.exists(folder_path) and os.listdir(folder_path):
        return jsonify({"message": f"User '{username}' already registered."}), 200

    # Create folder for new user
    os.makedirs(folder_path, exist_ok=True)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        return jsonify({"error": "Error loading cascade file."}), 500

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Error accessing the camera."}), 500

    count = 0
    max_images = 10      # Maximum number of images to capture
    delay = 0.5           # Delay (in seconds) between captures
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50)
        )

        # If at least one face is detected and enough time has passed, capture the face image
        if len(faces) > 0 and (time.time() - last_capture_time) >= delay:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            file_path = os.path.join(folder_path, f"{username}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1
            last_capture_time = time.time()

        if count >= max_images:
            break

    cap.release()

    return jsonify({
        "message": "Registration successful",
        "username": username,
        "images_captured": count,
        "folder": folder_path
    })

#########################################
#          Attendance Endpoint          #
#########################################
@app.route("/attendance", methods=["POST"])
@cross_origin()
def attendance():
    # Check that there are registered users
    if not os.path.exists(BASE_DIR):
        return jsonify({"error": "No registered users found."}), 400

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        return jsonify({"error": "Error loading cascade file."}), 500

    # Prepare training data from all registered users
    faces_data = []
    labels = []
    label_map = {}  # mapping from numeric label to username
    current_label = 0


    for user in os.listdir(BASE_DIR):
        user_folder = os.path.join(BASE_DIR, user)
        if not os.path.isdir(user_folder):
            continue
        image_files = [f for f in os.listdir(user_folder) if f.endswith(".jpg")]
        print(user_folder)
        if not image_files:
            continue
        label_map[current_label] = user
        
        for img_name in image_files:
            img_path = os.path.join(user_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # ✅ Resize all images to the same fixed size (e.g., 100x100)
            img = cv2.resize(img, (100, 100))

            faces_data.append(img)
            labels.append(current_label)
        current_label += 1

    if len(faces_data) == 0:
        return jsonify({"error": "No training data available."}), 400

    # ✅ Convert list to NumPy array after resizing images
    faces_data = np.array(faces_data, dtype="uint8")  # Ensures uniform type
    labels = np.array(labels, dtype="int32")  # Ensure labels are integer type

    # Create LBPH face recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        return jsonify({"error": "LBPHFaceRecognizer not available. Install opencv-contrib-python."}), 500

    recognizer.train(faces_data, labels)

    # Open the camera for live recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Error accessing the camera."}), 500

    recognized_user = None
    recognition_threshold = 70  # lower confidence means better match
    start_time = time.time()
    recognition_timeout = 15  # seconds to attempt recognition


    while time.time() - start_time < recognition_timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            try:
                label, confidence = recognizer.predict(face_roi)
            except Exception as e:
                continue  # if prediction fails, continue to next frame
            if confidence < recognition_threshold:
                recognized_user = label_map.get(label)
                break

        if recognized_user:
            break

    cap.release()

    if not recognized_user:
        return jsonify({"message": "No registered user recognized."}), 200
    
    print(recognized_user)


    # Mark attendance by appending a record in a CSV file named with today's date
    attendance_filename = os.path.join(os.getcwd(), attendance_file)
    file_exists = os.path.isfile(attendance_filename)
    next_id = 1
    if file_exists:
        with open(attendance_filename, "r") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                try:
                    next_id = int(rows[-1][0]) + 1
                except:
                    next_id = len(rows)
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Id", "Name", "Time"])
        writer.writerow([next_id, recognized_user, current_time_str])

    return jsonify({
        "message": f"Attendance marked for {recognized_user}",
        "attendance_csv": attendance_filename,
        "recognized_user": recognized_user
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(5000))
