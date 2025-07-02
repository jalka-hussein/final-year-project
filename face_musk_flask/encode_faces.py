##for face reconization section
import face_recognition
import os
import cv2
import pickle

dataset_path = "dataset"
encodings = []
names = []

print("[INFO] Encoding faces...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb, model='hog')

        # Compute encodings
        encs = face_recognition.face_encodings(rgb, boxes)

        for enc in encs:
            encodings.append(enc)
            names.append(person_name)

# Save to pickle file
data = {"encodings": encodings, "names": names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((encodings, names), f)

print(f"[INFO] Done. {len(encodings)} faces encoded.")
