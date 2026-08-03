import os
import cv2
import numpy as np
from deepface import DeepFace

TRAINER_PATH = "Trainer/embeddings.npy"
IMAGE_PATH = "Input_Image/h.png"

MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"      
DISTANCE_THRESHOLD = 0.33 #changes can be made from 0.40 to 0.30 for accuracy.


def load_embeddings():

    if not os.path.exists(TRAINER_PATH):
        raise FileNotFoundError(
            "Training data not found! Run train.py first."
        )

    return np.load(TRAINER_PATH, allow_pickle=True).item()


def cosine_distance(vec1, vec2):

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return 1 - (
        np.dot(vec1, vec2)
        /
        (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )


def recognize_classroom(image_path):

    known_embeddings = load_embeddings()

    print("\nLoading image...")

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read image.")
        return []

    print("Detecting faces...")

    try:

        representations = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

    except Exception as e:

        print(e)
        return []

    present_students = set()

    for face in representations:

        if face.get("face_confidence", 0) < 0.40:
            continue

        query_embedding = face["embedding"]
        facial_area = face["facial_area"]

        best_match = None
        lowest_distance = float("inf")

        for student_id, saved_embeddings in known_embeddings.items():

            for saved_embedding in saved_embeddings:

                distance = cosine_distance(
                    query_embedding,
                    saved_embedding
                )

                if distance < lowest_distance:

                    lowest_distance = distance
                    best_match = student_id

        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]

        if lowest_distance <= DISTANCE_THRESHOLD:

            present_students.add(best_match)

            label = f"ID : {best_match}"

            color = (0, 255, 0)

            print(
                f"Recognized -> {best_match} "
                f"({lowest_distance:.3f})"
            )

        else:

            label = "Unknown"

            color = (0, 0, 255)

            print("Unknown Face")

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.rectangle(
            img,
            (x, y - 30),
            (x + w, y),
            color,
            cv2.FILLED
        )

        cv2.putText(
            img,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    print("\n--------------------------------")

    print("Recognized Students")

    print("--------------------------------")

    if len(present_students) == 0:

        print("No students recognized.")

    else:

        for student in sorted(present_students):
            print(student)

    print("--------------------------------")

    cv2.imshow("Smart Attendance System", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return list(present_students)


if __name__ == "__main__":

    if os.path.exists(IMAGE_PATH):

        recognized_students = recognize_classroom(
            IMAGE_PATH
        )

    else:

        print("Input image not found.")