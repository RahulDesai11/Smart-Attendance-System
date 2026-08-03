import os
import numpy as np
from deepface import DeepFace

DATASET_PATH = "Dataset"
TRAINER_PATH = "Trainer/embeddings.npy"

student_id = input("Enter Student ID to train: ").strip()

student_folder = os.path.join(DATASET_PATH, student_id)

if not os.path.exists(student_folder):
    print("[ERROR] Student folder not found.")
    exit()

# Load existing embeddings
if os.path.exists(TRAINER_PATH):
    embeddings_dict = np.load(
        TRAINER_PATH,
        allow_pickle=True
    ).item()
else:
    embeddings_dict = {}

# Remove old embeddings if they exist
embeddings_dict.pop(student_id, None)

student_embeddings = []

print(f"\nProcessing Student ID: {student_id}")

for filename in os.listdir(student_folder):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(student_folder, filename)

    try:

        representation = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            enforce_detection=True
        )

        embedding = representation[0]["embedding"]

        student_embeddings.append(embedding)

        print(f"[OK] {filename}")

    except Exception as e:

        print(f"[FAILED] {filename}")
        print(e)

if len(student_embeddings) == 0:

    print("\nNo valid faces found.")

else:

    embeddings_dict[student_id] = student_embeddings

    os.makedirs("Trainer", exist_ok=True)

    np.save(
        TRAINER_PATH,
        embeddings_dict,
        allow_pickle=True
    )

    print("\nTraining Complete!")

    print(f"Stored {len(student_embeddings)} embeddings for Student {student_id}")