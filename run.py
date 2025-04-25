import os
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image

# Load original images & cache embeddings
original_dir = "./Datasets/faces/"
original_embeddings = {}

print("Loading original image embeddings...")
for file in os.listdir(original_dir):
    if file.endswith(".jpg"):
        path = os.path.join(original_dir, file)
        try:
            embedding = DeepFace.represent(img_path=path, model_name='Facenet')[0]["embedding"]
            original_embeddings[file] = embedding
        except Exception as e:
            print(f"Could not process {file}: {e}")

def find_closest_original(sketch_path):
    print(f"\nMatching sketch: {sketch_path}")
    sketch_embedding = DeepFace.represent(img_path=sketch_path, model_name='Facenet')[0]["embedding"]

    closest_file = None
    closest_distance = float("inf")

    for orig_file, orig_embedding in original_embeddings.items():
        distance = np.linalg.norm(np.array(sketch_embedding) - np.array(orig_embedding))
        if distance < closest_distance:
            closest_distance = distance
            closest_file = orig_file

    print(f"✅ Closest original image: {closest_file} (Distance: {closest_distance:.4f})")
    sketch_img = Image.open(sketch_path)
    original_img = Image.open(os.path.join(original_dir, closest_file))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(sketch_img)
    axs[0].set_title("Input Sketch")
    axs[0].axis("off")

    axs[1].imshow(original_img)
    axs[1].set_title(f"Matched Original\n({closest_file})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

# === Example Usage ===
sketch_path = "./Datasets/sketches/m1-041-01-sz1.jpg"
find_closest_original(sketch_path)
