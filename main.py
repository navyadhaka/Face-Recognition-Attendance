import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import glob
import os

def l2_normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# Load the InsightFace model
app = FaceAnalysis(name="buffalo_l")  # Uses 'ArcFace' model
app.prepare(ctx_id=0, det_size=(640, 640))

# Paths to images
group_photo_path = "Resources/Group/g6.png"  # Change this to your group image
data_folder = "data/new_2"  # Folder containing individual face images

# Load reference images and extract embeddings
reference_faces = {}
for folder_path in glob.glob(os.path.join(data_folder, "*")):
    for img_path in glob.glob(os.path.join(folder_path, "*.jpg")):
        img = cv2.imread(img_path)
        faces = app.get(img)
        if faces:
            reference_faces[folder_path] = {img_path : faces[0].embedding}  # Store first face embedding

# Load and process the group photo
group_img = cv2.imread(group_photo_path)
group_faces = app.get(group_img)

# Draw bounding boxes and match detected faces
for face in group_faces:
    bbox = face.bbox.astype(int)  # Convert bbox coordinates to integers
    x1, y1, x2, y2 = bbox

    best_match = "Unknown"
    best_score = 1.2

    # Compare with stored faces
    for fol, img in reference_faces.items():
        for ref_path, ref_embedding in img.items():
            ref_embedding = l2_normalize(ref_embedding)
            face_embedding = l2_normalize(face.embedding)
            dist = np.linalg.norm(face_embedding - ref_embedding)  # Compute Euclidean distance
            if dist < best_score:  # Lower distance means better match
                best_score = dist
                best_match = os.path.basename(fol)  # Use filename as the label
    print(f"Best match: {best_match} (Score: {best_score:.2f}) ")


    # Draw bounding box and label on the group photo
    color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
    cv2.rectangle(group_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(group_img, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



cv2.imwrite("matched_faces.png", group_img)
print("Processed image saved as 'matched_faces.png'")
