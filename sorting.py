import cv2
import os
import shutil

# Define paths
dataset_path = "frames"
safe_folder = os.path.join(dataset_path, "safe")
risky_folder = os.path.join(dataset_path, "risky")


os.makedirs(safe_folder, exist_ok=True)
os.makedirs(risky_folder, exist_ok=True)


images = [f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not images:
    print("No images found in 'frames/'! Please add images and try again.")
    exit()

print("Press 'S' for Safe, 'R' for Risky, 'Q' to Quit")


for image in images:
    img_path = os.path.join(dataset_path, image)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping unreadable file: {image}")
        continue

    cv2.imshow("Classify Image", img)
    key = cv2.waitKey(0) & 0xFF  

    if key == ord('s'):
        shutil.move(img_path, os.path.join(safe_folder, image))
        print(f"Moved {image} to SAFE")
    elif key == ord('r'):
        shutil.move(img_path, os.path.join(risky_folder, image))
        print(f"Moved {image} to RISKY")
    elif key == ord('q'):
        print("Quitting classification...")
        break

cv2.destroyAllWindows()
print("Classification complete!")