import os
import cv2
from tqdm import tqdm

def resize_rgb_images(input_root, output_root, size=(128, 128)):
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Creat folderul: {output_root}")

    files = [f for f in os.listdir(input_root) if f.endswith('.png')]
    print(f"Se procesează {len(files)} imagini din {input_root}...")
    for filename in tqdm(files):
        img = cv2.imread(os.path.join(input_root, filename))
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_root, filename), resized)


def resize_mask_images(input_root, output_root, size=(128, 128)):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Creat folderul: {output_root}")

    files = [f for f in os.listdir(input_root) if f.endswith('.png')]
    print(f"Se procesează {len(files)} imagini din {input_root}...")
    for filename in tqdm(files):
        img = cv2.imread(os.path.join(input_root, filename))
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_root, filename), resized)

if __name__ == "__main__":
    resize_rgb_images(
        "src/data/image-segmentation/original/CameraRGB/",
        "src/data/image-segmentation/resized/CameraRGB/" 
    )
    resize_mask_images(
        "src/data/image-segmentation/original/CameraMask/",
        "src/data/image-segmentation/resized/CameraMask/" 
    )