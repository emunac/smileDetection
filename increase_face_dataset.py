import os
from PIL import Image
from torchvision import transforms

img_dir = "../SMILEsmileD/SMILEs"
positive_path = os.path.join(img_dir, "positives/positives7")
negative_path = os.path.join(img_dir, "negatives/negatives7")

for path in [positive_path, negative_path]:
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image_path = os.path.join(path, file)
            image = Image.open(image_path)
            flipped_image = transforms.RandomHorizontalFlip(p=1)(image)
            flipped_file_name = "F" + file
            flipped_image.save(os.path.join(path, flipped_file_name))