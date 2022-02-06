import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
from torchvision import transforms
from tqdm import tqdm
from tkinter import Tk, filedialog
import subprocess
import platform
import pandas as pd
from smile_net import Net


class Face:
    # center is numpy array, to present the center point [x,y]
    def __init__(self, center, pred):
        self.current_center = center
        self.predictions = []
        self.predictions.append(pred)


def assign(faces, boxes, predictions):
    max_assign_distance = 50
    if len(boxes) != len(predictions):
        print("error, faces and predictions should have same length")
        return faces

    for box, prediction in zip(boxes, predictions):
        center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])

        min_distance = max_assign_distance
        closest_face = None
        for face in faces:
            distance = np.linalg.norm(face.current_center - center)
            if distance < min_distance:
                min_distance = distance
                closest_face = face
        if closest_face:
            closest_face.current_center = center
            closest_face.predictions.append(prediction)
        else:
            face = Face(center, prediction)
            faces.append(face)

    return faces

image_size = 64

mtcnn = MTCNN(image_size=image_size, margin=0, post_process=False)

root = Tk()
dir_path = '/home/emuna/PycharmProjects/SheCodes/examples/'
filepath = filedialog.askopenfilename(initialdir=dir_path, title="Select file",
                                      filetypes=(("video files", "*.mp4"), ("all files", "*.*")))
filename = os.path.basename(filepath)
new_filepath = dir_path + filename + '_detacted.mp4'

v_cap = cv2.VideoCapture(filepath)
frame_width = int(v_cap.get(3))
frame_height = int(v_cap.get(4))
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(new_filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                      24, (frame_width, frame_height))


SMILE_MODEL_PATH = "state_dict_model.pt"
smile_reco = Net()
smile_reco.load_state_dict(torch.load('state_dict_model.pt'))
smile_reco.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
faces = []
missed = []
# Loop through video
for i in tqdm(range(v_len)):
    # Load frame
    success, frame = v_cap.read()
    if not success:
        missed.append(i)
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    # face = mtcnn(frame)
    # face = transform(face)
    #
    # output = smile_reco(face[None, ...])
    # _, predicted = torch.max(output.data, 1)
    # predictions.append(predicted.item())

    frame_predictions = []
    boxes, probs = mtcnn.detect(frame, landmarks=False)
    draw = ImageDraw.Draw(frame)
    for box in boxes:
        face = frame.crop(box).resize((image_size, image_size), Image.BILINEAR)
        face = transform(face)
        output = smile_reco(face[None, ...])
        _, predicted = torch.max(output.data, 1)
        frame_predictions.append(predicted)

        color = 'green' if predicted else "red"
        draw.rectangle(box.tolist(), outline=color, width=6)

    faces = assign(faces, boxes, frame_predictions)
    out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

# When everything done, release the video capture and video write objects
v_cap.release()
out.release()


# test smiling accuracy
df = pd.read_csv('labels.csv')
if filename in df.columns:
    truth = df[[filename]].values.squeeze()
    truth = np.delete(truth, missed)
    accuracy = (truth == np.array(predictions)).sum()
    accuracy = 100 * accuracy / v_len
    print('accuracy = {}'.format(accuracy))

# open detected video
if platform.system() == 'Darwin':       # macOS
    subprocess.call(('open', new_filepath))
elif platform.system() == 'Windows':    # Windows
    os.startfile(new_filepath)
else:                                   # linux variants
    subprocess.call(('xdg-open', new_filepath))

