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
from datetime import datetime
from pathlib import Path
import pandas as pd
from smile_net import Net


class FaceObject:
    # center is numpy array, to present the last center point [x,y]
    def __init__(self, box: list, pred: int, idx: int, v_len: int):
        self.current_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
        self.predictions_in_frames = [None] * v_len
        self.predictions_in_frames[idx] = pred
        self.coordinates_in_frames = [None] * v_len
        self.coordinates_in_frames[idx] = box


def assign(face_objects_list: list, boxes, predictions, idx, v_len) -> list:
    max_assign_distance = 50
    if len(boxes) != len(predictions):
        print("error, faces and predictions should have same length")
        return face_objects_list

    for box, prediction in zip(boxes, predictions):
        center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])

        min_distance = max_assign_distance
        closest_face = None
        for face in face_objects_list:
            distance = np.linalg.norm(face.current_center - center)
            if distance < min_distance:
                min_distance = distance
                closest_face = face

        if closest_face:
            closest_face.current_center = center
            closest_face.predictions_in_frames[idx] = prediction
            closest_face.coordinates_in_frames[idx] = box
        else:
            face = FaceObject(box, prediction, idx, v_len)
            face_objects_list.append(face)

    return face_objects_list


def correct(face_objects_list: list):
    min_frame_appearance = 10
    window_size = 10
    sensitivity = 0
    for face in face_objects_list:
        count_frame_appearance = sum(x is not None for x in face.predictions_in_frames)
        if count_frame_appearance < min_frame_appearance:
            face_objects_list.remove(face)
            continue

        for i, x in enumerate(face.predictions_in_frames):
            stri = str(x.item()) if x is not None else ''
            print('frame #', i, 'frame detect:', stri)
            # print(x.item(), i)

        corrected_predictions: list = face.predictions_in_frames
        half_window_size = int(window_size/2)
        for i in range(half_window_size, len(face.predictions_in_frames) - half_window_size):
            if sum(x is None for x in
                   face.predictions_in_frames[i - half_window_size: i + 1 + half_window_size]) > 0:
                continue

            majority = sum(face.predictions_in_frames[i - half_window_size: i + 1 + half_window_size])
            if majority < half_window_size - sensitivity:
                corrected_predictions[i] = 0
            else:
                corrected_predictions[i] = 1

        face.predictions_in_frames = corrected_predictions


image_size = 64
mtcnn = MTCNN(image_size=image_size, margin=0, post_process=False)

root = Tk()
dir_path = '/home/emuna/PycharmProjects/SheCodes/examples/'
filepath = filedialog.askopenfilename(initialdir=dir_path, title="Select file",
                                      filetypes=(("video files", "*.mp4"), ("all files", "*.*")))
filename = Path(filepath).stem
current_time = datetime.now().strftime("%d-%m-%Y %H:%M")
new_filepath = dir_path + filename + '_' + current_time + '_detected.mp4'

v_cap = cv2.VideoCapture(filepath)
frame_width = int(v_cap.get(3))
frame_height = int(v_cap.get(4))
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

SMILE_MODEL_PATH = "state_dict_model_sensitivity.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smile_reco = Net()
smile_reco.load_state_dict(torch.load(SMILE_MODEL_PATH, map_location=device))
smile_reco.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
face_objects_list = []
missed = []
# Loop through video for prediction
for i in tqdm(range(v_len)):
    # Load frame
    success, frame = v_cap.read()
    if not success:
        missed.append(i)
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    frame_faces = []

    boxes, _ = mtcnn.detect(frame, landmarks=False)
    if boxes is None:
        continue
    for box in boxes:
        face = frame.crop(box).resize((image_size, image_size), Image.BILINEAR)
        face = transform(face)
        frame_faces.append(face)

    output = smile_reco(torch.stack(frame_faces))
    _, frame_predictions = torch.max(output.data, 1)
    face_objects_list = assign(face_objects_list, boxes, frame_predictions, i, v_len)

correct(face_objects_list)

v_cap = cv2.VideoCapture(filepath)
out = cv2.VideoWriter(new_filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                      24, (frame_width, frame_height))

# Loop through video for drawing
for i in tqdm(range(v_len)):
    # Load frame
    success, frame = v_cap.read()
    if not success:
        continue

    for face in face_objects_list:
        face: FaceObject
        prediction = face.predictions_in_frames[i]
        coordinates = face.coordinates_in_frames[i]
        if prediction is not None:
            pt_1 = int(coordinates[0]), int(coordinates[1])
            pt_2 = int(coordinates[2]), int(coordinates[3])
            color = (0, 255, 0) if prediction else (0, 0, 255)
            cv2.rectangle(frame, pt1=pt_1, pt2=pt_2, color=color, thickness=4)

    out.write(frame)

# When everything done, release the video capture and video write objects
v_cap.release()
out.release()


# test smiling accuracy
# df = pd.read_csv('labels.csv')
# if filename in df.file_name.values:
#     labels_loc = df.loc[df.file_name == filename, 'face_1'].apply(eval).values[0]
#     truth = np.zeros(labels_loc[-1])
#     for i in range(0, len(labels_loc) - 1, 2):
#         truth[labels_loc[i]+1 : labels_loc[i+1]] = [1] * (labels_loc[i+1] - labels_loc[i] - 1)
#
#     face_object: FaceObject = face_objects_list[0]
#     accuracy = (truth == np.array(face_object.predictions_in_frames)).sum()
#     accuracy = 100 * accuracy / v_len
#     print('accuracy = {}'.format(accuracy))

# open detected video
if platform.system() == 'Darwin':       # macOS
    subprocess.call(('open', new_filepath))
elif platform.system() == 'Windows':    # Windows
    os.startfile(new_filepath)
else:                                   # linux variants
    subprocess.call(('xdg-open', new_filepath))

