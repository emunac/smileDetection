import torch
import sys
from tkinter import Tk, filedialog
from facenet_pytorch import MTCNN
from pathlib import Path
from datetime import datetime
import cv2
from smile_net import Net
import consts
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from face_object import FaceObject, assign, correct
import subprocess


def detect_video(file_path):

    filename = Path(file_path).stem
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    new_filepath = f'{dir_path}{filename}_{current_time}_detected.mp4'

    v_cap = cv2.VideoCapture(filepath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smile_reco = Net()
    smile_reco.load_state_dict(
        torch.load(consts.SMILE_MODEL_PATH, map_location=device))
    smile_reco.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    face_objects_list = []

    for i in tqdm(range(v_len)):
        # Load frame
        success, frame = v_cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        frame_faces = []

        boxes, _ = mtcnn.detect(frame)
        if boxes is None:
            continue
        for box in boxes:
            face_image = frame.crop(box).resize(
                (consts.IMAGE_SIZE, consts.IMAGE_SIZE), Image.BILINEAR)
            face_image = transform(face_image)
            frame_faces.append(face_image)

        output = smile_reco(torch.stack(frame_faces))
        _, frame_predictions = torch.max(output.data, 1)
        face_objects_list = assign(face_objects_list, boxes, frame_predictions, i, v_len)

    correct(face_objects_list)

    v_cap = cv2.VideoCapture(filepath)
    frame_width = int(v_cap.get(3))
    frame_height = int(v_cap.get(4))
    out = cv2.VideoWriter(new_filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                          consts.FRAMES_PER_SECOND, (frame_width, frame_height))

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
                color = consts.GREEN if prediction else consts.RED
                cv2.rectangle(frame, pt1=pt_1, pt2=pt_2, color=color, thickness=4)

        out.write(frame)

    # When everything done, release the video capture and video write objects
    v_cap.release()
    out.release()

    # open detected video linux variants
    subprocess.call(('xdg-open', new_filepath))


if __name__ == '__main__':
    dir_path = '.' if len(sys.argv) < 2 else sys.argv[1]
    mtcnn = MTCNN(post_process=False)
    root = Tk()
    filepath = filedialog.askopenfilename(initialdir=dir_path, title="Select file",
                                          filetypes=(("video files", "*.mp4"), ("all files", "*.*")))
    detect_video(filepath)
