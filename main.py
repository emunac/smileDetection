import torch
import sys
from tkinter import Tk, filedialog
from facenet_pytorch import MTCNN
from pathlib import Path
from datetime import datetime
import cv2
from smile_net import Net as SmileRecognizerNet
import consts
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from face_object import FaceManager, FaceObject
import subprocess


def create_new_filepath(file_path):
    filename = Path(file_path).stem
    dir_path = Path(file_path).parent
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    new_filepath = f'{dir_path}/{filename}_{current_time}_detected.mp4'
    return new_filepath


def save_faces_and_smiling_predictions(file_path):
    video_capture = cv2.VideoCapture(file_path)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smile_recognizer_net = SmileRecognizerNet()
    smile_recognizer_net.load_state_dict(
        torch.load(consts.SMILE_MODEL_PATH, map_location=device))
    smile_recognizer_net.eval()

    image_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Grayscale()])

    face_manager = FaceManager(video_length)
    for i in tqdm(range(video_length)):
        # Load frame
        success, frame = video_capture.read()
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
            face_image = image_transforms(face_image)
            frame_faces.append(face_image)

        output = smile_recognizer_net(torch.stack(frame_faces))
        _, frame_predictions = torch.max(output.data, 1)
        face_manager.assign_faces_in_frame(boxes, frame_predictions, i)

    video_capture.release()
    face_manager.drop_mistaken_detected_faces()
    face_manager.correct_smiling_predictions()
    return face_manager


def draw_on_video(file_path, face_manager):

    video_capture = cv2.VideoCapture(filepath)
    new_filepath = create_new_filepath(file_path)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter(new_filepath, cv2.VideoWriter_fourcc(*'mp4v'),
                          consts.FRAMES_PER_SECOND, (frame_width, frame_height))

    for i in tqdm(range(video_length)):
        # Load frame
        success, frame = video_capture.read()
        if not success:
            continue

        for face in face_manager.face_objects:
            face: FaceObject
            prediction = face.predictions_in_frames[i]
            coordinates = face.coordinates_in_frames[i]
            if prediction is not None:
                pt_1 = int(coordinates[0]), int(coordinates[1])
                pt_2 = int(coordinates[2]), int(coordinates[3])
                color = consts.GREEN if prediction else consts.RED
                cv2.rectangle(frame, pt1=pt_1, pt2=pt_2, color=color, thickness=4)

        out.write(frame)

    video_capture.release()
    out.release()
    return new_filepath


if __name__ == '__main__':
    dir_path = '.' if len(sys.argv) < 2 else sys.argv[1]
    mtcnn = MTCNN(post_process=False)
    root = Tk()
    filepath = filedialog.askopenfilename(initialdir=dir_path, title="Select file",
                                          filetypes=(("video files", "*.mp4"), ("all files", "*.*")))
    face_manager = save_faces_and_smiling_predictions(filepath)
    new_file_path = draw_on_video(filepath, face_manager)

    # open detected video linux variants
    subprocess.call(('xdg-open', new_file_path))
