#this is a test for the pretrained face-recognetion model
import torch
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
from torchvision import transforms
from tqdm import tqdm
from smile_net import Net

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=64, margin=0, post_process=False)

dir_path = '/home/emuna/Downloads/'

v_cap = cv2.VideoCapture(dir_path + 'video.mp4')
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(v_len)

success, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)

face = mtcnn(frame, save_path=dir_path+'face_frame.jpeg')
plt.imshow(face.permute(1, 2, 0).int().numpy())
plt.show()

SMILE_MODEL_PATH = "state_dict_model.pt"
smile_reco = Net()
smile_reco.load_state_dict(torch.load('state_dict_model.pt'))
smile_reco.eval()

# Loop through video
batch_size = 16
frames = []
faces = []
predictions = []
transform = transforms.Grayscale()
for _ in tqdm(range(int(v_len))):

    # Load frame
    success, frame = v_cap.read()
    if not success:
        continue

    # Add to batch
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(frame))

    # When batch is full, detect faces and reset batch list
    if len(frames) >= batch_size:
        faces = (mtcnn(frames))
        faces = torch.stack(faces)
        faces = transform(faces)
        output = smile_reco(faces)
        _, predicted = torch.max(output.data, 1)
        predictions.extend(predicted)
        frames = []

print(predictions)

# frames = [Image.fromarray(frame).convert('L') for frame in video]
# print(len(frames))
# for frame in frames:
#     print(type(frame))
# transform = transforms.ToTensor()
# tensor_frames = transform(frames)
#
# SMILE_MODEL_PATH = "state_dict_model.pt"
# smile_reco = Net()
# smile_reco.load_state_dict(torch.load('state_dict_model.pt'))
# smile_reco.eval()
#
# for i, frame in enumerate(gray_frames):
#   print('\rTracking frame: {}'.format(i + 1), end='')
#   face = mtcnn(frame)
#   output = smile_reco(face[None, ...])
#   _, predicted = torch.max(output.data, 1)
#   print(predicted)
