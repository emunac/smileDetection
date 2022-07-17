# smileDetection
## This is a project that detects smiling faces in a video

### Description
Input: An .mp4 video file.

Output: An .mp4 video file in the same resolution with
a green bounding box around each smiling face,
and a red bounding box around each not smiling face.

To gain this I trained a CNN classifier that recognizes
smiles and used a pretrained face detection model to
detect faces in each frame in the video.

The dataset I used for training is
https://github.com/hromi/SMILEsmileD
which contains 13,165 grayscale face-focused images of
64*64 pixels each. 9475 of those examples are not smiling faces, while only 3690 are smiling.

To extract faces in each frame I used the open source 
https://github.com/timesler/facenet-pytorch that provides pretrained face detection model.

TO DO: How to use this repo in your pc
