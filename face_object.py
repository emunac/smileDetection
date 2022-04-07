import numpy as np
import consts


def center_of_box(box):
    return np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])


def count_none_objects(general_list):
    return sum(x is None for x in general_list)


class FaceManager:
    def __init__(self, video_length):
        self.face_objects = []
        self.video_length = video_length

    def add_face_object(self, box, prediction, frame_number):

        face = FaceObject(box, prediction, frame_number, self.video_length)
        self.face_objects.append(face)

    def assign_faces_in_frame(self, boxes, predictions, frame_number):

        if len(boxes) != len(predictions):
            raise Warning("faces and predictions should have same length")

        for box, prediction in zip(boxes, predictions):
            center = center_of_box(box=box)

            min_distance, closest_face = consts.MAX_ASSIGN_DISTANCE, None
            for face in self.face_objects:
                distance = np.linalg.norm(face.current_center - center)
                if distance < min_distance:
                    min_distance = distance
                    closest_face = face

            if closest_face:
                closest_face.current_center = center
                closest_face.predictions_in_frames[frame_number] = prediction
                closest_face.coordinates_in_frames[frame_number] = box
            else:
                self.add_face_object(box, prediction, frame_number)

    def correct_smiling_predictions(self):
        for face in self.face_objects:
            corrected_predictions = face.predictions_in_frames
            half_window_size = int(consts.SLIDING_WINDOW_SIZE / 2)
            for i in range(half_window_size, self.video_length - half_window_size):
                window = face.predictions_in_frames[i - half_window_size: i + 1 + half_window_size]
                if count_none_objects(window) > 0:
                    continue

                majority = sum(window)
                corrected_predictions[i] = 0 if majority < half_window_size else 1

            face.predictions_in_frames = corrected_predictions

    def drop_mistaken_detected_faces(self):
        for face in self.face_objects:
            frame_appearance = self.video_length - count_none_objects(face.predictions_in_frames)
            if frame_appearance < consts.MIN_FRAME_APPEARANCE:
                self.face_objects.remove(face)


class FaceObject:
    def __init__(self, box, prediction, frame_number, video_length):
        self.current_center = center_of_box(box=box)
        self.predictions_in_frames = [None] * video_length
        self.predictions_in_frames[frame_number] = prediction
        self.coordinates_in_frames = [None] * video_length
        self.coordinates_in_frames[frame_number] = box
