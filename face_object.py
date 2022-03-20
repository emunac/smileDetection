import numpy as np
import consts


def center_of_box(box):
    return np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])


class FaceObject:
    def __init__(self, box, pred, idx, v_len):
        self.current_center = center_of_box(box=box)
        self.predictions_in_frames = [None] * v_len
        self.predictions_in_frames[idx] = pred
        self.coordinates_in_frames = [None] * v_len
        self.coordinates_in_frames[idx] = box


def assign(face_objects_list, boxes, predictions, idx, v_len):
    if len(boxes) != len(predictions):
        print("error, faces and predictions should have same length")
        return face_objects_list

    for box, prediction in zip(boxes, predictions):
        center = center_of_box(box=box)

        min_distance, closest_face = consts.MAX_ASSIGN_DISTANCE, None
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
    for face in face_objects_list:
        count_frame_appearance = sum(x is not None for x in face.predictions_in_frames)
        if count_frame_appearance < consts.MIN_FRAME_APPEARANCE:
            face_objects_list.remove(face)
            continue

        corrected_predictions: list = face.predictions_in_frames
        half_window_size = int(consts.SLIDING_WINDOW_SIZE / 2)
        for i in range(half_window_size, len(face.predictions_in_frames) - half_window_size):
            if sum(x is None for x in
                   face.predictions_in_frames[i - half_window_size: i + 1 + half_window_size]) > 0:
                continue

            majority = sum(face.predictions_in_frames[i - half_window_size: i + 1 + half_window_size])
            corrected_predictions[i] = 0 if majority < half_window_size else 1

        face.predictions_in_frames = corrected_predictions
