from pytorchcv.model_provider import get_model
import cv2
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import warnings
from torchvision.transforms import Normalize
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, "./blazeface/")
sys.path.insert(0, "./deepfakes_inference_demo/")


if True:
    from helpers.face_extract_1 import FaceExtractor
    from helpers.read_video_1 import VideoReader
    from blazeface import BlazeFace


warnings.filterwarnings("ignore")
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = "./test_videos/"
train_dir = "./train_sample_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
train_videos = sorted([x for x in os.listdir(train_dir) if x[-4:] == ".mp4"])
frame_h = 5
frame_l = 5


facedet = BlazeFace().to(gpu)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
_ = facedet.train(False)

# frame_h * frame_l No of faces
frames_per_video = 64
video_reader = VideoReader()


def video_read_fn(x): return video_reader.read_frames(
    x, num_frames=frames_per_video)


face_extractor = FaceExtractor(video_read_fn, facedet)


input_size = 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


def isotropically_resize_image(img, size, resample: int = cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


class MyResNext(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNext, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)


checkpoint = torch.load(
    "./deepfakes_inference_demo/resnet.pth", map_location=gpu)

model_resnet = MyResNext().to(gpu)
model_resnet.load_state_dict(checkpoint)
_ = model_resnet.eval()

del checkpoint


# xception

class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


model_xception = get_model("xception", pretrained=False)
model_xception = nn.Sequential(*list(model_xception.children())[:-1])

model_xception = FCN(model_xception, 2048)
model_xception.load_state_dict(torch.load(
    './deepfake_xception_trained_model/model.pth', map_location=gpu))


def predict_on_video(video_path: str, batch_size: int, model, input_size) -> float:
    try:
        faces = face_extractor.process_video(video_path)
        face_extractor.keep_only_best_face(faces)

        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3),
                         dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" %
                              (n, batch_size))
            if n > 0:
                x = torch.tensor(x, device=gpu).float()
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                with torch.no_grad():
                    y_pred = model_resnet(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())

                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(
            test_dir, filename), frames_per_video, model_resnet, 224)
        y_pred1 = predict_on_video(os.path.join(
            test_dir, filename), frames_per_video, model_xception, 150)

        return (y_pred1 + y_pred) / 2

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)


if __name__ == "__main__":

    model.eval()
    predictions = predict_on_video_set(
        ["aagfhgtpmv.mp4", "aassnaulhq.mp4"], num_workers=4)

    print(predictions)
