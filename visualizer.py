
import numpy as np
import sys
import warnings
import torch
from backend import isotropically_resize_image, make_square_image, video_read_fn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
from torchvision import models, transforms
from torchvision.transforms import Normalize

sys.path.insert(0, "./blazeface/")
sys.path.insert(0, "./deepfakes_inference_demo/")

if True:
    from helpers.face_extract_1 import FaceExtractor
    from helpers.read_video_1 import VideoReader
    from blazeface import BlazeFace


batch_size = 64
input_size = 224
video_path = "./train_sample_videos/aknbdpmgua.mp4"

warnings.filterwarnings("ignore")
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_frames():
    try:
        facedet = BlazeFace().to(gpu)
        facedet.load_weights("./blazeface/blazeface.pth")
        facedet.load_anchors("./blazeface/anchors.npy")
        _ = facedet.train(False)
        face_extractor = FaceExtractor(video_read_fn, facedet)
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
            return x
        else:
            return []
    except Exception as e:
        print("Error face extraction video %s: %s" % (video_path, str(e)))


def get_model():
    from backend import MyResNext
    checkpoint = torch.load(
        "./deepfakes_inference_demo/resnet.pth", map_location=gpu)

    model_resnet = MyResNext().to(gpu)
    model_resnet.load_state_dict(checkpoint)
    _ = model_resnet.eval()

    del checkpoint
    return model_resnet


x = get_frames()
model = get_model()
print(x[0].shape)

model_weights = []
conv_layers = []
model_children = list(model.children())


counter = 0

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

print(f"Total convolutional layers: {counter}")

for weight, conv in zip(model_weights, conv_layers):
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

# In filter.png Convolutional Layer Filters
# plt.figure(figsize=(20, 17))
# for i, filter in enumerate(model_weights[0]):
#     plt.subplot(8, 8, i+1)
#     plt.imshow(filter[0, :, :].detach(), cmap='gray')
#     plt.axis('off')
#     plt.savefig('./filter.png')
# # plt.show()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

x = torch.tensor(x, device=gpu).float()
x = x.permute((0, 3, 1, 2))
y = []
for i in range(len(x)):
    x[i] = normalize_transform(x[i] / 255.)
    x[i] = transform(x[i])
    y.append(x[i].cpu().detach().numpy())


img = y[1]

img_show = img.reshape(224, 224, 3)
plt.imshow(img_show)
plt.savefig(f"./face_input.png")

img = isotropically_resize_image(img, input_size)
img = make_square_image(img)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

img = img.astype(np.uint8)
img = np.array(img)
img = transform(img)


img = img.unsqueeze(0)

print(conv_layers[0])

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results


for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64:
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"./layer_{num_layer}.png")
    # plt.show()
    plt.close()
