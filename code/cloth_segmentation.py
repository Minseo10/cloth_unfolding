import numpy as np
import cv2
import torch
import albumentations as albu
import os
from PIL import Image
import sys
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

sys.path.append("/home/minseo/robot_ws/src")

from cloths_segmentation.cloths_segmentation.pre_trained_models import create_model


def segmentation(root_dir):
    model = create_model("Unet_2020-10-30")
    model.eval()
    input_image_path = root_dir + "observation_start/image_left.png"
    output_dir = root_dir + "detected_edge"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_path = output_dir + f"/segmentation.png"
    image = load_rgb(input_image_path)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    # imshow(mask)
    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i, j] == 1:
    #             rgb_image[i, j] = [255, 255, 0]
    #         else:
    #             rgb_image[i, j] = [128, 0, 128]

    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # rgb_image[mask == 1] = [255, 255, 0]
    # rgb_image[mask != 1] = [128, 0, 128]

    rgb_image = np.full((mask.shape[0], mask.shape[1], 3), [128, 0, 128], dtype=np.uint8)
    rgb_image[mask == 1] = [255, 255, 0]

    pil_image = Image.fromarray(rgb_image)
    pil_image.save(output_image_path)



if __name__ == '__main__':
    # model = create_model("Unet_2020-10-30")
    # model.eval()
    #
    # for i in range(377):
    # # for i in [1]:
    #     print(f"{i}/377")
    #     # root_dir = f"./cloth_competition_dataset_0000/sample_{'{0:06d}'.format(i)}/"
    #     root_dir = f"./cloth_competition_dataset_0001/sample_{'{0:06d}'.format(i)}/"
    #     input_img_path = root_dir + "observation_start/image_left.png"
    #     output_dir = root_dir + "detected_edge"
    #     output_img_path = output_dir + f"/segmentation_{'{0:06d}'.format(i)}.png"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     segmentation(model, input_img_path, output_img_path)
    root_path = "/home/minseo/cc_dataset/sample_000003/"
    segmentation(root_path)