import os.path

import imageio
import numpy as np

from skimage.color import rgb2gray
from skimage.segmentation import slic
from skimage.measure import regionprops
import copy

from tqdm import tqdm
import json

import matplotlib.pyplot as plt

from data_set import DatasetWrapper

n_segments_threshold = 10
compactness_threshold = 75


def draw_superpixels(dataset_wrapper: DatasetWrapper):

    final_json = []

    for relative_image_path in tqdm(dataset_wrapper.get_dataset()):

        relative_image_path["slic"] = []
        original_image = imageio.imread(
            os.path.join(
                dataset_wrapper.get_base_folders()[0],
                relative_image_path['img']
            )
        )

        try:
            segments_slic = slic(
                original_image,
                n_segments=n_segments_threshold,
                compactness=compactness_threshold,
                sigma=1
            )

            regions = regionprops(
                segments_slic,
                intensity_image=rgb2gray(original_image)
            )

            index = -1

            while index < len(np.unique(segments_slic)):

                img_temp = copy.deepcopy(original_image)

                if index >= 0:
                    props = regions[index]
                    for coordinates in props.coords:
                        img_temp[coordinates[0]][coordinates[1]][0] = 255
                        img_temp[coordinates[0]][coordinates[1]][1] = 255
                        img_temp[coordinates[0]][coordinates[1]][2] = 255

                number = "_" + str(index)
                if index == -1:
                    number = ''

                plt.imsave(
                    os.path.join(
                        dataset_wrapper.get_base_folders()[0],
                        "img_slic",
                        relative_image_path['img'].split("/")[-1].split(".")[0] + number + ".png"
                    ),
                    img_temp
                )

                relative_image_path["slic"].append(
                    os.path.join(
                        "img_slic",
                        relative_image_path['img'].split("/")[-1].split(".")[0] + number + ".png"
                    )
                )

                index += 1

        except ValueError as ve:
            print(relative_image_path)

        final_json.append(relative_image_path)

    with open(os.path.join(dataset_wrapper.get_base_folders()[0], "dev_slic.jsonl"), 'w') as f:
        for item in final_json:
            f.write(json.dumps(item) + "\n")


draw_superpixels(
    dataset_wrapper=DatasetWrapper(
        dataset_tag="facebook_hateful_meme_dataset",
        base_folders=["/Users/naquee.rizwan/Desktop/hatevlms/Datasets/KaggleDataset"],
        split="dev"
    )
)
