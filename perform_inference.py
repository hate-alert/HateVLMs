import os
import json
from data_set import DatasetWrapper
from tqdm import tqdm
from models import IdeficsCheckpointInitializer


class PerformInference:
    def __init__(self, dataset_tag, base_folder, split, model_checkpoint):
        self.base_folder = base_folder

        # Initialize dataset wrapper class
        self.dataset_wrapper = DatasetWrapper(
            dataset_tag=dataset_tag,
            base_folder=base_folder,
            split=split
        )

        # Initialize checkpoint initializer class
        self.checkpoint_initializer = IdeficsCheckpointInitializer(model_checkpoint)

    def generate_output(self, output_file_name):
        for iteration in range(1):
            output_file_path = os.path.join(
                '/kaggle/working/', output_file_name + str(iteration) + '.txt')

            output_file = open(output_file_path, 'w')
            for image in tqdm(self.dataset_wrapper.dataset):
                output_file.writelines(json.dumps(image))
                output_file.write('\n----------\n')
                self.checkpoint_initializer.inference.generate_output(os.path.join(self.base_folder, image['img']), image, output_file, self.base_folder)
                output_file.write("\n##########\n")
            output_file.close()
            
    def dynamically_update_inference_class(self):
        self.checkpoint_initializer.update_inference_class()



perform_inference = PerformInference(
    dataset_tag="Hinglish_Dataset",
    base_folder="/kaggle/input/hinglish/hinglish",
    split="hinglish",
    model_checkpoint="HuggingFaceM4/idefics-9b-instruct"
)

# perform_inference = PerformInference(
#     dataset_tag="facebook_hateful_meme_dataset",
#     base_folder="/kaggle/input/facebook-hateful-memes/hateful_memes",
#     split="test_seen",
#     model_checkpoint="HuggingFaceM4/idefics-9b-instruct"
# )

# perform_inference = PerformInference(
#     dataset_tag="MAMI_dataset",
#     base_folder="/kaggle/input/mami-dataset-2",
#     split="test",
#     model_checkpoint="HuggingFaceM4/idefics-9b-instruct"
# )

# perform_inference = PerformInference(
#     dataset_tag="Harm_P_Dataset",
#     base_folder="/kaggle/input/harm-p-memes-dataset",
#     split="test",
#     model_checkpoint="HuggingFaceM4/idefics-9b-instruct"
# )

# perform_inference = PerformInference(
#     dataset_tag="Harm_C_Dataset",
#     base_folder="/kaggle/input/harm-c-memes-dataset",
#     split="test",
#     model_checkpoint="HuggingFaceM4/idefics-9b-instruct"
# )