from inference import IdeficsInference
from transformers import (
    IdeficsForVisionText2Text,
    AutoProcessor,
    BitsAndBytesConfig
)


class IdeficsCheckpointInitializer:

    def __init__(self, model_checkpoint):

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        # Initialize the model from model_checkpoint
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        # Initialize the processor from model_checkpoint
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)

        # Initialize inference class
        self.inference = IdeficsInference(self)

    def update_inference_class(self):
        self.inference = IdeficsInference(self)
