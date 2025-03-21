from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from inference import LlavaInference


class LlavaCheckpointInitializer:

    def __init__(self, model_checkpoint):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config
        )
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        self.inference = LlavaInference(self)

    def update_inference_class(self):
        self.inference = LlavaInference(self)
