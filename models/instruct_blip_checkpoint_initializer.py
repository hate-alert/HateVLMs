from accelerate import (
    init_empty_weights,
    infer_auto_device_map
)
from inference import InstructBlipInference
from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipConfig,
    AutoModelForVision2Seq,
    InstructBlipProcessor,
    BitsAndBytesConfig
)
import psutil

end_separation = "\n\n\n\n"


class InstructBlipCheckpointInitializer:

    def __init__(self, checkpoint):
        # Initialize model with empty weights context
        config = InstructBlipConfig.from_pretrained(checkpoint)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        with init_empty_weights():
            self.model = AutoModelForVision2Seq.from_config(config)
        self.model.tie_weights()

        max_memory = {0: "10GIB", 1: "10GIB", "cpu": psutil.virtual_memory().available}

        # Configuring device map
        device_map = infer_auto_device_map(
            self.model,
            no_split_module_classes=["InstructBlipEncoderLayer", "InstructBlipQFormerLayer", "LlamaDecoderLayer"],
            max_memory=max_memory
        )
        # print(device_map, end=end_separation)
        device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model'
                                                                                               '.embed_tokens')]

        # Configuring the model according to the above defined device map
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True
        )
        # print(self.model.hf_device_map, end=end_separation)

        # Processor
        self.processor = InstructBlipProcessor.from_pretrained(checkpoint)

        # Initialize inference class
        self.inference = InstructBlipInference(self)

        print(self.model.hf_device_map, end=end_separation)

        # for parameter in self.model.named_parameters():
        #     print(f"{parameter[0]} -> {parameter[1].device}")

    def update_inference_class(self):
        self.inference = InstructBlipInference(self)
