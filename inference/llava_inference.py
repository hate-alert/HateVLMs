from PIL import Image
import torch
from gpu import GPUInitializer

class LlavaInference:

    def __init__(self, checkpoint):
        self.gpu_initializer = GPUInitializer()

        self.checkpoint = checkpoint

        self.max_length = 800

    def generate_output(self, image_path, image_metadata, output_file):
        image = Image.open(image_path)

        prompt = [
            "USER: <image>\nConsider the following definitions.\n"
            "1. definition of 'misogynistic' meme: ```a meme is misogynous if it conceptually describes an offensive, "
            "sexist or hateful scene (weak or strong, implicitly or explicitly) having as target a woman or a group "
            "of women. Misogyny can be expressed in the form of shaming, stereotype, objectification and/or violence.```"
            "2. definition of 'not-misogynistic' meme: ```a meme that does not express any form "
            "of hate against women.```"
            "Classify the above meme as misogynistic or not-misogynistic based on the above definitions considering the image "
            f"as well as the extracted text from the image delimited by three backticks```{image_metadata['text']}```.\n"
            "Provide the answer as either misogynistic or not-misogynistic only.\n"
            "Example output for misogynistic meme : misogynistic\n"
            "Example output for not-misogynistic meme : not-misogynistic\n"
            "\nASSISTANT: First, I will thoroughly understand the provided definitions of misogynistic content. Then, I will examine the image and any text obtained through OCR to determine if any misogynistic elements are present. If 'Yes', I will classify the meme as misogynistic, otherwise, I will classify it as not misogynistic.So The meme is:"

        ]

        inputs = self.checkpoint.processor(text=prompt[0], images=image, return_tensors="pt").to(0, torch.float16)

        # Generate
        generated_ids = self.checkpoint.model.generate(**inputs, max_length=self.max_length)
        generated_text = self.checkpoint.processor.batch_decode(generated_ids, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)[0]

        print(generated_text)
        output_file.write(generated_text + "\n")
