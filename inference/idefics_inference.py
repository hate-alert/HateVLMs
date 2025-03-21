from gpu import GPUInitializer
from PIL import Image

class IdeficsInference:

    def __init__(self, checkpoint):
        self.gpu_initializer = GPUInitializer()
        # self.gpu_initializer.forcibly_use_cpu()

        self.checkpoint = checkpoint

        # Hyper parameters
        self.max_length = 500

    def generate_output(self, image_path, image_metadata, file):
        image = Image.open(image_path)

        prompt = [
            image,
            "User: Consider the following definitions.\n"
            "1. definition of 'misogynistic' meme: a meme that disseminates or amplifies misogynistic and "
            "sexist attitudes, beliefs, or behaviors; use humor, satire, or imagery to demean, objectify, "
            "or insult women, often perpetuating stereotypes, promoting derogatory stereotypes, or making light of "
            "gender-based discrimination and abuse and contribute to a hostile and harmful online "
            "environment, reinforcing negative and offensive views about women, and are widely condemned for "
            "perpetuating gender-based prejudice and discrimination.\n"
            "2. definition of 'not-misogynistic' meme: a meme which is not misogynistic and follows social norms.\n"""
            "Classify the above meme as either 'misogynistic' or 'not-misogynistic' based on the above definitions "
            "considering the image as well as "
            "the extracted text from the image which is delimited by three backticks.\n"
            f"""Extracted text from the image: ```{image_metadata['text']}```\n"""
            "Provide your answer in the format: 'misogynistic' or 'not-misogynistic', followed by a brief explanation "
            "justifying your classification within 30 words.\n"
            "Example output for misogynistic meme : 'misogynistic' - Explain within 30 words that why you classified this "
            "meme as misogynistic.\n"
            "Example output for not-misogynistic meme : 'not-misogynistic' - Explain within 30 words that why you classified "
            "this meme as not-misogynistic.\n"
            "<end_of_utterance>",
            "Assistant: ",
        ]

        inputs = self.checkpoint.processor(prompt, return_tensors="pt").to(self.gpu_initializer.get_device())

        # Generation args
        exit_condition = self.checkpoint.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.checkpoint.processor.tokenizer(["<image>", "<fake_token_around_image>"],
                                                            add_special_tokens=False).input_ids

        generated_ids = self.checkpoint.model.generate(**inputs, eos_token_id=exit_condition,
                                                       bad_words_ids=bad_words_ids, max_length=self.max_length)
        generated_texts = self.checkpoint.processor.batch_decode(generated_ids, skip_special_tokens=True)

        for text in generated_texts:
            print(text)
            file.writelines(text)
