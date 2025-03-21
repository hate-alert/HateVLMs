from PIL import Image


class InstructBlipInference:

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

        # Declare the hyperparameters up here itself
        self.do_sample = True
        self.num_beams = 2
        self.max_length = 100
        self.early_stopping = False
        self.repetition_penalty = 1.5
        self.length_penalty = 1.0
        self.temperature = 0.8
        self.top_k = 5
        self.top_p = 0.8

    def generate_output(
            self,
            image_path,
            image_metadata,
            output_file,
            rices_class_few_shot,
            shots
    ):
        image = Image.open(image_path)
        prompt = [
            "Consider the following definitions.\n"
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
        ]
        inputs = self.checkpoint.processor(images=image, text=prompt[0], return_tensors="pt").to("cuda:0")
        print(prompt[0])

        # Generate the output
        outputs = self.checkpoint.model.generate(
            **inputs,
            # do_sample=self.do_sample,
            # num_beams=self.num_beams,
            max_length=self.max_length,
            # early_stopping=self.early_stopping,
            # repetition_penalty=self.repetition_penalty,
            # length_penalty=self.length_penalty,
            # temperature=self.temperature,
            # top_k=self.top_k,
            # top_p=self.top_p
        )

        generated_text = self.checkpoint.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
        output_file.write(prompt[0])
        output_file.write(generated_text)
