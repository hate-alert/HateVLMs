"""-------------------------------------------------------1---------------------------------------------------------"""

# Vanilla as input
# output_file = idefics_facebook_dataset_vanilla_input.txt
# max_length=100
# prompt = [
#     image,
#     "User: Classify the above meme as hateful or not-hateful. Provide the answer as either hateful or "
#     "not-hateful only.\n"
#     "Example output for hateful meme : hateful\n"
#     "Example output for not-hateful meme : not-hateful\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------2---------------------------------------------------------"""

# Definition as input
# output_file = idefics_facebook_dataset_definition_input.txt
# max_length = 225
# prompt = [
#     image,
#     "User: Consider the following definitions.\n"
#     "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, "
#     "race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
#     "and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human "
#     "things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking "
#     "hate crime is also considered hateful.\n"
#     "2. not-hateful - an attack which is not hateful and follows social norms.\n"
#     "Classify the above meme as hateful or not-hateful based on the above definitions. "
#     "Provide the answer as either hateful or not-hateful only.\n"
#     "Example output for hateful meme : hateful\n"
#     "Example output for not-hateful meme : not-hateful\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------3---------------------------------------------------------"""

# OCR text as input
# output_file = idefics_facebook_dataset_ocr_input.txt
# max_length = 300
# prompt = [
#     image,
#     "User: Classify the above meme as hateful or not-hateful considering the image as well as "
#     "the extracted text from the image which is delimited by three backticks.\n"
#     f"""Extracted text from the image: ```{image_metadata['text']}```\n"""
#     "Provide your answer in the format: 'hateful' or 'not-hateful'.\n"
#     "Example output for hateful meme : 'hateful'.\n"
#     "Example output for not-hateful meme : 'not-hateful'.\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------4---------------------------------------------------------"""

# Definition + OCR text as input
# output_file = idefics_facebook_dataset_definition_ocr_input.txt
# max_length = 300
# prompt = [
#     image,
#     "User: Consider the following definitions.\n"
#     "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, "
#     "race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
#     "and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human "
#     "things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking "
#     "hate crime is also considered hateful.\n"
#     "2. not-hateful - an attack which is not hateful and follows social norms.\n"
#     "Classify the above meme as hateful or not-hateful based on the above definitions considering the image "
#     f"as well as the extracted text from the image delimited by three backticks```{image_metadata['text']}```.\n"
#     "Provide the answer as either hateful or not-hateful only.\n"
#     "Example output for hateful meme : hateful\n"
#     "Example output for not-hateful meme : not-hateful\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------5---------------------------------------------------------"""

# Explanation as output
# output_file = idefics_facebook_dataset_vanilla_input_explanation_output.txt
# max_length = 300
# prompt = [
#     image,
#     "User: Classify the above meme as hateful or not-hateful. Provide the answer as either hateful or "
#     "not-hateful only with an explanation within 30 words explaining your classification.\n"
#     "Example output for hateful meme : hateful - Explain within 30 words that why you classified this "
#     "meme as hateful.\n"
#     "Example output for not hateful meme : not-hateful - Explain within 30 words that why you classified "
#     "this meme as not-hateful.\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------6---------------------------------------------------------"""

# Definition as input && Explanation as output
# output_file = idefics_facebook_dataset_definition_input_explanation_output.txt
# max_length = 325
# prompt = [
#     image,
#     "User: Classify the above meme as hateful or not-hateful considering the below definitions.\n"
#     "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, "
#     "race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
#     "and disability or disease. Attack is defined as violent or dehumanizing speech, statements of inferiority, "
#     "and calls for exclusion or segregation. Mocking hate crime is also considered hateful.\n"
#     "2. not-hateful - an attack which is not hateful and follows social norms.\n"
#     "Provide your answer as either hateful or not-hateful only with an explanation within 30 words explaining "
#     "your classification.\n"
#     "Example output for hateful meme : hateful - Explain within 30 words that why you classified this "
#     "meme as hateful.\n"
#     "Example output for not-hateful meme : not-hateful - Explain within 30 words that why you classified "
#     "this meme as not-hateful.\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------7---------------------------------------------------------"""

# OCR text as input && Explanation as output
# output_file = idefics_facebook_dataset_ocr_input_explanation_output.txt
# max_length = 300
# prompt = [
#     image,
#     "User: Classify the above meme as hateful or not-hateful considering the image as well as "
#     f"""the extracted text from the image which is delimited by three backticks. ```{image_metadata['text']}```\n"""
#     "Provide the answer as either hateful or "
#     "not-hateful only with an explanation within 30 words explaining your classification.\n"
#     "Example output for hateful meme : hateful - Explain within 30 words that why you classified this "
#     "meme as hateful.\n"
#     "Example output for not hateful meme : not-hateful - Explain within 30 words that why you classified "
#     "this meme as not-hateful.\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]

"""-------------------------------------------------------8---------------------------------------------------------"""

# Definition + OCR text as input && Explanation as output
# output_file = idefics_facebook_dataset_definition_ocr_input_explanation_output.txt
# max_length = 400
# prompt = [
#     image,
#     "User: Consider the below definitions.\n"
#     "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, "
#     "race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and "
#     "disability or disease. Attack is defined as violent or dehumanizing speech, statements of inferiority, "
#     "and calls for exclusion or segregation. Mocking hate crime is also considered hateful.\n"
#     "2. not-hateful - an attack which is not hateful and follows social norms.\n"
#     "Classify the above meme as hateful or not-hateful based on the above definition considering the image "
#     f"as well as the extracted text from the image delimited by three backticks```{image_metadata['text']}```\n"
#     "Provide your answer as either hateful or not-hateful only with an explanation within 30 words explaining "
#     "your classification.\n"
#     "Example output for hateful meme : hateful - Explain within 30 words that why you classified this "
#     "meme as hateful.\n"
#     "Example output for not-hateful meme : not-hateful - Explain within 30 words that why you classified "
#     "this meme as not-hateful.\n"
#     "<end_of_utterance>",
#     "Assistant: ",
# ]
