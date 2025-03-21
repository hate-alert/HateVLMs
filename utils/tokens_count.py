import os
import spacy

nlp = spacy.load("en_core_web_md")
total_input_tokens = 0
total_output_tokens = 0


def extract_information_from_folder(folder_path):
    print("\nFolder Name:", folder_path, "\n")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            extract_information_from_txt(file_path)
        else:
            if filename == "analysis":
                continue
            else:
                if os.path.isdir(file_path):
                    extract_information_from_folder(file_path)


def extract_information_from_txt(file_path):
    if file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.read().split("USER:")
            file.close()

            original_json = []
            output_text = []
            for line in lines:
                strings = line.split("Assistant:")
                if len(strings) == 2:
                    original_json.append(strings[0])
                    output_text.append(strings[1])
            assert (len(original_json) == len(output_text))

            for meme in range(len(original_json)):
                input_prompt = output_text[meme].split("Assistant: ")[0] + "Assistant: "
                output_response = output_text[meme].split("Assistant: ")[-1]

                doc_input = nlp(input_prompt)
                doc_output = nlp(output_response)
                # Count the number of tokens
                global total_input_tokens
                total_input_tokens += len(doc_input)
                global total_output_tokens
                total_output_tokens += len(doc_output)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, file_path


extract_information_from_folder(
    "/Users/naquee.rizwan/Desktop/hatevlms/output/output_LLaVA_13B_8_bit")
print("Total input tokens:", total_input_tokens)
print("Total output tokens:", total_output_tokens)