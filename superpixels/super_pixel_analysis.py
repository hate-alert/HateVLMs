import json
import math
import os.path
from tqdm import tqdm

from data_set import DatasetWrapper

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from matplotlib.backends.backend_pdf import PdfPages


# -------------------------------------------------- FUNCTIONS --------------------------------------------------

tag = "hateful"


def eval_metric(y_true, y_prediction):
    accuracy = accuracy_score(y_true, y_prediction)
    macro_f_one_score = f1_score(y_true, y_prediction, average='macro')
    fpr, tpr, _ = roc_curve(y_true, y_prediction)
    area_under_c = auc(fpr, tpr)
    return {
        "accuracy": round(accuracy*100, 2),
        'macro_f_one_score': round(macro_f_one_score*100, 2),
        'area_under_c': round(area_under_c*100, 2)
    }


def check_if_string_contains_a_list_of_words(string, list_of_words):
    contains = False
    for word in list_of_words:
        contains = contains or string.lower().__contains__(word)
    return contains


def calculateHatefulNotHatefulAmbiguous(prompt_output):

    ambiguous_cases = [
        tag + " or not-" + tag,
        tag + " or not " + tag,
        tag + " or not a " + tag,
        "not-" + tag + " or " + tag,
        "not " + tag + " or " + tag,
        "not a " + tag + " or " + tag
    ]

    positive_cases = [tag]

    negative_cases = ["not " + tag, "not-" + tag, "not a " + tag]

    if check_if_string_contains_a_list_of_words(prompt_output, ambiguous_cases):
        return 2
    else:
        if check_if_string_contains_a_list_of_words(prompt_output, negative_cases):
            return 0
        elif check_if_string_contains_a_list_of_words(prompt_output, positive_cases):
            return 1
        else:
            return 2

# -------------------------------------------------- PARSER --------------------------------------------------


dataset_path = os.path.join("/Users/naquee.rizwan/Desktop/Projects/HateVLMs/data/dev_seen.json")
data_set = {}
with open(dataset_path, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    data_set[result["id"]] = result
    print(result["gold_attack"], result["gold_pc"])

files_hashmap = {

}

target_hashmap = {
    "race": 0,
    "religion": 0,
    "sex": 0,
    "nationality": 0,
    "disability": 0,
    "pc_empty": 0
}

full_target = {
    "race": 0,
    "religion": 0,
    "sex": 0,
    "nationality": 0,
    "disability": 0,
    "pc_empty": 0
}

maximum_slic_samples = 0

with open("idefics_8_bit_slic_fhm_all.txt", "r") as output_file:
    outputs = output_file.read().split("\n##########\n")
    print("Total sample points :", len(outputs)-1)

    for total_output in outputs:
        if "\n$$$$$$$$$$\n" not in total_output:
            continue

        assert len(total_output.split("\n$$$$$$$$$$\n")) == 2
        gold_label = total_output.split("\n$$$$$$$$$$\n")[0]
        print(json.loads(gold_label)["slic"])
        assert json.loads(gold_label)["label"] in [0, 1]
        output = total_output.split("\n$$$$$$$$$$\n")[-1]
        slic_samples = output.split("\n----------\n")
        index = 0
        for slic_sample in slic_samples:
            if len(slic_sample.split("\n..........\n")) == 2:
                slic_image_id = slic_sample.split("\n..........\n")[0].strip()
                output_slic = slic_sample.split("\n..........\n")[1].split("Assistant:")[1].strip()
                assert slic_image_id not in files_hashmap
                files_hashmap[slic_image_id] = {
                    "output": output_slic,
                    "label": calculateHatefulNotHatefulAmbiguous(output_slic)
                }
                file_name = slic_image_id.split("/")[-1].split(".")[0]
                if "_" not in file_name:
                    print(data_set[int(file_name)])
                    assert len(data_set[int(file_name)]["gold_hate"]) == 1

                    files_hashmap[slic_image_id]["gold_pc"] = data_set[int(file_name)]["gold_pc"]
                    files_hashmap[slic_image_id]["gold_attack"] = data_set[int(file_name)]["gold_attack"]
                    files_hashmap[slic_image_id]["gold_hate"] = 0 if data_set[int(file_name)]["gold_hate"][0] == "not_hateful" else 1

                    if files_hashmap[slic_image_id]["gold_hate"] != files_hashmap[slic_image_id]["label"]:
                        for item in files_hashmap[slic_image_id]["gold_pc"]:
                            target_hashmap[item] += 1

                    for item in files_hashmap[slic_image_id]["gold_pc"]:
                        full_target[item] += 1

                index += 1

        maximum_slic_samples = max(maximum_slic_samples, index)

    print(target_hashmap)
    print(full_target)


with open(
        os.path.join("output_jsons", "idefics_8_bit_slic_fhm.json"),
        "w"
) as json_output:
    json.dump(files_hashmap, json_output, indent=4)

print("Maximum Slic Samples :", maximum_slic_samples)

# -------------------------------------------------- ANALYSIS --------------------------------------------------

dataset_wrapper = DatasetWrapper(
    dataset_tag="facebook_hateful_meme_dataset",
    base_folders=["/Users/naquee.rizwan/Desktop/KaggleDataset"],
    split="dev_slic"
)

slic_files_misclassified_to_not_hateful_and_changed = PdfPages(
    os.path.join(
        dataset_wrapper.get_base_folders()[0],
        'slic_files_misclassified_to_not_' + tag + '_and_changed.pdf'
    )
)

slic_files_misclassified_to_not_hateful_and_not_changed = PdfPages(
    os.path.join(
        dataset_wrapper.get_base_folders()[0],
        'slic_files_misclassified_to_not_' + tag + '_and_not_changed.pdf'
    )
)

slic_files_misclassified_to_hateful_and_changed = PdfPages(
    os.path.join(
        dataset_wrapper.get_base_folders()[0],
        'slic_files_misclassified_to_' + tag + '_and_changed.pdf'
    )
)

slic_files_misclassified_to_hateful_and_not_changed = PdfPages(
    os.path.join(
        dataset_wrapper.get_base_folders()[0],
        'slic_files_misclassified_to_' + tag + '_and_not_changed.pdf'
    )
)

predicted_label = []
real_label = []

differences = 0
total_stored_points = 0

for data in tqdm(dataset_wrapper.get_dataset()):
    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = math.ceil(len(data["slic"]) / 3.0)

    ground_label = data["label"]

    slic_predictions = []

    increment = 1
    for iterator, slic_data_point in enumerate(data["slic"]):
        if slic_data_point not in files_hashmap:
            continue

        label = files_hashmap[slic_data_point]["label"]

        # Assertion for data quality check
        if iterator > 0:
            assert "_" in slic_data_point.split("/")[-1]
            assert slic_data_point.split("/")[-1].split(".")[0].split("_")[-1] == str(iterator-1)
        else:
            assert "_" not in slic_data_point.split("/")[-1]
            if label != 2:
                predicted_label.append(label)
                real_label.append(ground_label)
                if label != ground_label:
                    differences += 1

        slic_predictions.append(label)

        if iterator == 0 or label != files_hashmap[data["slic"][0]]["label"]:
            fig.add_subplot(rows, columns, increment)
            plt.imshow(
                Image.open(os.path.join(dataset_wrapper.get_base_folders()[0], slic_data_point))
            )
            plt.axis('off')
            increment += 1

            match label:
                case 0:
                    title_subplot = "Not-Hateful"
                case 1:
                    title_subplot = "Hateful"
                case 2:
                    title_subplot = "Ambiguous"
                case _:
                    title_subplot = "Error"

            plt.title(title_subplot)

    match ground_label:
        case 0:
            title_sup_plot = "Not-Hateful"
        case 1:
            title_sup_plot = "Hateful"
        case _:
            title_sup_plot = "Error in parsing"

    main_title = "Image : " + data["img"].split("/")[-1] + " ---- Ground Truth : " + title_sup_plot
    plt.suptitle(main_title)

    if len(slic_predictions) > 0 and slic_predictions[0] != 2:
        are_all_values_equal = True
        for slic_prediction in slic_predictions:
            if slic_prediction != slic_predictions[0]:
                are_all_values_equal = False
                break

        if ground_label == 1 and slic_predictions[0] == 0:
            total_stored_points += 1
            if are_all_values_equal:
                slic_files_misclassified_to_not_hateful_and_not_changed.savefig(fig)
            else:
                slic_files_misclassified_to_not_hateful_and_changed.savefig(fig)
        elif ground_label == 0 and slic_predictions[0] == 1:
            total_stored_points += 1
            if are_all_values_equal:
                slic_files_misclassified_to_hateful_and_not_changed.savefig(fig)
            else:
                slic_files_misclassified_to_hateful_and_changed.savefig(fig)

    plt.close()

slic_files_misclassified_to_not_hateful_and_not_changed.close()
slic_files_misclassified_to_not_hateful_and_changed.close()
slic_files_misclassified_to_hateful_and_not_changed.close()
slic_files_misclassified_to_hateful_and_changed.close()

assert len(predicted_label) == len(real_label)
print(len(predicted_label), differences, total_stored_points)
print(eval_metric(real_label, predicted_label))
