import json
import os.path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import re


def parse_and_calculate_metrics():
    input_file_path = os.path.join('output_instructBLIP_flan_t5_MAMI_memes',
                                   'instruct_blip_flan_t5_mami_dataset_definition_ocr_input_explanation_output_0.txt')
    with open(input_file_path, 'r') as file:

        lines = file.read().split("\n##########\n")
    file.close()

    original_json = []
    output_text = []
    for line in lines:
        strings = line.split("\n----------\n")
        if len(strings) == 2:
            original_json.append(strings[0])
            output_text.append(strings[1])

    assert (len(original_json) == len(output_text))

    original_label = []
    predicted_label = []
    count_other = 0

    for meme in range(len(original_json)):
        test_label = output_text[meme].split("Assistant:")[1].strip()

        pattern = re.compile(r'(?i)\bnot-misogynistic\b|\bnot\s*misogynistic\b|\bmisogynistic\b(?!-)')
        matches_iter = pattern.finditer(test_label)
        matches = [match.group().lower() for match in matches_iter]

        # caveat for instructBLIP: ....does not contain any misogynistic content. So this meme is not-misogynistic.
        # caveat resolved

        if test_label.lower().__contains__("not not-misogynistic") or test_label.lower().__contains__(
                "not not misogynistic") or test_label.lower().__contains__(
                "not 'not-misogynistic'") or test_label.lower().__contains__("not 'not misogynistic'"):
            original_label.append(json.loads(original_json[meme])['label'])
            predicted_label.append(1)
        elif "misogynistic" in matches and ("not-misogynistic" in matches or "not misogynistic" in matches):
            matches_iter = pattern.finditer(test_label)
            not_present_before_misogynistic = False
            for match in matches_iter:
                if match.group().lower() == "misogynistic":
                    start, end = match.span()
                    last_five_words = test_label[:start].split()[-5:]
                    # print(last_five_words)
                    if any(word.lower() == "not" for word in last_five_words):
                        not_present_before_misogynistic = True
                    else:
                        not_present_before_misogynistic = False
                        break
            if not_present_before_misogynistic:
                original_label.append(json.loads(original_json[meme])['label'])
                predicted_label.append(0)
            else:
                count_other += 1
                print(json.loads(original_json[meme])['img'])
        elif "not-misogynistic" in matches or "not misogynistic" in matches:
            original_label.append(json.loads(original_json[meme])['label'])
            predicted_label.append(0)
        elif "misogynistic" in matches:
            matches_iter = pattern.finditer(test_label)
            not_present_before_misogynistic = False
            for match in matches_iter:
                if match.group().lower() == "misogynistic":
                    start, end = match.span()
                    last_five_words = test_label[:start].split()[-5:]
                    # print(last_five_words)
                    if any(word.lower() == "not" for word in last_five_words):
                        not_present_before_misogynistic = True
                    else:
                        not_present_before_misogynistic = False
                        break
            if not_present_before_misogynistic:
                original_label.append(json.loads(original_json[meme])['label'])
                predicted_label.append(0)
            else:
                original_label.append(json.loads(original_json[meme])['label'])
                predicted_label.append(1)
        else:
            count_other += 1
            print(json.loads(original_json[meme])['img'])

    assert (len(original_label) == len(predicted_label))

    cnt = [0, 0, 0, 0]
    for index in range(len(original_label)):
        if original_label[index] == 0 and predicted_label[index] == 0:
            cnt[0] = cnt[0] + 1
        elif original_label[index] == 0 and predicted_label[index] == 1:
            cnt[1] = cnt[1] + 1
        elif original_label[index] == 1 and predicted_label[index] == 0:
            cnt[2] = cnt[2] + 1
        elif original_label[index] == 1 and predicted_label[index] == 1:
            cnt[3] = cnt[3] + 1
    print("Array for confusion matrix : OriginalLabel-PredictedLabel :\nNot_misogynistic-Not_misogynistic, "
          "Not_misogynistic-misogynistic,"
          "misogynistic-Not_misogynistic, misogynistic-misogynistic")
    for value in cnt:
        print('         ', value, end='       ')
    print("\n--------------------------------------------------")

    # compute the confusion matrix
    cm = confusion_matrix(original_label, predicted_label)

    # Plot the confusion matrix.
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',
        xticklabels=['Not misogynistic', 'misogynistic'],
        yticklabels=['Not misogynistic', 'misogynistic']
    )
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('Actual Label', fontsize=10)
    plt.title('Confusion Matrix', fontsize=20)
    # plt.show()

    print("Classification Report\n")
    print(classification_report(original_label, predicted_label))
    print("--------------------------------------------------")

    print(eval_metric(original_label, predicted_label))
    print("Third category Count: ", count_other)


def eval_metric(y_true, y_prediction):
    accuracy = accuracy_score(y_true, y_prediction)
    macro_f_one_score = f1_score(y_true, y_prediction, average='macro')
    f_one_score = f1_score(y_true, y_prediction)
    fpr, tpr, _ = roc_curve(y_true, y_prediction)
    area_under_c = auc(fpr, tpr)
    recall = recall_score(y_true, y_prediction)
    precision = precision_score(y_true, y_prediction)
    return {
        "accuracy": round(accuracy, 4),
        'macro_f_one_score': round(macro_f_one_score, 4),
        'f_one_score': f_one_score,
        'area_under_c': round(area_under_c, 4),
        'precision': precision,
        'recall': recall
    }


parse_and_calculate_metrics()
