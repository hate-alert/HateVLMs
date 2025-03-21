from sklearn.metrics import accuracy_score, f1_score

file_paths = ['821.txt', '822.txt', '823.txt']
dataset_tags = ["mami_", "covid_", "harmp_", "bhm_", "hinglish_"]


for file_path in file_paths:
    array = []
    print()
    print(file_path)
    print()
    with open(file_path, 'r') as file:
        for line in file:
            array.append(line.strip())

    file_names = array[-3].split(",")
    ground_truth_labels = [int(x.split(",")[0].strip().strip("().")) for x in array[-2].split("tensor") if x.strip()]
    predicted_logits = [int(x.split(",")[0].strip().strip("().")) for x in array[-1].split("tensor") if x.strip()]

    if len(file_names) != len(ground_truth_labels) or len(file_names) != len(predicted_logits):
        raise ValueError("Mismatch in lengths of file names, ground truth labels, and predicted logits")

    for dataset_tag in dataset_tags:
        print()
        print(dataset_tag)
        print()

        dataset_gt = []
        dataset_li = []

        for index, file_name in enumerate(file_names):
            if dataset_tag in file_name:
                dataset_gt.append(ground_truth_labels[index])
                dataset_li.append(predicted_logits[index])

        accuracy = accuracy_score(dataset_gt, dataset_li)
        mf1 = f1_score(dataset_gt, dataset_li, average='macro')

        print(f"Length: {len(dataset_li)}, {len(dataset_gt)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1 Score: {mf1:.4f}")