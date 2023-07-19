import argparse
import csv
import os
import random

import pandas as pd
from tqdm import tqdm

FINE_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_val_split_txt", type=str, required=True)
    arg_parser.add_argument("--test_split_txt", type=str, required=True)
    arg_parser.add_argument("--labels_csv", type=str, required=True)
    arg_parser.add_argument("--val_ratio", type=float, default=0.2)
    args = arg_parser.parse_args()

    output_folder = os.path.join(os.path.dirname(args.train_val_split_txt), "processed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    def is_normal(keywords):
        return keywords == 'No Finding'


    def to_labels(keywords):
        label = {label_name: 0 for label_name in FINE_LABELS}
        for token in keywords.split("|"):
            if token in label.keys():
                label[token] = 1
            else:
                assert token == 'No Finding'

        return label


    with open(args.train_val_split_txt, 'r') as f:
        train_val_list = set([line.strip() for line in f.readlines()])

    with open(args.test_split_txt, 'r') as f:
        test_list = set([line.strip() for line in f.readlines()])

    train_val_records = dict()
    test_records = dict()
    df = pd.read_csv(args.labels_csv)
    for _, row in tqdm(df.iterrows(), desc=f"Processing", total=len(df)):
        patient_id = row['Patient ID']
        filename = row['Image Index']
        findings = row['Finding Labels']
        records = train_val_records if filename in train_val_list else test_records

        normal_abnormal = 0 if is_normal(findings) else 1
        fine_grained_labels = to_labels(findings)
        if patient_id not in records:
            records[patient_id] = {
                "filenames": [filename],
                "normal-abnormal": [normal_abnormal],
                **{label_name: [label] for label_name, label in fine_grained_labels.items()}
            }
        else:
            records[patient_id]["filenames"].append(filename)
            records[patient_id]["normal-abnormal"].append(normal_abnormal)
            for label_name, label in fine_grained_labels.items():
                records[patient_id][label_name].append(label)

    patient_ids = list(train_val_records.keys())
    random.shuffle(patient_ids)
    val_count = int(len(patient_ids) * args.val_ratio)
    val_ids = patient_ids[:val_count]
    train_ids = patient_ids[val_count:]
    test_ids = list(test_records.keys())

    for prefix, ids, records in [("train", train_ids, train_val_records), ("val", val_ids, train_val_records),
                                 ("test", test_ids, test_records)]:
        new_csv = os.path.join(output_folder, f"{prefix}.csv")
        with open(new_csv, 'w', newline='') as w:
            csv_writer = csv.writer(w)
            csv_writer.writerow(['patient_id', 'filename', 'normal-abnormal'] + FINE_LABELS)
            for patient_id in ids:
                record = records[patient_id]
                for i in range(len(record["filenames"])):
                    row = [patient_id, record["filenames"][i], record["normal-abnormal"][i]]
                    row += [record[label_name][i] for label_name in FINE_LABELS]
                    csv_writer.writerow(row)
