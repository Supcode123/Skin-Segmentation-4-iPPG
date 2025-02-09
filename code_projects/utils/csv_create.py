import csv
from datetime import datetime
import os


def csv_file(path, accuracy, miou_skin, classes):

    file_name = "Multi_Binary.csv"
    file_path = os.path.join(path, file_name)
    file_exists = os.path.isfile(file_path)
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = [
        datetime.now().strftime("%m-%d %H:%M:%S"),  # Timestamp
        "Multi_labels(merge)" if classes == 18 else "Binary",  # Model
        accuracy,  # PA
       # miou,  # IoU(total)
        miou_skin  # IoU(Non_Skin)
    ]
    header = ["timestamp", "Model", "val_skin_accuracy", "M_IoU(Skin)"]

    if file_exists:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        model_found = False
        for i, row in enumerate(rows):
            if row and row[1] == data[1]:
                rows[i] = data
                model_found = True
                break

        if not model_found:
            rows.append(data)
    else:
        rows = [header, data]

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("csv_file saved.")