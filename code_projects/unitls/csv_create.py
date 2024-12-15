import csv
from datetime import datetime
import os


def csv_file(path, accuracy, miou, miou_skin, classes):

    file_name = "Multi/Binary.csv"
    file_exists = os.path.isfile(os.path.join(path, file_name))

    data = []
    header = ["timestamp", "Model", "PA(%)", "IoU(Skin)", "IoU(Non_Skin)"]
    data[0] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if classes == 18:
        data[1] = "Multi_labels(merge)"
    if classes == 2:
        data[1] = "Binary"

    data[2] = accuracy.item()
    data[3] = miou.item()
    data[4] = miou_skin.item()

    with open(file_name, "a", newline="") as f:
         writer = csv.writer(f)
         if not file_exists:
             writer.writerow(header)
         writer.writerow(data)

    print("csv_file saved.")