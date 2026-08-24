import os


def save_yolo_label(path, labels):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.writelines(line + "\n" for line in labels)
