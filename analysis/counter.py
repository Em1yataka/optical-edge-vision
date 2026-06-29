from collections import Counter


def count_objects(result):
    names = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = result.names[cls_id]
        names.append(name)

    return Counter(names)