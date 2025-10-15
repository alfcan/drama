def get_label(dataset):
    if "adult" in dataset:
        return ("income", 1)
    if "diabetic" in dataset:
        return ("readmitted", 0)
    if "bank" in dataset:
        return ("deposit", 1)
    # if "aps" in dataset:
    #     return "class"
    if "compas" in dataset:
        return ("two_year_recid", 0)
    return ("y", 1)
