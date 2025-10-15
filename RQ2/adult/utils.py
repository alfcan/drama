def get_label(dataset):
    if "adult" in dataset:
        return ("income", 1)
    if "ad" in dataset:
        return ("y", 1)
    if "athlete" in dataset:
        return ("Medal_Gold", 1)
    if "diabetic" in dataset:
        return ("diabetesMed", 0)
    if "ibm" in dataset:
        return ("Attrition", 0)
    if "nursery" in dataset:
        return ("y", 2)
    if "placement" in dataset:
        return ("status", 1)
    if "vaccine" in dataset:
        return ("lowtrustvaccinerec", 0)
    if "us" in dataset:
        return ("dIncome1", 3)
    if "bank" in dataset:
        return ("loan", 1)
    # if "aps" in dataset:
    #     return "class"
    if "cmc" in dataset:
        return ("contr_use", 2)
    if "compas" in dataset:
        return ("two_year_recid", 0)
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100)
    if "drug" in dataset:
        return ("y", 0)
    if "german" in dataset:
        return ("credit", 1)
    if "healt" in dataset:
        return ("y", 1)
    if "hearth" in dataset:
        return ("y", 0)
    # if "kickstarter" in dataset:
    #     return "State"
    if "law" in dataset:
        return ("gpa", 2)
    if "medical" in dataset:
        return ("IsChallenge", 0)
    if "obesity" in dataset:
        return ("y", 0)
    if "park" in dataset:
        return ("score_cut", 0)
    if "resyduo" in dataset:
        return ("tot_recommendations", 1)
    if "student" in dataset:
        return ("y", 1)
    if "wine" in dataset:
        return ("quality", 6)
    if "ricci" in dataset:
        return ("Combine", 1)
    return ("y", 1)
