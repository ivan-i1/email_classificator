from sklearn.metrics import accuracy_score


def dependency_aware_accuracy(y_true, y_pred, level="type_2"):

    correct = 0
    total = len(y_true)

    if level in ["type_3", "type_4"]:
        if "type_3" not in y_pred or "type_4" not in y_pred:
            y_pred["type_3"] = [
                s.split("_")[1] if "_" in s else "NA" for s in y_pred["type_2_3_4"]]
            y_pred["type_4"] = [
                s.split("_")[2] if "_" in s else "NA" for s in y_pred["type_2_3_4"]]

    for i in range(total):
        t2_correct = y_pred["type_2"][i] == y_true["type_2"].iloc[i]

        if level == "type_2" and t2_correct:
            correct += 1

        elif level == "type_3":
            t3_correct = y_pred["type_3"][i] == y_true["type_3"].iloc[i]
            if t2_correct and t3_correct:
                correct += 1

        elif level == "type_4":
            t3_correct = y_pred["type_3"][i] == y_true["type_3"].iloc[i]
            t4_correct = y_pred["type_4"][i] == y_true["type_4"].iloc[i]
            if t2_correct and t3_correct and t4_correct:
                correct += 1

    return correct / total
