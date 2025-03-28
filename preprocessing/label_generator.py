def generate_combined_labels(df):
    df["type_2"] = df["y2"]
    df["type_3"] = df["y3"]
    df["type_4"] = df["y4"]

    df["type_2_3"] = df["type_2"].astype(str) + "_" + df["type_3"].astype(str)
    df["type_2_3_4"] = df["type_2_3"] + "_" + df["type_4"].astype(str)

    return df
