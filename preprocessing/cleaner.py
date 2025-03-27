import re
from Config import *


def de_duplication(data):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english": [
            "(?:Aspiegel|\\*{5}\\(PERSON\\)) Customer Support team,?",
            "(?:Aspiegel|\\*{5}\\(PERSON\\)) SE is a company.*?Ireland\\.?",
            "(?:Aspiegel|\\*{5}\\(PERSON\\)) SE is the provider.*?",
        ],
        # add other languages...
    }

    cu_pattern = "|".join(f"({x})" for x in sum(cu_template.values(), []))

    email_patterns = [
        "(From\\s?:\\s?xxxxx@xxxx.com Sent\\s?:.{30,70}Subject\\s?:)",
        "(On.{30,60}wrote:)", "(Re\\s?:|RE\\s?:)",
        "(\\*{5}\\(PERSON\\) Support issue submit)", "(\\s?\\*{5}\\(PHONE\\))*$"
    ]
    split_pattern = "|".join(email_patterns)

    for t in data["Ticket id"].unique():
        df = data[data["Ticket id"] == t]
        ic_set = set()
        ic_deduplicated = []

        for ic in df[Config.INTERACTION_CONTENT]:
            ic_parts = re.split(split_pattern, ic)
            ic_parts = [p for p in ic_parts if p and p.strip()]
            ic_parts = [re.sub(split_pattern, "", p.strip()) for p in ic_parts]
            ic_parts = [re.sub(cu_pattern, "", p.strip()) for p in ic_parts]
            unique_parts = [p + "\n" for p in ic_parts if p not in ic_set]
            ic_set.update(ic_parts)
            ic_deduplicated.append(" ".join(unique_parts))

        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated

    data[Config.INTERACTION_CONTENT] = data["ic_deduplicated"]
    return data.drop(columns=["ic_deduplicated"])
