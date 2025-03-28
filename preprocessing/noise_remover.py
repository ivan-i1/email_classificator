import pandas as pd
from Config import *
import re

def noise_remover(df: pd.DataFrame):
    noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(
        noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()
    noise_1 = [
        "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\d{2}(:|.)\d{2}",
        "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you"
        "in order to give you the best solution",
        "could you please clarify your request with following information:"
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\d+",
        "[^0-9a-zA-Z]+",
        "(\s|^).(\s|$)"]
    for noise in noise_1:
        # print(noise)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(
            noise, " ", regex=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(
        r'\s+', ' ', regex=True).str.strip()
    # print(df.y1.value_counts())
    good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    df = df.loc[df.y1.isin(good_y1)]
    # print(df.shape)
    return df

