import pandas as pd
import re
from Config import *
from preprocessing.noise_remover import *
from preprocessing.cleaner import *

def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_input_data() -> pd.DataFrame:
    df = pd.read_csv("./data/AppGallery.csv", skipinitialspace=True)
    df.rename(columns={'Type 1': 'y1', 'Type 2': 'y2',
              'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype(
        'U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    df["y"] = df[Config.CLASS_COL]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df


def translate_to_en(texts: list[str]):
    import stanza
    from stanza.pipeline.core import DownloadMethod
    from transformers import pipeline
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)

    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES)
    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l = text_en_l + [text]
            continue

        doc = nlp_stanza(text)
        # print(doc.lang)
        if doc.lang == "en":
            text_en_l = text_en_l + [text]
            # print(text)
        else:
            # convert to model supported language code
            # https://stanfordnlp.github.io/stanza/available_models.html
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/m2m_100/tokenization_m2m_100.py
            lang = doc.lang
            if lang == "fro":  # fro = Old French
                lang = "fr"
            elif lang == "la":  # latin
                lang = "it"
            elif lang == "nn":  # Norwegian (Nynorsk)
                lang = "no"
            elif lang == "kmr":  # Kurmanji
                lang = "tr"

            case = 2

            if case == 1:
                text_en = t2t_pipe(
                    text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                text_en = text_en[0]['generated_text']
            elif case == 2:
                tokenizer.src_lang = lang
                encoded_hi = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]
            else:
                text_en = text
            text_en_l = text_en_l + [text_en]
            # print(text)
            # print(text_en)
    return text_en_l
