import cv2
import pandas as pd
from tqdm import tqdm
import os

IMAGE_DIR = "/atlas2/u/jonxuxu/datasets/merged"
CSV_FILE = "/atlas2/u/jonxuxu/datasets/merged.csv"

df = pd.read_csv(CSV_FILE, usecols=["filename"])
df.reset_index(inplace=True, drop=True)
examples = df["filename"]

for index in range(len(examples)):
    try:
        path = os.path.join(IMAGE_DIR, examples[index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        print(examples[index])
