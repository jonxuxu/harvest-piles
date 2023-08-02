import pandas as pd
import os
import shutil

FOLDER_PATH = "/atlas2/u/jonxuxu/datasets"
merged_df = pd.read_csv(os.path.join(FOLDER_PATH, "merged_labels.csv"))
merged_df = merged_df.iloc[:, 1:]
start_idx = len(merged_df)

src_df = pd.read_csv(os.path.join(FOLDER_PATH, "amhara_skysat_all_clip.csv"))
src_df = src_df.iloc[:, 1:]

IMG_PATH = os.path.join(FOLDER_PATH, "amhara_skysat_all", "amhara_skysat_all_clip")
DEST_FOLDER = os.path.join(FOLDER_PATH, "merged")
for index, row in src_df.iterrows():
    src_path = os.path.join(IMG_PATH, row["filename"])
    dest_path = os.path.join(DEST_FOLDER, str(index + start_idx) + ".tif")
    # print(src_path)
    # print(dest_path)
    # break
    try:
        shutil.copy(src_path, dest_path)
        src_df.at[index, "filename"] = str(index + start_idx) + ".tif"
    except Exception as e:
        print(e)
    if index % 100 == 0:
        print(index)


new_df = pd.concat([merged_df, src_df], ignore_index=True)
new_df.to_csv("out.csv")
shutil.copy("out.csv", "/atlas2/u/jonxuxu/datasets/merged_labels.csv")
