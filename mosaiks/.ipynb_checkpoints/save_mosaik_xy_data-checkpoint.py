import os
import pickle
import pandas as pd
import numpy as np
import json


img_res = "250m" #"640m" #6km#
NUM_FEATS = 512 #1024 #8192 
NUM_HOTSPOTS = 6987 #5924 #8439

FEATURES_SUBSET= "mosaik+env" # ["env_only", "mosaik_only", "mosaik+env"]


    
def save_features(df, save_dir, split_name):
    
    y_final = np.empty((len(df)))
    

    x_final = np.empty((len(df), NUM_FEATS)) 
    
    
    
    print(df.keys())
    for idx, row in df.iterrows():
        y_value = row['activity']
        curr_feat_idx = hotspot_to_idx[row['filename']]

        assert row['filename'] == hotspots_feats['ids_X'][curr_feat_idx]
        
        
        x_final[idx] = hotspots_feats['X'][curr_feat_idx]
        
        
            
        y_final[idx]=y_value
            #y_final[idx] = y_arr*correction_t
    
    # save the x features
    np.save(os.path.join(save_dir, f"x_{FEATURES_SUBSET}_"+split_name), x_final)
    # save the y labels
    np.save(os.path.join(save_dir,  f"y_{FEATURES_SUBSET}_"+split_name), y_final)
    
    
if __name__ == '__main__':
    # hotspots feature matrix
    
    
    main_dir = f"/atlas/u/amna/mosaiks/output_mosaiks/{img_res}_{NUM_FEATS}_feats"
    hotspots_feats_file = os.path.join(main_dir, f"{img_res}_jpg_visual_{NUM_FEATS}_feats.pkl")
    hotspots_feats_file = "/atlas/u/amna/mosaiks/output_mosaiks/250m_128_feats/imgs_v2_128_feats.pkl"
    hotspots_feats = None
    with open(hotspots_feats_file, 'rb') as f:
        hotspots_feats = pickle.load(f)

    # create a dictionary for hotspot names mapped to their indices in 'hotspots_feats'
    hotspot_to_idx = {} # key: value -> 'hotspot_name: zero_index'
    for idx in range(len(hotspots_feats['ids_X'])):
        hotspot_to_idx[hotspots_feats['ids_X'][idx]] = idx

    # read the x_train, x_val, x_test data
    train_path = "/atlas2/u/jonxuxu/datasets/train.csv"
    #"/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/train_june_vf.csv"
    val_path = "/atlas2/u/jonxuxu/datasets/test.csv"
    #"/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/val_june_vf.csv"
    test_path = "/atlas2/u/jonxuxu/datasets/test.csv"
    #"/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/test_june_vf.csv"


    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"len(hotspots_feats['X']): {len(hotspots_feats['X'])}")
    print(f"len(hotspots_feats['X'][0]): {len(hotspots_feats['X'][0])}")
    
    num_hotspots, num_feats = hotspots_feats['X'].shape
    assert num_hotspots == NUM_HOTSPOTS
    print(num_feats)
    assert num_feats == NUM_FEATS
    
    

    save_features(val_df, main_dir, 'val')
    save_features(test_df, main_dir, 'test')
    save_features(train_df, main_dir, 'train')