import os
import numpy as np
import pickle
import timeit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import torch
from sklearn.ensemble import GradientBoostingClassifier
import torchmetrics
from sklearn.model_selection import  RandomizedSearchCV
import numpy as np
RANDOM_STATES = [0,5, 10,15,20,25,30,35,40]

# read the X and y data
FEATURES_SUBSET= "mosaik+env" # ["env_only", "mosaik_only", "mosaik+env"]

img_res = "250m" #"640m" or "6km"
num_feats = "512" #"1024" or "8192" 
model_type = "xgboost" # "ridge" or "xgboost"

data_dir = f"/atlas/u/amna/mosaiks/output_mosaiks/{img_res}_{num_feats}_feats"
x_train_path =os.path.join("/atlas/u/amna/mosaiks/output_mosaiks/250m_512_feats", f"x_{FEATURES_SUBSET}_train.npy") #os.path.join(data_dir, f"x_{FEATURES_SUBSET}_train.npy")
y_train_path = os.path.join("/atlas/u/amna/mosaiks/output_mosaiks/250m_512_feats", f"y_{FEATURES_SUBSET}_train.npy")

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
# standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
#x_train=  2 * (x_train-x_train.min(axis = 0))/(x_train.max(axis = 0)- x_train.min(axis = 0)) -1
print(x_train.mean(axis = 0))
# save the scaler

scaler_filename = os.path.join("/atlas/u/amna/mosaiks", f"scaler_{FEATURES_SUBSET}.sc")
joblib.dump(scaler, scaler_filename) 
# def topkk(y, pred, numk=None):
#     topk = []
#     ks = []
#     for i in range(len(y)):
#         non_zeros = np.where(y[i] != 0)[0]
#         k = len(non_zeros)
#         if not numk is None:
#             k = min(k,numk)
#         if k==0:
#             pass
#         else:
#             non_zeros = np.where(y[i] != 0)[0]

#             non_zeros = np.argsort(y[i])[-k:]
#             #ks +=[k]
#             species_k = np.argsort(pred[i,:])[-k:]
#             acc = len([j for j in species_k if j in non_zeros])
#             acc = acc/k
#             topk += [acc]
#     return(np.mean(topk))

# def get_topks(y, pred):
    
#     top10 = topkk(y, pred, numk=10)
#     top30 = topkk(y, pred, numk=30)
#     topk = topkk(y, pred)
#     return({"top10":top10, "top30":top30, "topk":topk})
# def top_k(target, preds):
#     #preds, target = self._input_format(preds, target)
    
#     correct = 0
#     total = 0
    
#     assert preds.shape == target.shape
    
#     non_zero_counts =[len(np.nonzero(elem)[0]) for elem in target]
#     for i, elem in enumerate(target):
#         ki = non_zero_counts[i]
#         i_pred = preds[i].argsort()[-ki:][::-1]
       
#         i_targ = target[i].argsort()[-ki:][::-1]
#         if ki == 0 :
#             correct += 1
#         else:
#             correct += len(set(i_pred).intersection(i_targ)) / ki
#     total += target.shape[0]

#     return (correct / total)

# # train on non-songbirds subset
# #subset_file = "/network/projects/_groups/ecosystem-embeddings/species_splits/not_songbirds_idx.npy"
# #non_songbird_ids = np.load(subset_file)

# #y_train_non_songbird = y_train[:, non_songbird_ids]
# # X -> (n_samples, n_features)
# # y -> (n_samples, n_outputs_species)

# # Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=2)

print(f"X shape: {x_train.shape}")
print(f"y shape: {y_train.shape}")

def main():
    print("Starting")
    for RANDOM_STATE in RANDOM_STATES:
        print("running")
        start = timeit.default_timer()
        # model fitting
        #model = #GradientBoostingRegressor(random_state=RANDOM_STATE)
        model = xgb.XGBClassifier(objective="binary:logistic",tree_method = "gpu_hist", seed = RANDOM_STATE, random_state = RANDOM_STATE)
        params = {
    "colsample_bytree": np.random.uniform(0.7, 0.3,3),
    "gamma": np.random.uniform(0, 0.5,3),
    "learning_rate": np.random.uniform(0.03, 0.3,3), # default 0.1 
    "max_depth": np.random.randint(2, 6,3), # default 3
    "n_estimators": np.random.randint(100, 150,3), # default 100
    "subsample": np.random.uniform(0.6, 0.4,3)
}

        search = RandomizedSearchCV(model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)

        search.fit(x_train, y_train)
        # model.fit(x_train, y_train) #, tree_method = "gpu_hist", single_precision_histogram=True)).fit(x_train, y_train)
        #regr = MultiOutputRegressor(Ridge(random_state=RANDOM_STATE)).fit(x_train, y_train) #_non_songbird)
        
        # save the classifier
        with open(os.path.join("/atlas/u/amna/mosaiks",f'mosaik_{model_type}_{FEATURES_SUBSET}_{RANDOM_STATE}_all.pkl'), 'wb') as fid:
            pickle.dump(search, fid)

        # prediction result for the test data
        x_test_path = os.path.join(data_dir, f"x_{FEATURES_SUBSET}_test.npy")
        y_test_path = os.path.join(data_dir, f"y_{FEATURES_SUBSET}_test.npy")
        x_val_path = os.path.join(data_dir, f"x_{FEATURES_SUBSET}_val.npy")
        
        y_val_path = os.path.join(data_dir, f"y_{FEATURES_SUBSET}_val.npy")
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        #y_test_non_songbird = y_test[:, non_songbird_ids]
        
        print("y_val_shape")
        print(y_val.shape)

        x_test = scaler.transform (x_test)
        x_val = scaler.transform(x_val)
        #x_val=  2 * (x_val-x_val.min(axis = 0))/(x_val.max(axis = 0)- x_val.min(axis = 0)) -1
        #x_test=  2 * (x_test-x_test.min(axis = 0))/(x_test.max(axis = 0)- x_test.min(axis = 0)) -1
        y_pred = search.predict(x_test)
        y_val_pred = search.predict(x_val)
        np.save(os.path.join(data_dir, f"y_{model_type}_{FEATURES_SUBSET}_{RANDOM_STATE}_test_pred.npy"), y_pred)
        np.save(os.path.join(data_dir, f"y_{model_type}_{FEATURES_SUBSET}_{RANDOM_STATE}_val_pred.npy"), y_val_pred)
        # r_score = regr.score(x_test, y_test) #_non_songbird)
        # print(f"R score: {r_score}")

        print('acc ',torchmetrics.functional.accuracy(torch.tensor(y_pred),torch.tensor(y_test),task='binary'))
        print('f1 ',torchmetrics.functional.f1_score(torch.tensor(y_pred),torch.tensor(y_test),task='binary'))
        print('AUROC ',torchmetrics.AUROC(num_classes=2,task='binary')(torch.tensor(y_pred),torch.tensor(y_test)))
        print('recall ',torchmetrics.Recall(num_classes=1,task='binary')(torch.tensor(y_pred),torch.tensor(y_test)))
        print('precision ',  torchmetrics.Precision(num_classes=1,task='binary')(torch.tensor(y_pred),torch.tensor(y_test)))
        stop = timeit.default_timer()

        print('Elapsed time: ', stop - start)  

if __name__=="__main__":
    print("starting")
    main()