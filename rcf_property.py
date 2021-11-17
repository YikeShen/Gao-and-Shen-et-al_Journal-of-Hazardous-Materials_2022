import pandas as pd
import numpy as np
import phate
import scipy
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


data = pd.read_csv('Data/rcf_lipid_combined_property.csv',encoding='cp1252')

E = data['E'].to_numpy().reshape(-1,1)
S = data['S'].to_numpy().reshape(-1,1)
A = data['A'].to_numpy().reshape(-1,1)
B = data['B'].to_numpy().reshape(-1,1)
V = data['V'].to_numpy().reshape(-1,1)


SMILES = data['SMILES'].to_numpy()
MW = data['MW'].to_numpy().reshape(-1,1)
KOW = data['log Kow'].to_numpy().reshape(-1,1)
OM = data['fom (%)'].to_numpy().reshape(-1,1)
flipid = data['flip (%)'].to_numpy().reshape(-1,1)

RCF_water = data['log RCF- water'].to_numpy()
RCF_soil = data['log RCF-soil'].to_numpy()

NOCount = []
NHOHCount = []
NumHAcceptors = []
NumHDonors = []
Num_Aro_rings = []
tpsa = []
Num_sat_rings = []
mr = []

for sm in SMILES:
    mol = Chem.MolFromSmiles(sm)
    
    NOCount.append(Descriptors.NOCount(mol))
    NHOHCount.append(Descriptors.NHOHCount(mol))
    Num_Aro_rings.append(Descriptors.NumAromaticRings(mol))
    NumHDonors.append(Descriptors.NumHDonors(mol))
    NumHAcceptors.append(Descriptors.NumHAcceptors(mol))
    tpsa.append(Descriptors.TPSA(mol))
    Num_sat_rings.append(Descriptors.NumSaturatedRings(mol))
    mr.append(Descriptors.MolMR(mol))

NOCount = np.array(NOCount).reshape(-1,1)
NHOHCount = np.array(NOCount).reshape(-1,1)
NumHAcceptors = np.array(NOCount).reshape(-1,1)
NumHDonors = np.array(NOCount).reshape(-1,1)
Num_Aro_rings = np.array(NOCount).reshape(-1,1)
tpsa = np.array(tpsa).reshape(-1,1)
Num_sat_rings = np.array(Num_sat_rings).reshape(-1,1)
mr = np.array(mr).reshape(-1,1)

features = np.concatenate((E,V,OM,flipid,S,A,B,MW,NOCount,NHOHCount,NumHAcceptors,NumHDonors,Num_Aro_rings,tpsa,Num_sat_rings),1)



from sklearn.manifold import TSNE
feature_embedded = TSNE(n_components=2,perplexity=15).fit_transform(features)
fig = plt.scatter(feature_embedded[:,0],feature_embedded[:,1],c=RCF_soil,s=5,cmap='RdYlGn')
plt.colorbar(fig)
plt.xlabel('t-SNE_1',fontsize=15)
plt.ylabel('t-SNE_2', fontsize=15)


features_z = scipy.stats.mstats.zscore(features,0)
n_sample = len(RCF_soil)


def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)


#run below for random shuffling of data
#total_id = np.arange(n_sample)
#np.random.shuffle(total_id)
#run below for an example of train/test split
total_id = np.load('Data/property_index246_all_esabv.npy')
splits = 5
train_split_index,test_split_index = Kfold(n_sample,splits)

prediction = []
prediction_true = []
test_score_all = []
feature_importance_impurity = []

importance_all_dots = []
feature_importance_permute = []
fig = []
for k in range(splits):
    
    print('batch is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    train_feature = [features_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [features_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [features_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    n_estimator = [100,200,250,500,750,1000]
    max_depths = [2,3,4,5,6]
    
    best_valid_score = 0
    for ne in n_estimator:
        for m_d in max_depths:
            model = GradientBoostingRegressor(n_estimators=ne,max_depth=m_d,random_state=8)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                pred = model.predict(test_feature)
                best_n = ne
                best_d = m_d
                
    model = GradientBoostingRegressor(n_estimators=best_n,max_depth=best_d).fit(np.array(train_feature),np.array(train_label))
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots.append(permut_importance.importances)
    
    
    print(test_score)
    prediction.append(pred)
    prediction_true.append(test_label)
    test_score_all.append(test_score)
    
    feature_importance_permute.append(permut_importance.importances_mean)
    print('best n_estimator is',best_n)
    print('best depth is',best_d)
    feature_importance_impurity.append(model.feature_importances_)


feature_importance_impurity_all = np.mean(feature_importance_impurity,0)


sns.barplot(['E','V','$f_{OM}$','$f_{lipid}$','S','A','B','MW','NOCount','NHOHCount','NumHAcceptors','NumHDonors','Num_Aro_rings','tpsa','Num_sat_rings'],feature_importance_impurity_all)#.set_title('Feature Importance')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.ylabel('Relative Importance',fontsize=14)
plt.tick_params(labelsize=14)


violin_dots = np.concatenate(np.array(importance_all_dots),1).transpose()
df = pd.DataFrame(data=violin_dots,columns=['E','V','$f_{om}$','$f_{lipid}$','S','A','B','MW','NOCount','NHOHCount','NumHAcceptors','NumHDonors','Num_Aro_rings','tpsa','Num_sat_rings'])

sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylabel('Relative Importance',fontsize=14)
plt.tick_params(labelsize=14)



prediction_gbdt = []
for l in prediction:
    for v in l:
        prediction_gbdt.append(v)



prediction_true_gbdt = []
for l in prediction_true:
    for v in l:
        prediction_true_gbdt.append(v)

sns.scatterplot(prediction_true_gbdt,prediction_gbdt)
sns.lineplot(np.arange(-3,2.),np.arange(-3,2.),color='r')
plt.xlabel('Measured logRCF',fontsize=14)
plt.ylabel('Predicted logRCF',fontsize=14)


train_split_index,test_split_index = Kfold(len(RCF_soil),5)

splits = 5
prediction_rf = []
prediction_true_rf = []
test_score_all_rf = []

feature_importance_impurity_rf = []
importance_all_dots_rf = []
feature_importance_permute_rf = []
for k in range(splits):
    
    print('batch is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    
    train_feature = [features_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [features_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [features_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    n_estimator = [50,100,200,300,400]
    max_depths = [2,3,4,5,6]
    
    best_valid_score = 0
    for ne in n_estimator:
        for m_d in max_depths:
            model = RandomForestRegressor(n_estimators=ne,max_depth=m_d)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            #print(valid_score)
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                pred = model.predict(test_feature)
                best_n = ne
                best_d = m_d
    model = RandomForestRegressor(n_estimators=best_n,max_depth=best_d).fit(np.array(train_feature),np.array(train_label))
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots_rf.append(permut_importance.importances)
    
    print(test_score)
    prediction_rf.append(pred)
    prediction_true_rf.append(test_label)
    test_score_all_rf.append(test_score)
    print('best n_estimator is',best_n)
    print('best depth is',best_d)
    feature_importance_permute_rf.append(permut_importance.importances_mean)
    feature_importance_impurity_rf.append(model.feature_importances_)



prediction_rf_all = []
for l in prediction_rf:
    for v in l:
        prediction_rf_all.append(v)



prediction_true_rf_all = []
for l in prediction_true_rf:
    for v in l:
        prediction_true_rf_all.append(v)



sns.scatterplot(prediction_true_rf_all,prediction_rf_all)
sns.lineplot(np.arange(-3,2.),np.arange(-3,2.),color='r')
plt.xlabel('Measured logRCF',fontsize=14)
plt.ylabel('Predicted logRCF',fontsize=14)


train_split_index,test_split_index = Kfold(len(features_z),5)

splits = 5

test_score_all_mlp = []
prediction_mlp = []
prediction_true_mlp = []
importance_all_dots_nn = []
fig = []
feature_importance_permute_nn = []
for k in range(splits):
    
    print('batch is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    
    train_feature = [features_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [features_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [features_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    
    n_estimator = [(64,32),(128,64),(32,16),(128,64,32)]
    
    best_valid_score = 0
    for ne in n_estimator:
        model = MLPRegressor(hidden_layer_sizes=ne,learning_rate_init=0.0005,batch_size=64,max_iter=1000,random_state=1)
        model.fit(np.array(train_feature),np.array(train_label))
        valid_score = model.score(valid_feature,np.array(valid_label))
        if valid_score>best_valid_score:
            best_valid_score = valid_score
            test_score = model.score(test_feature,np.array(test_label))
            #df = model.predict_proba(test_feature)
            pred = model.predict(test_feature)
            best_n = ne
    
    
    model = MLPRegressor(hidden_layer_sizes=best_n,learning_rate_init=0.0005,batch_size=64,max_iter=1000,random_state=0).fit(np.array(train_feature),np.array(train_label))
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots_nn.append(permut_importance.importances)
    fig.append(plot_partial_dependence(model, np.array(train_feature),[0,1,2,3],feature_names=['E','V','OM','flipid'],method = 'brute',kind = 'both',percentiles=(0,1)))
    
    print(test_score)
    prediction_mlp.append(pred)
    prediction_true_mlp.append(test_label)
    test_score_all_mlp.append(test_score)
    feature_importance_permute_nn.append(permut_importance.importances_mean)

    print('best n_estimator is',best_n)



figs,new_ax = plt.subplots(2,2,figsize=(15,15))

for i in range(170):
    new_ax[0][0].plot(fig[2].pd_results[0]['values'][0],fig[2].pd_results[0]['individual'][0][i],color='salmon',alpha=0.05)
new_ax[0][0].plot(fig[2].pd_results[0]['values'][0],fig[2].pd_results[0]['average'][0],color='red',label='average')
new_ax[0][0].legend(loc='upper right')
new_ax[0][0].set_ylabel('Partial Dependence',fontsize=16)
new_ax[0][0].set_xlabel('E',fontsize=16)

for i in range(170):
    new_ax[0][1].plot(fig[2].pd_results[1]['values'][0],fig[2].pd_results[1]['individual'][0][i],color='yellowgreen',alpha=0.05)
new_ax[0][1].plot(fig[2].pd_results[1]['values'][0],fig[2].pd_results[1]['average'][0],color='green',label='average')
new_ax[0][1].legend(loc='upper right')
new_ax[0][1].set_ylabel('Partial Dependence',fontsize=16)
new_ax[0][1].set_xlabel('V',fontsize=16)

for i in range(170):
    new_ax[1][0].plot(fig[2].pd_results[2]['values'][0],fig[2].pd_results[2]['individual'][0][i],color='khaki',alpha=0.05)
new_ax[1][0].plot(fig[2].pd_results[2]['values'][0],fig[2].pd_results[2]['average'][0],color='gold',label='average')
new_ax[1][0].legend(loc='upper right')
new_ax[1][0].set_ylabel('Partial Dependence',fontsize=16)
new_ax[1][0].set_xlabel('$f_{OM}$',fontsize=16)

for i in range(170):
    new_ax[1][1].plot(fig[2].pd_results[3]['values'][0],fig[2].pd_results[3]['individual'][0][i],color='lightskyblue',alpha=0.05)
new_ax[1][1].plot(fig[2].pd_results[3]['values'][0],fig[2].pd_results[3]['average'][0],color='deepskyblue',label='average')
new_ax[1][1].legend(loc='upper right')
new_ax[1][1].set_ylabel('Partial Dependence',fontsize=16)
new_ax[1][1].set_xlabel('$f_{lipid}$',fontsize=16)



prediction_mlp_all = []
for l in prediction_mlp:
    for v in l:
        prediction_mlp_all.append(v)


prediction_true_mlp_all = []
for l in prediction_true_mlp:
    for v in l:
        prediction_true_mlp_all.append(v)



violin_dots = np.concatenate(np.array(importance_all_dots_nn),1).transpose()
df = pd.DataFrame(data=violin_dots,columns=['E','V','fom','flipid','S','A','B','MW','NOCount','NHOHCount','NumHAcceptors','NumHDonors','Num_Aro_rings','tpsa','Num_sat_rings'])

sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylabel('Relative Importance',fontsize=14)
plt.tick_params(labelsize=14)



sns.scatterplot(prediction_true_mlp_all,prediction_mlp_all)
sns.lineplot(np.arange(-3.7,2.),np.arange(-3.7,2.),color='r')
plt.xlabel('Measured logRCF',fontsize=14)
plt.ylabel('Predicted logRCF',fontsize=14)





from sklearn.svm import SVR
#np.random.shuffle(total_id)
train_split_index,test_split_index = Kfold(len(features_z),5)

splits = 5
prediction_svc = []
prediction_true_svc = []
test_score_all_svc = []
for k in range(splits):
    
    print('batch is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]
    
    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]
    
    train_feature = [features_z[i] for i in train_id]
    train_label = [RCF_soil[i] for i in train_id]
    
    valid_feature = [features_z[i] for i in valid_id]
    valid_label = [RCF_soil[i] for i in valid_id]
    
    test_feature = [features_z[i] for i in test_id]
    test_label = [RCF_soil[i] for i in test_id]
    
    G_pool = [0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100]

    C_pool = [0.0001,0.001, 0.01, 0.1, 1, 10,25,50,100,1000]
    
    best_valid_score = float('-inf')
    for c in C_pool:
        for g in G_pool:
            model = SVR(kernel='rbf',C=c,gamma=g)
            model.fit(train_feature,train_label)
            valid_score = model.score(valid_feature,valid_label)
            #print('valid score is',valid_score)
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                best_n = c
                best_d = g
                pred = model.predict(test_feature)
    print('test score is',test_score)
    test_score_all_svc.append(test_score) 
    prediction_svc.append(pred)
    prediction_true_svc.append(test_label)
    print('best c is',best_n)
    print('best g is',best_d)
    #print('feature importance',model.feature_importances_)



prediction_svc_all = []
for l in prediction_svc:
    for v in l:
        prediction_svc_all.append(v)



prediction_true_svc_all = []
for l in prediction_true_svc:
    for v in l:
        prediction_true_svc_all.append(v)




sns.scatterplot(prediction_true_svc_all,prediction_svc_all)
sns.lineplot(np.arange(-3,2.),np.arange(-3,2.),color='r')
plt.xlabel('Measured logRCF',fontsize=14)
plt.ylabel('Predicted logRCF',fontsize=14)
