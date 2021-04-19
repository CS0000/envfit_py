import pandas as pd
from sklearn.datasets import load_iris
from skbio.stats.ordination import pcoa
from math import sqrt
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np



# pcoa: get corrdination for each sample
def pcoa_coord(data,metric='euclidean'):

    dm = pd.DataFrame(squareform(pdist(data, metric=metric)),
                     index=data.index.tolist(),
                     columns=data.index.tolist())
    dm_pcoa = pcoa(dm, number_of_dimensions=2)
    coord = pd.DataFrame(dm_pcoa.samples.values, index=dm.index.tolist(), columns=['X1', 'X2'])
    # print(coord.head())
    return coord


def permutation_ntimes(perm_n=999,sample_n=150):
    perm_ind = dict()
    ori_ind = range(sample_n)
    for i in range(perm_n):
        perm_ind[i] = np.random.permutation(ori_ind)
    return perm_ind


def vectorfit(vector_env,coord,perm_n=99):
    vecs = vector_env.columns.tolist()
    reg = LinearRegression()
    vec_fit = dict()
    for vec in vecs:
        reg = reg.fit(coord, vector_env[vec])
        head = reg.coef_
        tmp = sqrt(head[0] * head[0] + head[1] * head[1])
        arrow = head / tmp
        r = r2_score(vector_env[vec], reg.predict(coord))
        vec_fit[vec] = dict(r=r, X1=arrow[0], X2=arrow[1])

        perm_ind = permutation_ntimes(perm_n=perm_n, sample_n=vector_env.shape[0])
        perm_r = dict()
        for perm in perm_ind.keys():
            vec_shuffle = vector_env.iloc[[int(i) for i in perm_ind[perm]], :]
            vec_shuffle = vec_shuffle[vec]
            reg = reg.fit(coord, vec_shuffle)
            r_shuffle = r2_score(vec_shuffle, reg.predict(coord))
            perm_r[perm] = (r_shuffle >= r)
        pval = (sum([v for k, v in perm_r.items()]) + 1) / (len(perm_ind) + 1)
        vec_fit[vec]['pval'] = pval
    vec_fit_df = pd.DataFrame.from_dict(vec_fit).T
    # print(vec_fit_df)
    return vec_fit_df


def SS(fac_temp,dim='X1',fac_name='type'):
    fac_temp_gb = fac_temp[[dim,fac_name]].groupby(fac_name)
    mean = fac_temp_gb.mean()
    var = fac_temp_gb.var()
    n = fac_temp_gb.count()
    fac_summary = pd.concat([n,mean,var],axis=1)
    fac_summary.columns = ['n','mean','var']
    return fac_summary


def fac_fit(fac_temp,fac_name='type'):
    # fac = pd.DataFrame(env[fac_list])
    # fac_temp = pd.concat([fac_df, coord], axis=1)
    SS_X1 = SS(fac_temp,dim='X1',fac_name=fac_name)
    SST_X1 = dict(nt=SS_X1['n'].sum(),
                  mean=fac_temp['X1'].mean(),
                  var=fac_temp['X1'].var())
    SS_X2 = SS(fac_temp,dim='X2',fac_name=fac_name)
    SST_X2 = dict(nt=SS_X2['n'].sum(),
                  mean=fac_temp['X2'].mean(),
                  var=fac_temp['X2'].var())

    SSW = sum(SS_X1['n'] * SS_X1['var']) + sum(SS_X2['n'] * SS_X2['var'])
    SSB = sum(SS_X1['n']*((SS_X1['mean']-SST_X1['mean'])**2)) + sum(SS_X2['n']*((SS_X2['mean']-SST_X2['mean'])**2))
    return 1-SSW/(SSB+SSW)

def factorfit(fac_df_oh,coord,perm_n=999):

    perm_ind = permutation_ntimes(perm_n=perm_n, sample_n=fac_df_oh.shape[0])
    fac_list = fac_df_oh.columns.tolist()
    fac_fit_result = {}
    for fac in fac_list:
        perm_r = dict()
        fac_df = pd.DataFrame(fac_df_oh[fac])
        fac_temp = pd.concat([coord, fac_df], axis=1)
        r = fac_fit(fac_temp, fac_name=fac)

        cen = dict()
        ff = fac_df[fac].unique()
        for f in ff:
            fac_ind = fac_df.loc[fac_df[fac] == f,].index.tolist()
            cen[f] = {'X1': coord.loc[fac_ind, 'X1'].sum() / len(fac_ind),
                      'X2': coord.loc[fac_ind, 'X2'].sum() / len(fac_ind)}

        for perm in perm_ind.keys():
            fac_shuffle = fac_df.iloc[[int(i) for i in perm_ind[perm]], :]
            fac_temp = coord.copy()
            fac_temp[fac] = fac_shuffle[fac].tolist()

            r_shuffle = fac_fit(fac_temp, fac_name=fac)
            perm_r[perm] = (r_shuffle >= r)
        pval = (sum([v for k, v in perm_r.items()]) + 1) / (len(perm_ind) + 1)
        fac_fit_result[fac] = dict(pval=pval, r=r, cen=cen)

    fac_other = []
    fac_cen = []
    for k,v in fac_fit_result.items():
        cen_key = 'cen'
        other_key = [i for i in v.keys() if i != cen_key]
        fac_other.append(pd.DataFrame.from_dict({k:{ok:v[ok] for ok in other_key}}).T)
        fac_cen.append(pd.DataFrame.from_dict({str(k)+'_'+str(i): v[cen_key][i] for i in v[cen_key].keys()}))
        # print(fac_cen)
    fac_other_df = pd.concat(fac_other,axis=0)
    fac_cen_df = pd.concat(fac_cen,axis=1).T
    return fac_other_df, fac_cen_df


if __name__ == '__main__':

    ### use iris dataset for demo, prepare vector and factor dataframe
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    # print(data.head())
    iris_type = [dict(zip([0,1,2],iris.target_names))[i] for i in list(iris.target)]
    env = data.copy()
    env['type'] = iris_type
    vector_env = env.iloc[:,0:4]
    # print(vector_env.head())

    fac_df = pd.DataFrame(env['type'])
    fac_df['type'] = pd.Categorical(fac_df['type'])
    fac_df_oh = pd.get_dummies(fac_df['type'], prefix='type')
    fac_df_oh = fac_df_oh.replace(1, 'T').replace(0, 'F')
    # print(fac_df_oh.head())


    ### pcoa
    coord = pcoa_coord(data)

    ### vectorfit
    vec_fit_df = vectorfit(vector_env, coord, perm_n=999)
    print(vec_fit_df)

    ### factorfit
    fac_other_df, fac_cen_df = factorfit(fac_df_oh, coord, perm_n=999)
    print(fac_cen_df)
    print(fac_other_df)
    # test for branch
