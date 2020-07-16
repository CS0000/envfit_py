# implement basic function of vegan::envfit using python

vegan::envfit R source code: https://github.com/vegandevs/vegan/blob/5f952c9f0f30a1851f99e4887fcfde5c7a8b65a1/R/envfit.default.R

R code: `envfit(ordination,env,permutations = n)`
ordination: ordination configuration
env: environmental variable(s)
permutations:numbers of permutations for assessing significance of vectors or factors.
(from http://www.sortie-nd.org/lme/R%20Packages/vegan.pdf)

### python implement in envfit2py.py
use iris dataset for demo
```
### prepare vector and factor dataframe
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
```

### R output:
```
***VECTORS
                   X1       X2     r2   Pr(>r)
Sepal.Length  0.48219  0.87607 0.9579 0.009901 **
Sepal.Width  -0.11499  0.99337 0.8400 0.009901 **
Petal.Length  0.98013 -0.19836 0.9981 0.009901 **
Petal.Width   0.97852 -0.20615 0.9366 0.009901 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Permutation: free
Number of permutations: 100
***FACTORS:
Centroids:
                 X1      X2
setosaF      1.3212 -0.0954
setosaT     -2.6424  0.1909
versicolorF -0.2666  0.1228
versicolorT  0.5332 -0.2455
virginicaF  -1.0546 -0.0273
virginicaT   2.1092  0.0547
Goodness of fit:
               r2   Pr(>r)
setosa     0.7902 0.009901 **
versicolor 0.0388 0.009901 **
virginica  0.5012 0.009901 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Permutation: free
Number of permutations: 100
```

### python output:
```
vec_fit_df
                     X1        X2      pval       r
petal length (cm)  0.979589 -0.201010  0.001  0.998117
petal width (cm)   0.979009 -0.203817  0.001  0.936413
sepal length (cm)  0.482423  0.875938  0.001  0.957867
sepal width (cm)  -0.112032  0.993705  0.001  0.838197

fac_cen_df
                     X1        X2
type_setosa_F      1.320420 -0.095260
type_setosa_T     -2.640841  0.190520
type_virginica_F  -1.054424 -0.026755
type_virginica_T   2.108848  0.053510
type_versicolor_F -0.265996  0.122015
type_versicolor_T  0.531993 -0.244030


fac_other_df
                 pval         r
type_setosa      0.001  0.788100
type_virginica   0.001  0.498654
type_versicolor  0.013  0.038214
```

### note
* the output of python and R have "tiny" difference. That's due to the way python and R implement linear regression.
*
