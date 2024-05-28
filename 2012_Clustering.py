# %%
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%%
# Load the CSV file
df_raw = pd.read_csv('df_clean.csv')

#%%
# Drop irrelevant columns
columns_to_drop = [
    'year', 'pct', 'ser_num', 'datestop', 'timestop', 'recstat', 'inout', 'trhsloc', 'perobs',
    'perstop', 'typeofid', 'othpers', 'arstoffn', 'sumoffen', 'compyear', 'comppct',
    'offunif', 'officrid', 'frisked', 'searched', 'contrabn', 'adtlrept', 'pistol', 'riflshot',
    'asltweap', 'knifcuti', 'machgun', 'othrweap', 'pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep',
    'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other', 'radio', 'ac_rept', 'ac_inves',
    'rf_vcrim', 'rf_othsw', 'ac_proxm', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout',
    'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc', 'cs_furtv', 'rf_rfcmp', 'ac_cgdir',
    'rf_verbl', 'cs_vcrim', 'cs_bulge', 'cs_other', 'ac_incid', 'ac_time', 'rf_knowl', 'ac_stsnd',
    'ac_other', 'sb_hdobj', 'sb_outln', 'sb_admis', 'sb_other', 'repcmd', 'revcmd', 'rf_furt',
    'rf_bulg', 'offverb', 'offshld', 'forceuse', 'sex', 'dob', 'height', 'weight',
    'haircolr', 'eyecolor', 'build', 'othfeatr', 'addrtyp', 'rescode', 'premtype', 'premname',
    'addrnum', 'stname', 'stinter', 'crossst', 'aptnum', 'state', 'zip', 'addrpct', 'sector',
    'beat', 'post', 'dettypcm', 'linecm', 'detailcm'
]
df_raw = df_raw.drop(columns=columns_to_drop)

#%%
df_raw.head()

# Cluster the location for Felony in MANHATTAN
#%%
df_raw['crimsusp'].unique()

#%%
df_raw['city'].unique()

#%%
# Filter data 
df_fellony = df_raw[(df_raw['crimsusp'] == 'FELONY') & (df_raw['city'] == 'MANHATTAN')]
df_fellony[['xcoord', 'ycoord']] = df_fellony[['xcoord', 'ycoord']].apply(pd.to_numeric, errors='coerce')
df_fellony = df_fellony.dropna()

#%%
# Drop columns from filtered data 
df_fellony = df_fellony.drop(columns = ['crimsusp','explnstp', 'arstmade', 'city', 'sumissue','age', 'race'])

#%%
# Elbow method for optimal number of clusters
wcss = []
for index in range(1, 11):
    kmeans = KMeans(n_clusters=index)
    kmeans.fit(df_fellony)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal Number of Clusters for Fellony in MANHATTAN Dataset')
plt.grid(True)
plt.show()

#%%
# Visualization of clusters with centroids
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_fellony)
clusters = kmeans.predict(df_fellony)
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 8))
plt.scatter(df_fellony.iloc[:, 0], df_fellony.iloc[:, 1], c=clusters, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', s=100, color='red', label='Centroids')
plt.xlabel('x Coordinate')
plt.ylabel('y Coordinate')
plt.title('Clustering for Fellony in MANHATTAN')
plt.show()

