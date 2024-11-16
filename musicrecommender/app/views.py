from django.shortcuts import render
import pandas as pd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import difflib
import copy
import warnings
warnings.filterwarnings("ignore")
plotly.offline.init_notebook_mode(connected=True)
data=pd.read_csv(r"C:\Users\vibho\Documents\Data Science\Project\genres_v2.csv")
cols=list(data.columns[11:])
del cols[7]
df=copy.deepcopy(data)
df.drop(columns=cols,inplace=True)
data.drop('Unnamed: 0',axis=1,inplace=True)
data=data.dropna(subset=['song_name'])
df=data[data.columns[:11]]
df['genre']=data['genre']
df['time_siganture']=data['time_signature']
df['duration_ms']=data['duration_ms']
df['song_name']=data['song_name']
x=df[df.drop(columns=['song_name','genre']).columns].values
scaler=StandardScaler().fit(x)
X_scaled=scaler.transform(x)
df[df.drop(columns=['song_name','genre']).columns]=X_scaled
v=[]
# Create your views here.
def index(request):
    return render(request,'index.html')  

def make_matrix_correlation(data,song,number):
    v=[]
    df=pd.DataFrame()
    data.drop_duplicates(inplace=True)
    songs=data['song_name'].values
    best = difflib.get_close_matches(song,songs,1)[0]
    genre=data[data['song_name']==best]['genre'].values[0]
    df=data[data['genre']==genre]
    x=df[df['song_name']==best].drop(columns=['genre','song_name']).values
    if len(x)>1:
        x=x[1]
    song_names=df['song_name'].values
    df.drop(columns=['genre','song_name'],inplace=True)
    df=df.fillna(df.mean())
    p=[]
    count=0
    for i in df.values:
        p.append([distance.correlation(x,i),count])
        count+=1
    p.sort()
    for i in range(1,number+1):
        v.append([song_names[p[i][1]]])
    return v
        
def predict(request):
    song=request.GET['song']
    number=request.GET['number']
    number = int(number)
    v = make_matrix_correlation(df, song, number)
    return render(request, 'predict.html', {'prediction': v})