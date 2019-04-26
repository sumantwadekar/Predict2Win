import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from flask import Flask,request,render_template,send_file
import pickle
import math
import pycricbuzz
from pycricbuzz import Cricbuzz


c=Cricbuzz()

# df=pd.read_csv("odi.csv")
# df.isna().any(axis=0)

# df1=pd.get_dummies(df['bat_team'])
# df=df.join(df1)
# l1=[7,8,9,12,13]
# for i in range(15,36):
#     l1.append(i)
# #using runs,wickets,overs,striker,non striker as features
# X=df.iloc[:,l1].values
# #using total runs 
# y=df.iloc[:,14].values

# #SPLITTING DATA AS TRAINING AND TESTING
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# #FEATURE SCALING
# sc = StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# rf=pickle.load(open('trained_data_odi.pkl','rb'))


# df_ipl =pd.read_csv("ipl.csv")
# df_ipl1=pd.get_dummies(df_ipl['venue'])
# df_ipl=df_ipl.join(df_ipl1)
# df_ipl2=pd.get_dummies(df_ipl['bat_team'])
# df_ipl=df_ipl.join(df_ipl2)

# listI=[7,8,9,12,13]
# for i in range(15,64):
#     listI.append(i)


# X_ipl=df_ipl.iloc[:,listI].values
# y_ipl=df_ipl.iloc[:,14].values
# sc_ipl=StandardScaler()
# X_train_ipl,X_test_ipl,y_train_ipl,y_test_ipl=train_test_split(X_ipl,y_ipl,test_size=0.25,random_state=0)

# X_train_ipl=sc_ipl.fit_transform(X_train_ipl)
# X_test_ipl=sc_ipl.transform(X_test_ipl)
# rf1=pickle.load(open('trained_data_ipl.pkl','rb'))

# venu_list=df_ipl1.columns
# dict1={}
# for i in range (0,len(venu_list)):
#     dict1.update({venu_list[i]:i})

# bat_team_list=df_ipl2.columns
# dict2={}
# for i in range(0,len(bat_team_list)):
#     dict2.update({bat_team_list[i]:i})



app=Flask(__name__)#root path

@app.route('/index',methods=['GET','POST'])#home page of website
def index():
    if(request.method=='GET'):
        return render_template('index.html')


# @app.route('/teams',methods=['GET','POST'])
# def teams():
#     if(request.method=='GET'):
#         return render_template("teams.html")


@app.route('/odi_prediction',methods=['POST','GET'])
def odi_prediction():
    l1=[]
    for i in range(9):
        l1.append("")
    if(request.method=='GET'):
        return render_template("odi_prediction.html",res=l1)
    else:
        ds=pd.read_csv("odi.csv")
        ds1=ds.loc[(ds['bat_team']==request.form['bat']) & (ds['bowl_team']==request.form['bowl'])]


        l1.clear()
        btt=str(request.form['bat'])
        bwt=str(request.form['bowl'])
        s=request.form['score']
        w=request.form['wickets']
        o=request.form['overs']
        mx=request.form['max']
        mn=request.form['min']

        ds=pd.read_csv("odi.csv")
        ds1=ds.loc[(ds['bat_team']==request.form['bat']) & (ds['bowl_team']==request.form['bowl'])]
        p=ds1.iloc[:,[7,8,9,12,13]].values
        q=ds1.iloc[:,14].values

        p_train,p_test,q_train,q_test=train_test_split(p,q,test_size=0.2,random_state=0)
        sc1= StandardScaler()
        p_train=sc1.fit_transform(p_train)
        p_test=sc1.transform(p_test)

        rf1=RandomForestRegressor(n_estimators=100,max_features=None)

        rf1.fit(p_train,q_train)
        
        pred=rf1.predict(sc1.transform(np.array([[s,w,o,mx,mn]])))
        res=math.floor(pred[0])
        l1.append(btt);l1.append(bwt);l1.append(s);l1.append(w);l1.append(o);l1.append(mx);l1.append(mn);
        l1.append(res-10);l1.append(res+10)
        return render_template("odi_prediction.html",res=l1)


@app.route('/ipl',methods=['POST','GET'])
def ipl():
    l=[]
    for i in range(10):
        l.append("")
    
    if(request.method=='GET'):
        return render_template("ipl.html",res=l)
    else:
        l.clear()

        dp=pd.read_csv("ipl.csv")
        dp1=dp.loc[(dp['bat_team']==request.form['battingteam']) & (dp['bowl_team']==request.form['bowlingteam']) | (dp['venue']==request.form['venue'])]

        a=dp.iloc[:,[7,8,9,12,13]].values
        b=dp.iloc[:,14].values
        

        a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,random_state=0)
        sc2=StandardScaler()
        a_train=sc2.fit_transform(a_train)
        a_test=sc2.transform(a_test)

        rf2=RandomForestRegressor(n_estimators=100,max_features=None)
        rf2.fit(a_train,b_train)

        v=str(request.form['venue']) 
        b=str(request.form['battingteam'])
        bw=str(request.form['bowlingteam'])
        s=request.form['score']
        w=request.form['wickets']
        o=request.form['overs']
        mx=request.form['max']
        mn=request.form['min']

        new_Prediction=rf2.predict(sc2.transform(np.array([[s,w,o,mx,mn]])))

        res=math.floor(new_Prediction[0])
        l.append(v);l.append(b);l.append(bw);l.append(s);l.append(w);
        l.append(o);l.append(mx);l.append(mn);
        l.append(res-5);l.append(res+5);
        
        return render_template('ipl.html',res=l)
        
        
# @app.route('/odiPrediction_india',methods=['GET','POST'])
# def odiPrediction_india():
#     l=[]
#     for i in range(7):
#         l.append("")
#     if(request.method=='GET'):
#         return render_template("odi_prediction_india.html",res=l)
    
#     else:
#         s=request.form['score']
#         w=request.form['wickets']
#         o=request.form['overs']
#         mx=request.form['max']
#         mn=request.form['min']
#         #new_Prediction=rf.predict(np.array([[s,w,o,mx,mn]]))
#         new_Prediction=rf.predict(sc.transform(np.array([[s,w,o,mx,mn,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])))
#         res=new_Prediction[0]
#         l.clear()
#         l.append(s);
#         l.append(w);
#         l.append(o);
#         l.append(mx);
#         l.append(mn);
#         l.append(math.floor(res-10));
#         l.append(math.floor(res+10));
        
#         return render_template("odi_prediction_india.html",res=l)



@app.route('/matches',methods=['POST','GET'])
def matches():
    if(request.method=='GET' or request.method=='POST'):
        matches=c.matches()
        d={}
        for match in matches:
            if((match['mchstate']=='mom' or match['mchstate']=='complete') and match['srs']=='Indian Premier League 2019'):
                mid=match['id']
                team1=match['team1']['name']
                team2=match['team2']['name']
                # l.append(team1);l.append(team2);
                # l.append(match['venue_name']);
                # l.append(match['venue_location'])
                # l.append(match['status']);

                d.update({'past_team1':team1,'past_team2':team2,'past_venue_name':match['venue_name'],
                    'past_venue_location':match['venue_location'],'past_status':match['status']});

            if(match['mchstate']=='preview' and match['srs']=='Indian Premier League 2019'):
                mid=match['id']
                team1=match['team1']['name']
                team2=match['team2']['name']
                d.update({'up_team1':team1,'up_team2':team2,'up_venue_name':match['venue_name'],
                    'up_venue_location':match['venue_location'],'up_status':match['status']});


            if(match['mchstate']=='inprogress' and match['srs']=='Indian Premier League 2019'):
                mid=match['id']
                livescore=c.livescore(mid)
                bat_team=livescore['batting']['team']; 
                bowl_team=livescore['bowling']['team'];
                
                score=livescore['batting']['score'];
                score1=livescore['bowling']['score']
                dict1=score[0]
                
                if(dict1['inning_num']=='2'):    
                    dict2=score1[0]
                    # print("BATTING TEAM : ",bat_team)
                    # print("RUNS :",dict1['runs'])
                    # print("WICKETS : ",dict1['wickets'])
                    # print("OVERS : ",dict1['overs'])
                    # print("NEED ",int(dict2['runs'])-int(dict1['runs'])+1," extra")

                    d.update({'live_team1':bat_team,'live_team2':bowl_team,'live_venue_name':match['venue_name'],
                    'live_venue_location':match['venue_location'],'live_runs':dict1['runs']
                    ,'live_wickets':dict1['wickets'],'live_overs':dict1['overs'],'live_need':int(dict2['runs'])-int(dict1['runs'])+1});

                    
                if(dict1['inning_num']=='1'):
                    d.update({'live_team1':bat_team,'live_team2':bowl_team,'live_venue_name':match['venue_name'],
                    'live_venue_location':match['venue_location'],'live_runs':dict1['runs']
                    ,'live_wickets':dict1['wickets'],'live_overs':dict1['overs']});
        return render_template('matches.html',res=d);



app.run(debug=True)

