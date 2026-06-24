import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score , accuracy_score 
from sklearn.compose  import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

## OBTAINING DATA

matches = pd.read_csv("D:\\Programs\\IPL WINNING PROBABILITY\\matches.csv")
deliveries = pd.read_csv("D:\\Programs\\IPL WINNING PROBABILITY\\deliveries.csv")

## ANALYSING AND EVALUATING DATA 

print(matches.head())
print(matches.shape)

print(deliveries.head())
print(deliveries.shape)

## FILTERING THE DATA AS PER OUR NEEDS

TOTAL_RUNS = deliveries.groupby(["match_id" , "inning"]).sum()["total_runs"].reset_index()
TOTAL_RUNS = TOTAL_RUNS[TOTAL_RUNS["inning"]==1]

matches_new = matches.merge(TOTAL_RUNS[["match_id" , "total_runs"]],left_on="id",right_on="match_id")


 ## DELETING AND RENAMING TEAMS

team = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore','Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings','Rajasthan Royals','Delhi Capitals']


matches_new["team1"] = matches_new["team1"].str.replace("Delhi Daredevils","Delhi Capitals")
matches_new["team2"] = matches_new["team2"].str.replace("Delhi Daredevils","Delhi Capitals")
matches_new["team1"] = matches_new["team1"].str.replace("Deccan Chargers","Sunrisers Hyderabad")
matches_new["team2"] = matches_new["team2"].str.replace("Deccan Chargers","Sunrisers Hyderabad")


# #  FILTERING THOSE RECORDS OF THE TEMS WHICH ARE CURRENTLY PLAYING IPL 

matches_new = matches_new[matches_new["team1"].isin(team)]
matches_new = matches_new[matches_new["team2"].isin(team)]


## REMOVING THOSE MATCH RECORDS IN WHICH DL_APPLIED IS 1

matches_new = matches_new[matches_new["dl_applied"]==0]
print(matches_new)

## EXTRACTING REQUIRED COLUMNS AND MERGING THE NEW DATAFRAMES

matches_new = matches_new[["match_id" , "city" , "winner" , "total_runs"]]
MERGED = matches_new.merge(deliveries,on="match_id")

## FILTERING 2ND INNING'S DATA

MERGED = MERGED[MERGED["inning"] == 2]
print(MERGED)

## CREATING RUNS SCORED COLUMN , RUNS REMAINING COLUMN AND BALLS REMAINS COLUMN

MERGED["runs_scored"] = MERGED.groupby("match_id")["total_runs_y"].cumsum()
print(MERGED["runs_scored"])


MERGED["runs_remaining"] = MERGED["total_runs_x"] - MERGED["runs_scored"]
print(MERGED["runs_remaining"])

MERGED["balls_remains"] = 126 - (MERGED['over']*6 + MERGED['ball'])
print(MERGED["balls_remains"])

## CHANGING NAN TO 0 AND CREATING A WICKETS LEFT COLUMN

MERGED["player_dismissed"] = MERGED["player_dismissed"].fillna("0")
MERGED["player_dismissed"] = MERGED["player_dismissed"].apply(lambda x:x if x == "0" else "1") 
MERGED["player_dismissed"] = MERGED["player_dismissed"].astype(int)
wickets_left = MERGED.groupby("match_id")["player_dismissed"].cumsum().values
MERGED["wickets_left"] = 10 - wickets_left
print(MERGED["wickets_left"])

## CREATING  CURRENT RUN RATE COLUMN AND REQUIRED RUN RATE COLUMN
MERGED["CRR"] = (MERGED["runs_scored"]*6)/(120 - MERGED["balls_remains"])
print(MERGED["CRR"])

MERGED["RRR"] = (MERGED["runs_remaining"]*6)/MERGED["balls_remains"]
print(MERGED["RRR"])

## CREATING A FUNVTION WHICH WOULD GIVE 1 IF BATTING TEAM WINS ELSE IT  WOULD GIVE 0
def result(row):
    return 1 if row["batting_team"] == row["winner"] else 0

MERGED["WIN"] = MERGED.apply(result,axis=1)
print(MERGED["WIN"])

## CREATING THE FINAL DATASET 
final = MERGED[["batting_team" , "bowling_team" , "city" , "runs_remaining" , "balls_remains" , "wickets_left" , "total_runs_x" , "CRR" , "RRR" , "WIN"]]
print(final)


## REMOVING NAN VALUES IN CITY AS IT CAN THROW AN ERROR
final.dropna(inplace=True)


## REMOVING VALUES WHERE BALLS IS EQUAL TO 0 AS IT CAN CAUSE RRR AND CRR TO BE INFINITE
final = final[final["balls_remains"] != 0]


## CHECKING OUR DATA
print(final.describe())

## SHUFFLING THE DATASET VALUES
final = final.sample(final.shape[0])


## SPLITTING DATA INTO TRAINING AND TESTING DATA 
x = final.iloc[:,:-1]
y = final.iloc[:,-1]
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=1)

## STANDARDIZING CATEGORICAL DATA USING OneHotEncoder and ColumnTransformer
trc = ColumnTransformer([("trc" , OneHotEncoder(sparse_output=False,drop="first"),["batting_team","bowling_team","city"])],remainder="passthrough")

## TRAINING OUR MODEL
pipe = Pipeline(steps=[
    ("step=1",trc) ,
    ("step=2",LogisticRegression(solver="liblinear"))
])
pipe.fit(x_train,y_train)

## CHECKING THE ACCURACY OF OUR MODEL
y_pred = pipe.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)


## MAKING A FUCTION FOR BETTER PREDICTION
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[["batting_team" , "bowling_team" , "city" , "runs_remaining" , "balls_remains" , "wickets_left" , "total_runs_x" , "CRR" , "RRR"]].dropna()
    temp_df = temp_df[temp_df['balls_remains'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_remaining'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target

temp_df,target = match_progression(MERGED,74,pipe)
print(temp_df)

print(team)
print("/n/n/n",MERGED["city"].unique())


pickle.dump(pipe, open("pipe.pkl", "wb"))