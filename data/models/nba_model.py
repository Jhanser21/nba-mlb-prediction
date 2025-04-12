import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_nba_model():
    df = pd.read_csv('data/nba/games.csv')
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    features = ['PTS', 'AST', 'REB', 'FG_PCT']
    X = df[features]
    y = df['WIN']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
