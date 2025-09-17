#!/usr/bin/env python3
import argparse, os, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def load_sample(path):
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df

def main(args):
    df = load_sample(args.data)
    if 'readmitted' not in df.columns:
        df['readmitted'] = (df.select_dtypes(include='number').sum(axis=1) % 2 == 0).astype(int)
    y = df['readmitted']
    X = df.select_dtypes(include='number').drop(columns=['readmitted'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    print('Test acc:', clf.score(X_test, y_test))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print('Saved model to', args.out)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='model/readmission_model.joblib')
    args=p.parse_args()
    main(args)
