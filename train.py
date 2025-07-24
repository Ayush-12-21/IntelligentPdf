import os, json, re, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def extract_features(span):
    text = span.get("text","")
    fs   = span.get("font_size",0)
    fonts= span.get("font",[])
    is_bold     = int(any("Bold" in f for f in fonts))
    is_numbered = int(bool(re.match(r'^\d+(?:\.\d+)*\b', text)))
    is_colon    = int(text.endswith(':'))
    words = text.split()
    wc    = len(words)
    stops = {"and","the","of","in","to","for","with","on","by","at","from"}
    stop_ratio  = sum(w.lower() in stops for w in words)/max(1,wc)
    starts_upper= int(bool(re.match(r'^[A-Z0-9]', text)))
    ends_punct  = int(text.endswith(('.', '?', '!')))
    return [fs, is_bold, is_numbered, is_colon, wc, stop_ratio, starts_upper, ends_punct]

if __name__=="__main__":
    JSON_FOLDER = 'output'
    LABEL_CSV   = 'labels.csv'

    # 1) Read your manual labels
    df_labels = pd.read_csv(LABEL_CSV)

    X, y = [], []
    for _, r in df_labels.iterrows():
        json_path = os.path.join(JSON_FOLDER, os.path.basename(r['file']).replace('.pdf','.json'))
        data = json.load(open(json_path))
        for span in data.get('outline',[]):
            if span['page']==int(r['page']) and span['text']==r['text']:
                span.setdefault('font',[])
                X.append(extract_features(span))
                y.append(int(r['label']))
                break

    # 2) Train/Test split
    cols = ['font_size','is_bold','is_numbered','is_colon',
            'word_count','stop_ratio','starts_upper','ends_punct']
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df['label'], test_size=0.2, random_state=42)

    # 3) Train & evaluate
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    # 4) Save the model
    with open('heading_model.pkl','wb') as f:
        pickle.dump(clf, f)
    print("Model saved to heading_model.pkl")
