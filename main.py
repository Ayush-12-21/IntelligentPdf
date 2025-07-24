import os
import json
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Feature extraction for a candidate line
# Assumes each span dict has: 'text', 'page', 'font_size', 'font'
def extract_features(span):
    text = span.get("text", "")
    fs = span.get("font_size", 0)
    # Boldness: True if font name contains 'Bold'
    fonts = span.get("font", [])
    is_bold = int(any("Bold" in f for f in fonts))
    # Numbered: starts with digit or digit-dot
    is_numbered = int(bool(re.match(r'^\d+(?:\.\d+)*\b', text)))
    # Colon at end
    is_colon = int(text.endswith(':'))
    # Word count
    words = text.split()
    word_count = len(words)
    # Stopword ratio
    stops = {"and","the","of","in","to","for","with","on","by","at","from"}
    stop_ratio = sum(1 for w in words if w.lower() in stops) / max(1, word_count)
    # Starts uppercase/digit
    starts_upper = int(bool(re.match(r'^[A-Z0-9]', text)))
    # Ends punctuation
    ends_punct = int(text.endswith(('.', '?', '!')))
    return [fs, is_bold, is_numbered, is_colon,
            word_count, stop_ratio, starts_upper, ends_punct]

if __name__ == "__main__":
    JSON_FOLDER = 'output'
    LABEL_CSV = 'labels.csv'  # Create this with columns: file,page,text,label

    # Read labeled data
    df_labels = pd.read_csv(LABEL_CSV)

    feature_rows = []
    label_rows = []
    # Iterate labels
    for _, row in df_labels.iterrows():
        pdf_file = row['file']
        page_num = int(row['page'])
        text     = row['text']
        label    = int(row['label'])
        json_path = os.path.join(JSON_FOLDER, os.path.basename(pdf_file).replace('.pdf','.json'))
        if not os.path.exists(json_path):
            continue
        data = json.load(open(json_path))
        for span in data.get('outline', []):
            if span.get('page') == page_num and span.get('text') == text:
                # ensure 'font' key present
                span.setdefault('font', [])
                feat = extract_features(span)
                feature_rows.append(feat)
                label_rows.append(label)
                break

    # Build DataFrame
    cols = ['font_size','is_bold','is_numbered','is_colon',
            'word_count','stop_ratio','starts_upper','ends_punct']
    df = pd.DataFrame(feature_rows, columns=cols)
    df['label'] = label_rows

    # Train/test split
    X = df[cols]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    with open('heading_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model trained and saved to heading_model.pkl")
