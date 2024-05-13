import pandas as pd

df = pd.read_json('dataset_en_train.json')

print(df.columns.values) # id, text, category, annotations, spacy tokens
print(df.spacy_tokens[1]) # category is CRITICAL vs CONSPIRACY

# annotation needs to be broken up within itself
# thats probably why its json file
# its basically assigning more labels to specific tokens
# anyway this is useful for now for the binary classification task


