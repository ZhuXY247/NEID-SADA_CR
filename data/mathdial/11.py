# convert_to_textoir.py
import pandas as pd
import os

def convert(input_tsv, output_csv):
    df = pd.read_csv(input_tsv, sep='\t', header=0,
                     names=['speaker', 'text', 'label'])
    # 过滤 none 等无效标签（和你代码里保持一致）
    ignore = {'none', 'non-english', '0.0', '0', ''}
    df = df[~df['label'].astype(str).str.lower().str.strip().isin(ignore)]
    df = df.dropna(subset=['text', 'label'])
    df[['text', 'label']].to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")

# TalkMoves
# os.makedirs('data/talkmoves', exist_ok=True)
# convert('../../data/talkmoves/train.tsv', 'data/talkmoves/train.csv')
# convert('../../data/talkmoves/dev.tsv',   'data/talkmoves/dev.csv')
# convert('../../data/talkmoves/test.tsv',  'data/talkmoves/test.csv')

# MathDial（同理）
os.makedirs('data/mathdial', exist_ok=True)
convert('../../data/mathdial/train.tsv', 'data/mathdial/train.csv')
convert('../../data/mathdial/dev.tsv',   'data/mathdial/dev.csv')
convert('../../data/mathdial/test.tsv',  'data/mathdial/test.csv')