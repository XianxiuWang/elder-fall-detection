import sys, os
sys.path.insert(0, r'D:\Users\wangxianxiu\.openclaw\workspace\huling_model')
import pandas as pd
from config import STATE_NAMES

base = r'D:\Users\wangxianxiu\.openclaw\workspace\huling_model\data'

# URFD 旧数据
old = pd.read_csv(os.path.join(base, 'urfd_features_20260507_205529.csv'))
print('=== URFD ===')
print(f'Shape: {old.shape}')
print(f'Cols (last 5): {list(old.columns[-5:])}')
print(f'Labels: {old["label"].value_counts().sort_index().to_dict()}')
if 'label_name' in old.columns:
    print(f'Label names: {old["label_name"].value_counts().to_dict()}')
if 'split' in old.columns:
    print(f'Splits: {old["split"].value_counts().to_dict()}')

print()

# Main data 新数据
new = pd.read_csv(os.path.join(base, 'main_data_features_20260509_162752.csv'))
print('=== Main Data ===')
print(f'Shape: {new.shape}')
print(f'Cols (last 5): {list(new.columns[-5:])}')
print(f'Labels: {new["label"].value_counts().sort_index().to_dict()}')
if 'label_name' in new.columns:
    print(f'Label names: {new["label_name"].value_counts().to_dict()}')
if 'split' in new.columns:
    print(f'Splits: {new["split"].value_counts().to_dict()}')

print()
print(f'STATE_NAMES: {STATE_NAMES}')
print(f'Col name match: old={list(old.columns[:5])[:3]}...')
print(f'Col name match: new={list(new.columns[:5])[:3]}...')
