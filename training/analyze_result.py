import json, os, re
from collections import Counter

# Find the result file
d = r'E:\老人跌倒\training'
files = [f for f in os.listdir(d) if f.startswith('results_') and 'human' in f.lower() or 'ren' in f]
print('Files found:', files)
for fn in files:
    fp = os.path.join(d, fn)
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preds = data['predictions']
    print(f'\n=== {fn} ===')
    print(f'Total predictions: {len(preds)}')
    
    raw_labels = [p['raw'] for p in preds]
    filt_labels = [p['filt'] for p in preds]
    print('Raw distribution:', dict(Counter(raw_labels)))
    print('Filtered distribution:', dict(Counter(filt_labels)))
    
    fall_probs = [p['fall_prob'] for p in preds]
    print(f'Fall prob range: {min(fall_probs):.3f} ~ {max(fall_probs):.3f}')
    print(f'Fall prob > 0.5: {sum(1 for p in fall_probs if p > 0.5)}/{len(fall_probs)}')
    print(f'Fall prob > 0.8: {sum(1 for p in fall_probs if p > 0.8)}/{len(fall_probs)}')
    
    # Top 20 fall prob
    sorted_by_fall = sorted(preds, key=lambda x: x['fall_prob'], reverse=True)
    print(f'\nTop 20 by fall_prob:')
    print(f'{"time":>6s}  {"raw":>8s}  {"raw_conf":>8s}  {"filt":>8s}  {"fall_p":>6s}')
    print('-' * 50)
    for r in sorted_by_fall[:20]:
        print(f'{r["time_s"]:6.1f}s  {r["raw"]:>8s}  {r["raw_conf"]:8.3f}  {r["filt"]:>8s}  {r["fall_prob"]:6.3f}')
    
    # Count Falls in first half vs second half
    first_half = sum(1 for p in preds if p['time_s'] < 48 and p['raw'] == 'Fall')
    second_half = sum(1 for p in preds if p['time_s'] >= 48 and p['raw'] == 'Fall')
    print(f'\nFalls in first 48s: {first_half}, last 48s: {second_half}')
