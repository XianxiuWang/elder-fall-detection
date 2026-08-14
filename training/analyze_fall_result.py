# analyze_fall_result.py
import json, os
from collections import Counter

d = r'D:\Users\wangxianxiu\clawd'
found = None
for fn in os.listdir(d):
    if fn.endswith('.json') and 'result' in fn.lower():
        if '50 Ways' not in fn and '100 Ways' not in fn:
            found = os.path.join(d, fn)
            break

if not found:
    print('Not found')
    exit(1)

print('File:', os.path.basename(found))
with open(found, 'r', encoding='utf-8') as f:
    data = json.load(f)

preds = data['predictions']
print('Total predictions:', len(preds))

raw_labels = [p['raw'] for p in preds]
filt_labels = [p['filt'] for p in preds]
print('Raw:', dict(Counter(raw_labels)))
print('Filtered:', dict(Counter(filt_labels)))

fall_probs = [p['fall_prob'] for p in preds]
print('Fall prob: min=%.3f max=%.3f' % (min(fall_probs), max(fall_probs)))
chigh = sum(1 for p in fall_probs if p > 0.5)
cvery = sum(1 for p in fall_probs if p > 0.8)
print('Fall prob >0.5: %d/%d' % (chigh, len(fall_probs)))
print('Fall prob >0.8: %d/%d' % (cvery, len(fall_probs)))

sorted_by_fall = sorted(preds, key=lambda x: x['fall_prob'], reverse=True)
print('\nTop 20 by fall_prob:')
print('  %6s  %8s  %8s  %8s  %6s' % ('time', 'raw', 'raw_conf', 'filt', 'fall_p'))
print('  ' + '-' * 48)
for r in sorted_by_fall[:20]:
    print('  %6.1fs  %8s  %8.3f  %8s  %6.3f' % (
        r['time_s'], r['raw'], r['raw_conf'], r['filt'], r['fall_prob']))

# Detect clusters
high_fall = [(p['time_s'], p['fall_prob']) for p in preds if p['fall_prob'] > 0.5]
if high_fall:
    print('\nHigh fall prob clusters (>0.5):')
    cluster = [high_fall[0]]
    for t, prob in high_fall[1:]:
        if t - cluster[-1][0] < 3.0:
            cluster.append((t, prob))
        else:
            if len(cluster) >= 3:
                avg = sum(c[1] for c in cluster) / len(cluster)
                print('  %.1fs - %.1fs  (%d preds, avg_p=%.3f)' % (cluster[0][0], cluster[-1][0], len(cluster), avg))
            cluster = [(t, prob)]
    if len(cluster) >= 3:
        avg = sum(c[1] for c in cluster) / len(cluster)
        print('  %.1fs - %.1fs  (%d preds, avg_p=%.3f)' % (cluster[0][0], cluster[-1][0], len(cluster), avg))
else:
    print('\nNo high fall prob clusters')

# Time series
print('\nTimeline (every 3rd prediction):')
for i in range(0, len(preds), 3):
    p = preds[i]
    bar = '#' * int(p['fall_prob'] * 20)
    print('  %6.1fs  %s  fall=%.3f  %s' % (p['time_s'], p['raw'], p['fall_prob'], bar))
