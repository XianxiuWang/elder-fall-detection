import os, json, sys

path = r'D:\Users\wangxianxiu\.openclaw\workspace\项目文档'
output = {}
for f in sorted(os.listdir(path)):
    if '.tmp' not in f:
        continue
    fp = os.path.join(path, f)
    with open(fp, 'r', encoding='utf-8') as fh:
        content = fh.read()
    # Strip .tmp from name
    clean_name = f.replace('.tmp.md', '.md')
    output[clean_name] = content

with open(os.path.join(path, '_all_content.json'), 'w', encoding='utf-8') as fw:
    json.dump(output, fw, ensure_ascii=False, indent=2)
print("Done - wrote _all_content.json")
