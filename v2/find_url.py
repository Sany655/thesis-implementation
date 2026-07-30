import re
html = open(r'C:\Users\Sany\.gemini\antigravity-ide\brain\90153d1e-8d8f-4b1b-8696-c25a44dd0f93\.system_generated\steps\98\content.md', encoding='utf-8').read()
matches = re.findall(r'https://data.mendeley.com/public-api/datasets/jsbmtk8hty/files/[a-zA-Z0-9\-]+/download', html)
if not matches:
    matches = re.findall(r'href="(.*?)"', html)
print(matches[:20])
