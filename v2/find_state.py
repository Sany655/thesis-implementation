import re, json
html = open(r'C:\Users\Sany\.gemini\antigravity-ide\brain\90153d1e-8d8f-4b1b-8696-c25a44dd0f93\.system_generated\steps\98\content.md', encoding='utf-8').read()
match = re.search(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});', html)
if match:
    state = json.loads(match.group(1))
    files = state.get('dataset', {}).get('dataset', {}).get('data', {}).get('files', [])
    for f in files:
        print(f.get('filename'), f.get('content_details', {}).get('download_url'))
else:
    print("No initial state found")
