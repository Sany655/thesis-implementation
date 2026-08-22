import urllib.request, re
try:
    html = urllib.request.urlopen('https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4794248/').read().decode('utf-8')
    links = re.findall(r'href=["\']([^"\']+(?:bin)[^"\']+)["\']', html, re.I)
    print('Found bin links:', set(links))
except Exception as e:
    print(e)
