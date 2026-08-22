import urllib.request, re
req = urllib.request.Request('https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0004497', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
links = re.findall(r'href=["\']([^"\']+)["\'][^>]*>S\d+\s+Data[^<]*<', html, re.I)
print('S Data links:', links)
