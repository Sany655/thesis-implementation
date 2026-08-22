import urllib.request, re
req = urllib.request.Request('https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0004497', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.I):
    href = m.group(1)
    text = m.group(2)
    if 'type=supplementary' in href or 'xls' in href.lower() or 'csv' in href.lower():
        print(f'Href: {href} | Text: {text}')
