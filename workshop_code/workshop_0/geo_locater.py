import json
from urllib.request import urlopen

# Split location to two variables
def splitLoc(loc):
    split = loc.split(',')
    lat = split[0]
    lon = split[1]
    return lat, lon

# Make url request to retrieve geo location 
url = 'http://ipinfo.io/json'
info = json.loads(urlopen(url).read().decode('utf-8'))

# Create titles
titles = ['City',  'Province' , 'Country', 'Organization',  'IP Address', 'Latitude', 'Longitude']

# Create labels
labels = []
labels.append(info['city'])
labels.append(info['region'])
labels.append(info['country'])
labels.append(info['org'])
labels.append(info['ip'])

# Split lat and long
lat, lon = splitLoc(info['loc'])
labels.append(lat)
labels.append(lon)  

# Print everything out
for x in range(len(titles)):
    if titles[x] == 'Latitude' or titles[x] == 'Longitude':
        print(titles[x] + ': ' + labels[x] + ' degrees')
    else:
        print(titles[x] + ': ' + labels[x])
