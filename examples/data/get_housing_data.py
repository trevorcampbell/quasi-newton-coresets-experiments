import numpy as np
import sys
import pandas as pd
import zipfile
from download import download_file

print("----UK Housing prices data----")

# Get the geocoding data
print("downloading geocoding data")
download_file("http://download.geonames.org/export/zip/GB_full.csv.zip", "GB_full.zip")

print("unzipping geocoding data to GB_full.txt")
# Unzip the geocoding data
with zipfile.ZipFile("GB_full.zip", "r") as zf:
    zf.extractall(".")

# get the housing price data
print("downloading the housing prices data")
download_file("http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv", "pp-2018.csv")

#load geocoding data and preprocess into sorted format
geodata_fields = ['country code', 'postal_code', 'place_name',
               'state_name', 'state_code', 'county_name', 'county_code',
               'community_name', 'community_code',
               'latitude', 'longitude', 'accuracy']

print('loading full GB postcodes database')
geodata = pd.read_csv('GB_full.txt', sep='\t', header=0, names=geodata_fields, dtype={'postal_code': str})

post_code_to_int = lambda x : int(''.join([str(ord(a.lower())-97) if ord(a.lower())-97 >= 0 else str(ord(a.lower())-48)  for a in x.replace(' ', '')]))

print('extracting postcode, lat, lon')
geodata = geodata[['postal_code', 'latitude', 'longitude']]
print('converting post codes to integers')
geodata['postal_code'] = geodata['postal_code'].apply(post_code_to_int)
print('sorting by integer tags')
geodata.sort_values(by='postal_code', inplace=True)
print('converting to np array')
geodata = np.array(geodata)

print('loading price paid data')
f = open('pp-2018.csv', 'r')
lines = f.readlines()
f.close()

#[integer post code tag, price]
data = np.zeros((len(lines), 2))
print('extracting post code and price')
for i in range(data.shape[0]):
  if i % 1000 == 0:
      sys.stdout.write(f"processing entry {i+1}/{data.shape[0]}                 \r")
      sys.stdout.flush()
  tokens = [s.strip(' "') for s in lines[i].split(',')]
  price = int(tokens[1])
  try:
    postcode = post_code_to_int(tokens[3])
  except:
    postcode = -1
  data[i, :] = np.array([postcode, price])
sys.stdout.write("\n")
sys.stdout.flush()

print('found ' + str(data.shape[0]) + ' entries')

print('removing bad entries')
data = data[data[:, 0] >= 0, :]

print(str(data.shape[0]) + ' entries remaining')

#sort by integer post tag
print('sorting by integer post tag')
data = data[data[:,0].argsort(), :]

#now iterate through geodata and data, incrementing post tags on each as needed
data_lat_lon = np.zeros((data.shape[0], 3))
geo_idx = 0
print('converting post code to lat lon')
for i in range(data.shape[0]):
  if i % 1000 == 0:
      sys.stdout.write(f"processing entry {i+1}/{data.shape[0]}                 \r")
      sys.stdout.flush()
  while(geodata[geo_idx,0] < data[i,0]):
    geo_idx += 1
  if geodata[geo_idx,0] != data[i, 0]:
    #geodata doesn't have this post code, give up
    data_lat_lon[i, 0] = np.nan
  else:
    data_lat_lon[i, 0] = geodata[geo_idx, 1]
    data_lat_lon[i, 1] = geodata[geo_idx, 2]
    data_lat_lon[i, 2] = data[i, 1]
sys.stdout.write("\n")
sys.stdout.flush()

print('filtering bad entries')
data_lat_lon = data_lat_lon[np.logical_not(np.isnan(data_lat_lon[:, 0])), :]
print(str(data_lat_lon.shape[0]) + ' entries remaining')
np.save('prices2018.npy', data_lat_lon)
print('done')
