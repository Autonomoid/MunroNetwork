#! /usr/bin/env python
import sys
import random

# Write data to file
out = open('sim_data.csv', 'w')
data = []
count = int(sys.argv[1])

prefixes = ["NC", "NG", "NH", "NJ", "NM", "NN", "NO"]

for n in range(1, count):
    prefix = prefixes[random.randint(0,len(prefixes)-1)]
    x = random.randint(1, 999)
    y = random.randint(1, 999)
    alt = random.randint(915, 1350)
    name = "FAKE_DATA"
    lat = 0.0
    lon = 0.0
    data += [str(n) + "," + prefix + "," + str(x) + "," + str(y) + "," + str(alt) + ","
             + name + "," + str(lat) + "," + str(lon) + "\n"]

for entry in data:
    out.write(entry)

print "DONE!"
