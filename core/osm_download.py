import overpy
from PIL import Image
api = overpy.Overpass()



query = """
[out:json][timeout:25];
// gather results
(
  // query part for: “building and type!=node”

  
  way["building"]({{bbox}});

);
// print results
out center meta;
"""

print("hello world")
