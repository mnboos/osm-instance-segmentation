import overpy
from PIL import Image
from pygeotile.tile import Tile
from pygeotile.point import Point
import requests
import secrets
import shapely.geometry as geometry
import numpy as np
from core.settings import IMAGE_WIDTH

api = overpy.Overpass()


query_template = """
/* TMS {tile} */
[out:json][timeout:50];
( 
  relation["building"]({bbox});
  //node["building"]({bbox});
  way["building"]({bbox});
);
//out geom;
(._;>;);
out body;
"""


def download(tile):
    pass


def tiles_from_bbox(bbox, zoom_level):
    point_min = Point.from_latitude_longitude(latitude=bbox['tl'], longitude=bbox['tr'])
    point_max = Point.from_latitude_longitude(latitude=bbox['bl'], longitude=bbox['br'])
    tile_min = Tile.for_point(point_min, zoom_level)
    tile_max = Tile.for_point(point_max, zoom_level)
    tiles = []
    for x in range(tile_min.tms_x, tile_max.tms_x + 1):
        for y in range(tile_min.tms_y, tile_max.tms_y + 1):
            tiles.append(Tile.from_tms(tms_x=x, tms_y=y, zoom=zoom_level))
    return tiles


def osm_downloader(bbox, zoom_level):
    tiles = tiles_from_bbox(bbox=bbox, zoom_level=zoom_level)
    response = requests.get("https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial?key={key}"
                            .format(key=secrets.BING_KEY))
    data = response.json()

    tile_url_template = data['resourceSets'][0]['resources'][0]['imageUrl']
    subdomain = data['resourceSets'][0]['resources'][0]['imageUrlSubdomains'][0]

    for t in tiles:
        tile_pixels = t.bounds[0].pixels(zoom_level)
        b = []
        # tile_start = t.bounds[0].meters
        # print(t.bounds[0].pixels(17))
        # break
        b.extend(t.bounds[0].latitude_longitude)
        b.extend(t.bounds[1].latitude_longitude)
        url = tile_url_template.format(subdomain=subdomain, quadkey=t.quad_tree)
        query = query_template.format(bbox="{},{},{},{}".format(*b), tile=t.tms)
        print(url)
        # print(query)

        res = api.query(query)
        polygons = []
        for way in res.ways:
            points = []
            for node in way.nodes:
                p = Point(float(node.lat), float(node.lon))
                px = p.pixels(zoom=zoom_level)
                points.append((px[0]-tile_pixels[0], px[1]-t.bounds[1].pixels(zoom_level)[1]))
            poly = geometry.Polygon(points)
            svg = poly.svg()
            print(svg)
            polygons.append(poly)

        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, 3], dtype=np.uint8)
        # for poly in polygons:
            # for p in poly:
            #     print(p)

        # if res.ways:
        #     break
        # print(t.quad_tree)
    # print(tiles)


if __name__ == "__main__":
    bbox = {
        'tl': 47.355106,
        'tr': 8.518195,
        'bl': 47.383125,
        'br': 8.560596
    }
    zoom_levels = [18]
    for z in zoom_levels:
        osm_downloader(bbox=bbox, zoom_level=z)
