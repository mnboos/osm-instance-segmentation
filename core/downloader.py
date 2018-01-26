import sys
import os
import overpy
from pygeotile.tile import Tile
from pygeotile.point import Point
import requests
import secrets
import shapely.geometry as geometry
import numpy as np
from core.settings import IMAGE_WIDTH, IMAGE_OUTPUT_FOLDER
from skimage import draw
import scipy.misc
import shutil


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


def tiles_from_bbox(bbox, zoom_level):
    """
     * Returns all tiles for the specified bounding box
    """

    if isinstance(bbox, dict):
        point_min = Point.from_latitude_longitude(latitude=bbox['tl'], longitude=bbox['tr'])
        point_max = Point.from_latitude_longitude(latitude=bbox['bl'], longitude=bbox['br'])
    elif isinstance(bbox, list):
        point_min = Point.from_latitude_longitude(latitude=bbox[1], longitude=bbox[0])
        point_max = Point.from_latitude_longitude(latitude=bbox[3], longitude=bbox[2])
    else:
        raise RuntimeError("bbox must bei either a dict or a list")
    tile_min = Tile.for_point(point_min, zoom_level)
    tile_max = Tile.for_point(point_max, zoom_level)
    tiles = []
    for x in range(tile_min.tms_x, tile_max.tms_x + 1):
        for y in range(tile_min.tms_y, tile_max.tms_y + 1):
            tiles.append(Tile.from_tms(tms_x=x, tms_y=y, zoom=zoom_level))
    return tiles


def osm_downloader(bbox_name, bbox, zoom_level, output_directory):
    api = overpy.Overpass()

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    tiles = tiles_from_bbox(bbox=bbox, zoom_level=zoom_level)
    response = requests.get("https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial?key={key}"
                            .format(key=secrets.BING_KEY))
    data = response.json()

    tile_url_template = data['resourceSets'][0]['resources'][0]['imageUrl']
    subdomain = data['resourceSets'][0]['resources'][0]['imageUrlSubdomains'][0]

    tiles_path = os.path.join(output_directory, 'tiles.txt')
    loaded_tiles = []
    if os.path.isfile(tiles_path):
        with open(tiles_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            loaded_tiles = list(map(lambda l: l[:-1], lines))  # remove '\n'

    nr_tiles = len(tiles)
    for i, t in enumerate(tiles):
        print("{} @ zoom {}: {:.1f}% (Tile {}/{})".format(bbox_name, zoom_level, 100/nr_tiles*i, i+1, nr_tiles))
        tms_x, tms_y = t.tms
        tile_name = "{z}_{x}_{y}".format(z=zoom_level, x=tms_x, y=tms_y)
        if tile_name in loaded_tiles:
            continue

        minx, maxy = t.bounds[0].pixels(zoom_level)
        maxx, miny = t.bounds[1].pixels(zoom_level)
        b = []
        b.extend(t.bounds[0].latitude_longitude)
        b.extend(t.bounds[1].latitude_longitude)
        url = tile_url_template.format(subdomain=subdomain, quadkey=t.quad_tree)
        query = query_template.format(bbox="{},{},{},{}".format(*b), tile=t.tms)
        # print(url)
        # print(query)
        res = api.query(query)
        mask = np.zeros((IMAGE_WIDTH, IMAGE_WIDTH), dtype=np.uint8)
        for way in res.ways:
            points = []
            for node in way.nodes:
                p = Point(float(node.lat), float(node.lon))
                px = p.pixels(zoom=zoom_level)
                points.append((px[0]-minx, px[1]-miny))

            try:
                poly = geometry.Polygon(points)
                tile_rect = geometry.box(0, 0, IMAGE_WIDTH, IMAGE_WIDTH)
                poly = poly.intersection(tile_rect)
            except:
                # print("Intersection failed for polygon and rectangle: poly='{}', box='{}'".format(poly, tile_rect))
                continue
            polygons = []
            if isinstance(poly, geometry.MultiPolygon):
                for p in poly.geoms:
                    if isinstance(p, geometry.Polygon):
                        polygons.append(p)
                    elif isinstance(p, geometry.MultiPolygon):
                        print(poly)
            else:
                polygons.append(poly)
            update_mask(mask, polygons)

        if res.ways and mask.max():
            file_name = "{}.tif".format(tile_name)
            mask_path = os.path.join(output_directory, file_name)
            img_path = os.path.join(output_directory, file_name+'f')
            scipy.misc.imsave(mask_path, mask)
            if not os.path.isfile(img_path):
                response = requests.get(url, stream=True)
                response.raw.decode_content = True
                with open(img_path, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                del response
        with open(os.path.join(output_directory, "tiles.txt"), 'a') as f:
            f.write("{}\n".format(tile_name))


def update_mask(mask, polygons):
    """
     * The first polygon is the exterior ring. All others are treated as interior rings and will just invert
       the corresponding area of the mask.
    :param mask:
    :param polygons:
    :return:
    """
    for p in polygons:
        if isinstance(p, geometry.MultiPolygon):
            update_mask(mask, p.geoms)
        elif not isinstance(p, geometry.Polygon):
            continue
        xs, ys = p.exterior.coords.xy
        rr, cc = draw.polygon(xs, ys, (IMAGE_WIDTH, IMAGE_WIDTH))
        mask[cc, rr] = np.invert(mask[cc, rr])


def download():
    bboxes = {
        'chicago': [-87.779857, 41.87806, -87.659265, 41.953457],
        'chicago2': [-87.739813, 41.837528, -87.70282, 41.865722],
        'firenze': [11.239844, 43.765851, 11.289969, 43.790065],
        'nuernberg': [11.046668, 49.470696, 11.129795, 49.492332],
        'wettingen': [8.309857, 47.456269, 8.340327, 47.473386],
        'duebendorf': [8.606587, 47.392654, 8.627401, 47.404593],
        'goldach': {
            'tr': 9.462776,
            'tl': 47.465723,
            'br': 9.489598,
            'bl': 47.485157
        },
        'stgallen': {
            'tr': 9.405892,
            'tl': 47.433161,
            'br': 9.424131,
            'bl': 47.444191
        },
        'rapperswil': {
            'tr': 8.818724,
            'tl': 47.222126,
            'br': 8.847435,
            'bl': 47.234629
        },
        'hombrechtikon': {
            'tr': 8.815956,
            'tl': 47.237018,
            'br': 8.826664,
            'bl': 47.247157
        },
        'zurich': {
            'tr': 8.47716,
            'tl': 47.36036,
            'br': 8.573806,
            'bl': 47.401508,
        },
        'boston_financial_district': {
            'tr': -71.07081,
            'tl': 42.351557,
            'br': -71.053601,
            'bl': 42.362942
        },
        'boston_houses': {
            'tr': -71.079802,
            'tl': 42.280151,
            'br': -71.05062,
            'bl': 42.29958
        },
        'bern': {
            'tr': 7.420455,
            'tl': 46.935277,
            'br': 7.46337,
            'bl': 46.965862
        },
        'new_york': {
            'tr': -74.02059,
            'tl': 40.646089,
            'br': -73.864722,
            'bl': 40.77413
        },
        'berlin': {
            'tr': 13.326222,
            'tl': 52.418412,
            'br': 13.548696,
            'bl': 52.563071
        },
        'munich': {
            'tr': 11.465968,
            'tl': 48.096287,
            'br': 11.626643,
            'bl': 48.189756
        },
        'rome': {
            'tr': 12.449309,
            'tl': 41.869589,
            'br': 12.531363,
            'bl': 41.923766
        },
        'san_francisco': {
            'tr': -122.508763,
            'tl': 37.714932,
            'br': -122.383106,
            'bl': 37.787682
        },
    }

    print(sys.argv)
    if len(sys.argv) > 1:
        city = sys.argv[1]
        valid_cities = bboxes.keys()
        if city not in valid_cities:
            raise RuntimeError("'{}' is not a valid city. Valid cities are: {}".format(city, valid_cities))
        cities = [city]
    else:
        cities = bboxes.keys()

    for bbox_name in cities:
        print("Processing bbox '{}'".format(bbox_name))
        bbox = bboxes[bbox_name]
        zoom_levels = [18, 19]
        for z in zoom_levels:
            osm_downloader(bbox_name=bbox_name,
                           bbox=bbox,
                           zoom_level=z,
                           output_directory=os.path.join(IMAGE_OUTPUT_FOLDER, bbox_name))


if __name__ == "__main__":
    download()