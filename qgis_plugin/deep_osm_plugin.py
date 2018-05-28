import os
import uuid
from .qgis_2to3 import *
from .ui.dialogs import AboutDialog, SettingsDialog, PredictionDialog
from .log_helper import info, get_temp_dir
from .tile_helper import get_code_from_epsg, convert_coordinate, get_zoom_by_scale
import tempfile
import base64
import json
from .network_helper import post, post_async


class DeepOsmPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.settings = QSettings("Vector Tile Reader", "vectortilereader")
        self.settings_dialog = SettingsDialog(self.settings)
        self.prediction_dialog = PredictionDialog(self.settings)
        self.prediction_dialog.on_image_layer_change.connect(self._refresh_canvas)
        self.canvas = QgsMapCanvas()
        self.canvas.mapCanvasRefreshed.connect(self._update_image_data)
        self.image_data = None
        self.canvas_refreshed = False

    def initGui(self):
        self.about_action = self._create_action("About", "info.svg", self.show_about)
        self.detect_rectangles_action = self._create_action("Detect building area (rectangularized)", "group.svg", lambda: self.detect(True))
        self.detect_raw_action = self._create_action("Detect building area (raw)", "group.svg", lambda: self.detect(False))
        self.settings_action = self._create_action("Settings", "settings.svg", lambda: self.settings_dialog.show())

        self.popupMenu = QMenu(self.iface.mainWindow())
        self.popupMenu.addAction(self.detect_raw_action)
        self.popupMenu.addAction(self.detect_rectangles_action)
        self.popupMenu.addAction(self.settings_action)
        self.popupMenu.addAction(self.about_action)
        self.toolButton = QToolButton()
        self.toolButton.setMenu(self.popupMenu)
        self.toolButton.setDefaultAction(self.detect_raw_action)
        self.toolButton.setPopupMode(QToolButton.MenuButtonPopup)
        self.toolButtonAction = self.iface.layerToolBar().addWidget(self.toolButton)

    def _create_action(self, title, icon, callback, is_enabled=True):
        new_action = QAction(QIcon(':/plugins/deep_osm/{}'.format(icon)), title, self.iface.mainWindow())
        new_action.triggered.connect(callback)
        new_action.setEnabled(is_enabled)
        return new_action

    def _get_qgis_crs(self):
        canvas = self.iface.mapCanvas()
        return get_code_from_epsg(canvas.mapSettings().destinationCrs().authid())

    def _get_current_map_scale(self):
        canvas = self.iface.mapCanvas()
        current_scale = int(round(canvas.scale()))
        return current_scale

    def detect(self, rectangularize):
        info("extent: {}", self.iface.mapCanvas().extent().asWktPolygon())
        layer_names = list(map(lambda l: l.name(), QgsMapLayerRegistry.instance().mapLayers().values()))
        self.prediction_dialog.update_layers(layer_names)
        self._refresh_canvas()
        res = self.prediction_dialog.show()
        if res:
            self.continue_detect(rectangularize)

    def get_reference_features(self):
        rect = self.iface.mapCanvas().extent()
        imagery_layer_name = self.imagery_layer_name
        feature_types = ['Gebaeude', 'Strasse_Weg']
        result = []
        crs = None
        for layer in list(QgsMapLayerRegistry.instance().mapLayers().values()):
            layer_name = layer.name()
            if layer_name == imagery_layer_name or not isinstance(layer, QgsVectorLayer):
                continue

            layer_crs = layer.crs().authid()
            info("crs: {}", layer_crs)
            x_min, y_min = convert_coordinate(source_crs=self._get_qgis_crs(), target_crs=layer_crs, lat=rect.yMinimum(), lng=rect.xMinimum())
            x_max, y_max = convert_coordinate(source_crs=self._get_qgis_crs(), target_crs=layer_crs, lat=rect.yMaximum(), lng=rect.xMaximum())
            wkt = QgsRectangle(x_min, y_min, x_max, y_max).asWktPolygon()

            for feature_type in feature_types:
                expr = QgsExpression("\"Typ\" = '{}' and intersects(bounds($geometry), geom_from_wkt('{}'))"
                                     .format(feature_type, wkt))
                feature_iterator = layer.getFeatures(QgsFeatureRequest(expr))
                geoms = [i.geometry().exportToWkt() for i in feature_iterator]
                info("layer '{}' has {} matching features", layer_name, len(geoms))
                if geoms:
                    result.extend(geoms)
                    if not crs:
                        crs = layer_crs
        return result, crs

    def continue_detect(self, rectangularize):
        qgis_crs = self._get_qgis_crs()
        extent = self.iface.mapCanvas().extent()

        features, feature_layer_crs = self.get_reference_features()
        info("Feature layer CRS: {}", feature_layer_crs)
        if not feature_layer_crs:
            feature_layer_crs = qgis_crs

        lon_min, lat_min = convert_coordinate(qgis_crs, feature_layer_crs, extent.yMinimum(), extent.xMinimum())
        lon_max, lat_max = convert_coordinate(qgis_crs, feature_layer_crs, extent.yMaximum(), extent.xMaximum())
        scale = self._get_current_map_scale()
        zoom = get_zoom_by_scale(scale)

        info("extent @ zoom {}: {}", zoom, (lon_min, lat_min, lon_max, lat_max))
        data = {
            'rectangularize': rectangularize,
            'x_min': lon_min,
            'x_max': lon_max,
            'y_min': lat_min,
            'y_max': lat_max,
            'zoom_level': zoom,
            'image_data': str(self.image_data),
            'reference_features': features
        }
        # print(data['image_data'], type(data['image_data']))
        status, raw = post("http://localhost:8000/predict", json.dumps(data))
        if status == 200 and raw:
            response = {}
            try:
                response = json.loads(raw)
            except Exception as e:
                info("Parsing response failed: {}", str(e))
            if "features" in response:
                self.detection_finished(response["features"], feature_layer_crs)
            else:
                info("Prediction failed: {}", response)

    @property
    def imagery_layer_name(self):
        return self.settings.value("IMAGERY_LAYER", None)

    def _refresh_canvas(self):
        self.canvas_refreshed = False
        layer_name = self.imagery_layer_name
        info("Refreshing canvas for layer: {}", layer_name)
        layers = list(filter(lambda l: l.name() == layer_name, QgsMapLayerRegistry.instance().mapLayers().values()))
        if layers:
            layer = layers[0]
        else:
            self.image_data = None
            return
        canvas = self.canvas
        if QGIS3:
            canvas.setLayers([layer])
        else:
            canvas.setLayerSet([QgsMapCanvasLayer(layer)])
        canvas.setCanvasColor(Qt.white)
        canvas.setExtent(self.iface.mapCanvas().extent())
        canvas.refreshAllLayers()

    def _update_image_data(self):
        temp_dir = os.path.join(tempfile.gettempdir(), "deep_osm")
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, "screenshot.png")
        # canvas = self.canvas
        canvas = self.iface.mapCanvas()
        canvas.saveAsImage(file_path, None, 'PNG')
        assert os.path.isfile(file_path)
        info("Canvas refreshed and saved: {}", file_path)
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        self.image_data = base64.standard_b64encode(binary_data)
        self.canvas_refreshed = True

    def detection_finished(self, features, crs):
        info("detection finished. {} features predicted", len(features))
        # layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "Prediction", "memory")
        # layer.setCrs(QgsCoordinateReferenceSystem(3857))
        # layer.startEditing()

        layer_source = get_temp_dir("src_{}.json".format(uuid.uuid4()))
        feature_collection = self._get_feature_collection(features, crs)
        with open(layer_source, "w") as f:
            f.write(json.dumps(feature_collection))

        layer = QgsVectorLayer(layer_source, "Predictions", "ogr")

        # for f in features:
        #     feature = QgsFeature()
        #     geom = QgsGeometry.fromWkt(f)
        #     if geom:
        #         feature.setGeometry(geom)
        #         layer.addFeature(feature, True)
        # layer.commitChanges()
        layer.updateExtents()
        QgsMapLayerRegistry.instance().addMapLayer(layer)
        info("done")

    def _get_feature_collection(self, features, source_crs):
        """
         * Returns an empty GeoJSON FeatureCollection with the coordinate reference system (crs) set to EPSG3857
        """

        # source_crs = self._get_qgis_crs()
        if source_crs:
            epsg_id = get_code_from_epsg(source_crs)
        else:
            epsg_id = 3857

        crs = {
            "type": "name",
            "properties": {
                    "name": "urn:ogc:def:crs:EPSG::{}".format(epsg_id)}}

        return {
            "type": "FeatureCollection",
            "crs": crs,
            "features": features}

    def unload(self):
        self.iface.layerToolBar().removeAction(self.toolButtonAction)

    @staticmethod
    def show_about():
        AboutDialog().show()
