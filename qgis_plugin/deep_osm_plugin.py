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
        self.prediction_dialog.on_refresh_preview.connect(self._refresh_canvas)
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
            add_raw_predictions = self.prediction_dialog.add_raw_predictions
            self.continue_detect(rectangularize, create_raw_predictions_layer=add_raw_predictions)

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

    def continue_detect(self, rectangularize, create_raw_predictions_layer):
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
        status, raw = post("http://localhost:8000/predict", json.dumps(data))
        if status == 200 and raw:
            response = {}
            try:
                response = json.loads(raw)
            except Exception as e:
                info("Parsing response failed: {}", str(e))
            if "features" in response:
                all_features = []
                all_features.extend(response["deleted"])
                all_features.extend(response["added"])
                all_features.extend(response["changed"])
                if create_raw_predictions_layer:
                    self.create_layer("Predictions", response["features"], feature_layer_crs, False)
                self.create_layer("Changes", all_features, feature_layer_crs, True)
            else:
                info("Prediction failed: {}", response)

    @property
    def imagery_layer_name(self):
        """
         * Returns the name of the imagery layer from the settings
        :return:
        """

        return self.settings.value("IMAGERY_LAYER", None)

    def _update_predict_button(self):
        self.prediction_dialog.set_predict_enabled(self.canvas_refreshed)

    def _refresh_canvas(self):
        """
         * Refreshes the content of the custom canvas, which is used to save the content of the selected imagery
           layer to an image. A custom canvas is required, as we don't want anything else like vectors on the image.
           This is an asynchronous process and due to this, the method _update_image_data is the second step in this
           process.
        :return:
        """

        self.canvas_refreshed = False
        self._update_predict_button()
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

    def save_image(self, path):
        """
         * Saves the current content of the canvas to the specified path.
           Notice, that this is NOT the main QGIS canvas, but a custom one.
        :param path:
        :return:
        """

        canvas = self.canvas
        canvas.clearCache()
        size = canvas.size()
        image = QImage(size, QImage.Format_RGB32)

        painter = QPainter(image)
        settings = canvas.mapSettings()

        job = QgsMapRendererCustomPainterJob(settings, painter)
        job.renderSynchronously()
        painter.end()

        image.save(path)

    def _update_image_data(self):
        """
         * Is called as soon as the canvas is actually refreshed and its content can be saved to an image.
        :return:
        """

        temp_dir = os.path.join(tempfile.gettempdir(), "deep_osm")
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, "screenshot.png")
        self.save_image(file_path)

        assert os.path.isfile(file_path)
        info("Canvas refreshed and saved: {}", file_path)
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        self.image_data = base64.standard_b64encode(binary_data)
        self.canvas_refreshed = True
        self.prediction_dialog.set_image_preview(file_path)
        self._update_predict_button()

    def create_layer(self, name, features, crs, apply_style):
        """
         * Creates a new vector layer
        :param name: The name of the layer
        :param features: The features that will be added to the layer
        :param crs: The CRS of the layer
        :param apply_style: Whether the changes-style will be applied
        :return:
        """

        if not features:
            return

        layer_source = get_temp_dir("src_{}.json".format(uuid.uuid4()))
        feature_collection = self._get_feature_collection(features, crs)
        with open(layer_source, "w") as f:
            f.write(json.dumps(feature_collection))

        layer = QgsVectorLayer(layer_source, name, "ogr")
        if apply_style:
            style_path = os.path.join(os.path.dirname(__file__), "style.qml")
            res = layer.loadNamedStyle(style_path)
            info("Style applied: {}, {}", style_path, res)

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
