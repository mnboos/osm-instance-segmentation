from .qgis_2to3 import *
from .ui.dialogs import AboutDialog
from .log_helper import info
from .tile_helper import get_code_from_epsg, convert_coordinate, get_zoom_by_scale
import tempfile
import base64
import json
from .network_helper import post, post_async


class DeepOsmPlugin:

    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.about_action = self._create_action("About", "info.svg", self.show_about)
        self.detect_action = self._create_action("Detect", "group.svg", self.detect)

        self.popupMenu = QMenu(self.iface.mainWindow())
        self.popupMenu.addAction(self.about_action)
        self.toolButton = QToolButton()
        self.toolButton.setMenu(self.popupMenu)
        self.toolButton.setDefaultAction(self.detect_action)
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

    def detect(self):
        extent = self.iface.mapCanvas().extent()
        x = extent.xMinimum()
        y = extent.yMaximum()
        qgis_crs = self._get_qgis_crs()

        lon, lat = convert_coordinate(qgis_crs, 4326, y, x)
        scale = self._get_current_map_scale()
        zoom = get_zoom_by_scale(scale)

        info("extent @ zoom {}: {}", zoom, (lon, lat))
        temp_dir = os.path.join(tempfile.gettempdir(), "deep_osm")
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, "screenshot.tiff")
        self.iface.mapCanvas().saveAsImage(file_path, None, 'TIFF')
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        image_data = base64.standard_b64encode(binary_data)
        data = {
            'lat': lat,
            'lon': lon,
            'zoom_level': zoom,
            'image_data': image_data
        }
        response = post("http://localhost:8000/inference", json.dumps(data))
        info("response: {}", response)

    def detection_finished(self, data):
        info("detection finished: {}", data)

    def unload(self):
        self.iface.layerToolBar().removeAction(self.toolButtonAction)

    @staticmethod
    def show_about():
        AboutDialog().show()
