from .qgis_2to3 import *
from .ui.dialogs import AboutDialog


class DeepOsmPlugin:

    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.about_action = self._create_action("About", "info.svg", self.show_about)
        self.detect_action = self._create_action("Detect", "group.svg", self.show_about)

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

    def unload(self):
        self.iface.layerToolBar().removeAction(self.toolButtonAction)

    @staticmethod
    def show_about():
        AboutDialog().show()