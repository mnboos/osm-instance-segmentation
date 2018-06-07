from ..qgis_2to3 import *
from ..log_helper import info
try:
    from .dlg_about_qt5 import Ui_DlgAbout
    from .dlg_settings_qt5 import Ui_DlgSettings
    from .dlg_predict_qt5 import Ui_DlgPredict
except:
    from .dlg_about_qt4 import Ui_DlgAbout
    from .dlg_settings_qt4 import Ui_DlgSettings
    from .dlg_predict_qt4 import Ui_DlgPredict


def _update_size(dialog):
    """
     * A helper which simplifies scaling the windows for other resolutions
    :param dialog:
    :return:
    """

    screen_resolution = QApplication.desktop().screenGeometry()
    screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
    new_width = None
    new_height = None
    if screen_width > 1920 or screen_height > 1080:
        new_width = dialog.width() / 1920.0 * screen_width
        new_height = dialog.height() / 1080.0 * screen_height
        dialog.setMinimumSize(new_width, new_height)
    elif dialog.width() >= screen_width or dialog.height() >= screen_height:
        margin = 40
        new_width = screen_width - margin
        new_height = screen_height - margin

    if new_width and new_height:
        dialog.resize(new_width, new_height)


class AboutDialog(QDialog, Ui_DlgAbout):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        self._load_about()
        _update_size(self)

    def _load_about(self):
        about_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "about.html")
        if os.path.isfile(about_path):
            with open(about_path, 'r') as f:
                html = f.read()
                self.txtAbout.setHtml(html)

    def show(self):
        self.exec_()


class SettingsDialog(QDialog, Ui_DlgSettings):
    def __init__(self, settings):
        QDialog.__init__(self)
        self.setupUi(self)
        _update_size(self)
        self._settings = settings

        self.txtServerPath.textChanged.connect(self.on_host_changed)
        server_path = settings.value("HOST", None)
        if server_path:
            self.txtServerPath.setText(server_path)

    def show(self):
        self.exec_()

    def on_host_changed(self):
        self._settings.setValue("HOST", self.txtServerPath.text())


class PredictionDialog(QDialog, Ui_DlgPredict):

    on_image_layer_change = pyqtSignal("QString")
    on_refresh_preview = pyqtSignal()

    def __init__(self, settings):
        QDialog.__init__(self)
        self.setupUi(self)
        _update_size(self)
        self._settings = settings
        self._updating_layers = False
        self.cbxImageryLayer.currentIndexChanged['QString'].connect(self._handle_imagery_layer_change)
        self.btnRefresh.clicked.connect(self._refresh_preview)

    def _refresh_preview(self):
        self.on_refresh_preview.emit()

    @property
    def add_raw_predictions(self):
        return self.chkAddPlainPredictions.isChecked()

    def set_predict_enabled(self, enabled):
        """
         * Enables / disabled the predict button
        :param enabled:
        :return:
        """

        self.btnPredict.setEnabled(enabled)

    def _handle_imagery_layer_change(self, new_layer):
        updated = self._handle_layer_change(new_layer, "IMAGERY_LAYER")
        if updated:
            self.on_image_layer_change.emit(new_layer)

    def _handle_layer_change(self, layer_name, setting_key):
        updated = False
        if not self._updating_layers:
            info("Selected layer: {}", layer_name)
            self._settings.setValue(setting_key, layer_name)
            updated = True
        return updated

    def set_image_preview(self, path):
        """
         * Updates the preview image
        :param path:
        :return:
        """

        img = QImage(path)
        pixmap = QPixmap(img)
        self.lblImage.setPixmap(pixmap)

    @staticmethod
    def _add_layers_to(combobox, layers):
        for layer_name in sorted(layers):
            is_already_added = combobox.findText(layer_name) != -1
            if not is_already_added:
                combobox.addItem(layer_name)

    def update_layers(self, layer_names):
        """
         * Adds the specified layers to the combobox for the selection of the imagery layer
        :param layer_names:
        :return:
        """
        self._updating_layers = True
        self._add_layers_to(self.cbxImageryLayer, layer_names)

        imagery_layer = self._settings.value("IMAGERY_LAYER", None)
        if imagery_layer:
            self._select_layer(imagery_layer)

        self._updating_layers = False

    def _select_layer(self, name):
        """
         * selects the specified layer in the dropdown
        :param name:
        :param target_combobox:
        :return:
        """

        if name:
            index = self.cbxImageryLayer.findText(name)
            if index:
                self.cbxImageryLayer.setCurrentIndex(index)

    def show(self):
        """
         * Just a wrapper for the exec_() method of a dialog
        """

        return self.exec_()

