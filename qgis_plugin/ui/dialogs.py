from ..qgis_2to3 import *
try:
    from .dlg_about_qt5 import Ui_DlgAbout
except:
    from .dlg_about_qt4 import Ui_DlgAbout


def _update_size(dialog):
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