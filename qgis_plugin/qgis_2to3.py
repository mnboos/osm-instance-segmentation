from qgis.gui import *
from qgis.core import *

try:
    from qgis.core import (
        QgsMapLayerRegistry,
        QgsPoint
    )
except ImportError:
    from qgis.core import (
        QgsProject as QgsMapLayerRegistry,
        QgsPointXY as QgsPoint
    )

import os
try:
    QGIS3 = False
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4.QtNetwork import *
    from .ui import resources_rc_qt4
except ImportError:
    QGIS3 = True
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtNetwork import *
    from .ui import resources_rc_qt5
