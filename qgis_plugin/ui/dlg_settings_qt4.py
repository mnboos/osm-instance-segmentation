# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlg_settings.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_DlgSettings(object):
    def setupUi(self, DlgSettings):
        DlgSettings.setObjectName(_fromUtf8("DlgSettings"))
        DlgSettings.resize(463, 161)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DlgSettings.sizePolicy().hasHeightForWidth())
        DlgSettings.setSizePolicy(sizePolicy)
        self.gridLayout = QtGui.QGridLayout(DlgSettings)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.btnClose = QtGui.QPushButton(DlgSettings)
        self.btnClose.setMinimumSize(QtCore.QSize(80, 0))
        self.btnClose.setObjectName(_fromUtf8("btnClose"))
        self.gridLayout.addWidget(self.btnClose, 2, 1, 1, 1)
        self.groupSettings = QtGui.QGroupBox(DlgSettings)
        self.groupSettings.setObjectName(_fromUtf8("groupSettings"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupSettings)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.txtServerPath = QtGui.QLineEdit(self.groupSettings)
        self.txtServerPath.setObjectName(_fromUtf8("txtServerPath"))
        self.gridLayout_2.addWidget(self.txtServerPath, 0, 2, 1, 1)
        self.lblServerPath = QtGui.QLabel(self.groupSettings)
        self.lblServerPath.setObjectName(_fromUtf8("lblServerPath"))
        self.gridLayout_2.addWidget(self.lblServerPath, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 2)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)

        self.retranslateUi(DlgSettings)
        QtCore.QObject.connect(self.btnClose, QtCore.SIGNAL(_fromUtf8("clicked()")), DlgSettings.reject)
        QtCore.QMetaObject.connectSlotsByName(DlgSettings)

    def retranslateUi(self, DlgSettings):
        DlgSettings.setWindowTitle(_translate("DlgSettings", "Settings", None))
        self.btnClose.setText(_translate("DlgSettings", "Close", None))
        self.groupSettings.setTitle(_translate("DlgSettings", "Settings", None))
        self.txtServerPath.setText(_translate("DlgSettings", "http://localhost:8080", None))
        self.lblServerPath.setText(_translate("DlgSettings", "Server URL", None))

