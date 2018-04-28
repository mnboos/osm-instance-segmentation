# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlg_settings.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DlgSettings(object):
    def setupUi(self, DlgSettings):
        DlgSettings.setObjectName("DlgSettings")
        DlgSettings.resize(463, 161)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DlgSettings.sizePolicy().hasHeightForWidth())
        DlgSettings.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(DlgSettings)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.btnClose = QtWidgets.QPushButton(DlgSettings)
        self.btnClose.setMinimumSize(QtCore.QSize(80, 0))
        self.btnClose.setObjectName("btnClose")
        self.gridLayout.addWidget(self.btnClose, 2, 1, 1, 1)
        self.groupSettings = QtWidgets.QGroupBox(DlgSettings)
        self.groupSettings.setObjectName("groupSettings")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupSettings)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.txtServerPath = QtWidgets.QLineEdit(self.groupSettings)
        self.txtServerPath.setObjectName("txtServerPath")
        self.gridLayout_2.addWidget(self.txtServerPath, 0, 1, 1, 1)
        self.lblServerPath = QtWidgets.QLabel(self.groupSettings)
        self.lblServerPath.setObjectName("lblServerPath")
        self.gridLayout_2.addWidget(self.lblServerPath, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)

        self.retranslateUi(DlgSettings)
        self.btnClose.clicked.connect(DlgSettings.reject)
        QtCore.QMetaObject.connectSlotsByName(DlgSettings)

    def retranslateUi(self, DlgSettings):
        _translate = QtCore.QCoreApplication.translate
        DlgSettings.setWindowTitle(_translate("DlgSettings", "About Vector Tile Reader"))
        self.btnClose.setText(_translate("DlgSettings", "Close"))
        self.groupSettings.setTitle(_translate("DlgSettings", "Settings"))
        self.txtServerPath.setText(_translate("DlgSettings", "http://localhost:8080"))
        self.lblServerPath.setText(_translate("DlgSettings", "Server URL"))

