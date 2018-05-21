# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlg_predict.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DlgPredict(object):
    def setupUi(self, DlgPredict):
        DlgPredict.setObjectName("DlgPredict")
        DlgPredict.resize(463, 172)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DlgPredict.sizePolicy().hasHeightForWidth())
        DlgPredict.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(DlgPredict)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.btnClose = QtWidgets.QPushButton(DlgPredict)
        self.btnClose.setMinimumSize(QtCore.QSize(80, 0))
        self.btnClose.setObjectName("btnClose")
        self.gridLayout.addWidget(self.btnClose, 2, 1, 1, 1)
        self.groupSettings = QtWidgets.QGroupBox(DlgPredict)
        self.groupSettings.setObjectName("groupSettings")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupSettings)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lblImageryLayer = QtWidgets.QLabel(self.groupSettings)
        self.lblImageryLayer.setObjectName("lblImageryLayer")
        self.gridLayout_2.addWidget(self.lblImageryLayer, 0, 0, 1, 1)
        self.cbxImageryLayer = QtWidgets.QComboBox(self.groupSettings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbxImageryLayer.sizePolicy().hasHeightForWidth())
        self.cbxImageryLayer.setSizePolicy(sizePolicy)
        self.cbxImageryLayer.setObjectName("cbxImageryLayer")
        self.gridLayout_2.addWidget(self.cbxImageryLayer, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)

        self.retranslateUi(DlgPredict)
        self.btnClose.clicked.connect(DlgPredict.reject)
        QtCore.QMetaObject.connectSlotsByName(DlgPredict)

    def retranslateUi(self, DlgPredict):
        _translate = QtCore.QCoreApplication.translate
        DlgPredict.setWindowTitle(_translate("DlgPredict", "Prediction"))
        self.btnClose.setText(_translate("DlgPredict", "Close"))
        self.groupSettings.setTitle(_translate("DlgPredict", "Settings"))
        self.lblImageryLayer.setText(_translate("DlgPredict", "Imagery Layer"))

