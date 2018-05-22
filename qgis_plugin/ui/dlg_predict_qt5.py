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
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 0, 1, 1)
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
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 5)
        self.btnCancel = QtWidgets.QPushButton(DlgPredict)
        self.btnCancel.setMinimumSize(QtCore.QSize(80, 0))
        self.btnCancel.setObjectName("btnCancel")
        self.gridLayout.addWidget(self.btnCancel, 2, 2, 1, 1)
        self.btnPredict = QtWidgets.QPushButton(DlgPredict)
        self.btnPredict.setObjectName("btnPredict")
        self.gridLayout.addWidget(self.btnPredict, 2, 1, 1, 1)

        self.retranslateUi(DlgPredict)
        self.btnCancel.clicked.connect(DlgPredict.reject)
        self.btnPredict.clicked.connect(DlgPredict.accept)
        QtCore.QMetaObject.connectSlotsByName(DlgPredict)

    def retranslateUi(self, DlgPredict):
        _translate = QtCore.QCoreApplication.translate
        DlgPredict.setWindowTitle(_translate("DlgPredict", "Prediction"))
        self.groupSettings.setTitle(_translate("DlgPredict", "Settings"))
        self.lblImageryLayer.setText(_translate("DlgPredict", "Imagery Layer"))
        self.btnCancel.setText(_translate("DlgPredict", "Cancel"))
        self.btnPredict.setText(_translate("DlgPredict", "Predict"))

