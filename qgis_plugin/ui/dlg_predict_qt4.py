# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlg_predict.ui'
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

class Ui_DlgPredict(object):
    def setupUi(self, DlgPredict):
        DlgPredict.setObjectName(_fromUtf8("DlgPredict"))
        DlgPredict.resize(463, 172)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DlgPredict.sizePolicy().hasHeightForWidth())
        DlgPredict.setSizePolicy(sizePolicy)
        self.gridLayout = QtGui.QGridLayout(DlgPredict)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 0, 1, 1)
        self.groupSettings = QtGui.QGroupBox(DlgPredict)
        self.groupSettings.setObjectName(_fromUtf8("groupSettings"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupSettings)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.lblImageryLayer = QtGui.QLabel(self.groupSettings)
        self.lblImageryLayer.setObjectName(_fromUtf8("lblImageryLayer"))
        self.gridLayout_2.addWidget(self.lblImageryLayer, 0, 0, 1, 1)
        self.cbxImageryLayer = QtGui.QComboBox(self.groupSettings)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbxImageryLayer.sizePolicy().hasHeightForWidth())
        self.cbxImageryLayer.setSizePolicy(sizePolicy)
        self.cbxImageryLayer.setObjectName(_fromUtf8("cbxImageryLayer"))
        self.gridLayout_2.addWidget(self.cbxImageryLayer, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 5)
        self.btnCancel = QtGui.QPushButton(DlgPredict)
        self.btnCancel.setMinimumSize(QtCore.QSize(80, 0))
        self.btnCancel.setObjectName(_fromUtf8("btnCancel"))
        self.gridLayout.addWidget(self.btnCancel, 2, 2, 1, 1)
        self.btnPredict = QtGui.QPushButton(DlgPredict)
        self.btnPredict.setObjectName(_fromUtf8("btnPredict"))
        self.gridLayout.addWidget(self.btnPredict, 2, 1, 1, 1)

        self.retranslateUi(DlgPredict)
        QtCore.QObject.connect(self.btnCancel, QtCore.SIGNAL(_fromUtf8("clicked()")), DlgPredict.reject)
        QtCore.QObject.connect(self.btnPredict, QtCore.SIGNAL(_fromUtf8("clicked()")), DlgPredict.accept)
        QtCore.QMetaObject.connectSlotsByName(DlgPredict)

    def retranslateUi(self, DlgPredict):
        DlgPredict.setWindowTitle(_translate("DlgPredict", "Prediction", None))
        self.groupSettings.setTitle(_translate("DlgPredict", "Settings", None))
        self.lblImageryLayer.setText(_translate("DlgPredict", "Imagery Layer", None))
        self.btnCancel.setText(_translate("DlgPredict", "Cancel", None))
        self.btnPredict.setText(_translate("DlgPredict", "Predict", None))

