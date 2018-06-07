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
        DlgPredict.resize(597, 289)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DlgPredict.sizePolicy().hasHeightForWidth())
        DlgPredict.setSizePolicy(sizePolicy)
        self.gridLayout = QtGui.QGridLayout(DlgPredict)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.groupSettings = QtGui.QGroupBox(DlgPredict)
        self.groupSettings.setObjectName(_fromUtf8("groupSettings"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupSettings)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.cbxImageryLayer = QtGui.QComboBox(self.groupSettings)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbxImageryLayer.sizePolicy().hasHeightForWidth())
        self.cbxImageryLayer.setSizePolicy(sizePolicy)
        self.cbxImageryLayer.setObjectName(_fromUtf8("cbxImageryLayer"))
        self.gridLayout_2.addWidget(self.cbxImageryLayer, 0, 2, 1, 1)
        self.lblImageryLayer = QtGui.QLabel(self.groupSettings)
        self.lblImageryLayer.setObjectName(_fromUtf8("lblImageryLayer"))
        self.gridLayout_2.addWidget(self.lblImageryLayer, 0, 0, 1, 1)
        self.lblAddRawPredictions = QtGui.QLabel(self.groupSettings)
        self.lblAddRawPredictions.setObjectName(_fromUtf8("lblAddRawPredictions"))
        self.gridLayout_2.addWidget(self.lblAddRawPredictions, 1, 0, 1, 1)
        self.chkAddPlainPredictions = QtGui.QCheckBox(self.groupSettings)
        self.chkAddPlainPredictions.setObjectName(_fromUtf8("chkAddPlainPredictions"))
        self.gridLayout_2.addWidget(self.chkAddPlainPredictions, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.groupSettings, 0, 0, 1, 5)
        self.grpPreview = QtGui.QGroupBox(DlgPredict)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grpPreview.sizePolicy().hasHeightForWidth())
        self.grpPreview.setSizePolicy(sizePolicy)
        self.grpPreview.setObjectName(_fromUtf8("grpPreview"))
        self.gridLayout_3 = QtGui.QGridLayout(self.grpPreview)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.lblPreviewText = QtGui.QLabel(self.grpPreview)
        self.lblPreviewText.setWordWrap(True)
        self.lblPreviewText.setObjectName(_fromUtf8("lblPreviewText"))
        self.gridLayout_3.addWidget(self.lblPreviewText, 0, 0, 1, 1)
        self.lblImage = QtGui.QLabel(self.grpPreview)
        self.lblImage.setObjectName(_fromUtf8("lblImage"))
        self.gridLayout_3.addWidget(self.lblImage, 2, 0, 1, 1)
        self.btnRefresh = QtGui.QPushButton(self.grpPreview)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnRefresh.sizePolicy().hasHeightForWidth())
        self.btnRefresh.setSizePolicy(sizePolicy)
        self.btnRefresh.setObjectName(_fromUtf8("btnRefresh"))
        self.gridLayout_3.addWidget(self.btnRefresh, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        self.gridLayout.addWidget(self.grpPreview, 1, 0, 1, 5)
        self.btnPredict = QtGui.QPushButton(DlgPredict)
        self.btnPredict.setObjectName(_fromUtf8("btnPredict"))
        self.gridLayout.addWidget(self.btnPredict, 3, 4, 1, 1)
        self.btnCancel = QtGui.QPushButton(DlgPredict)
        self.btnCancel.setMinimumSize(QtCore.QSize(80, 0))
        self.btnCancel.setObjectName(_fromUtf8("btnCancel"))
        self.gridLayout.addWidget(self.btnCancel, 3, 3, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 3, 0, 1, 3)

        self.retranslateUi(DlgPredict)
        QtCore.QObject.connect(self.btnCancel, QtCore.SIGNAL(_fromUtf8("clicked()")), DlgPredict.reject)
        QtCore.QObject.connect(self.btnPredict, QtCore.SIGNAL(_fromUtf8("clicked()")), DlgPredict.accept)
        QtCore.QMetaObject.connectSlotsByName(DlgPredict)

    def retranslateUi(self, DlgPredict):
        DlgPredict.setWindowTitle(_translate("DlgPredict", "Prediction", None))
        self.groupSettings.setTitle(_translate("DlgPredict", "Settings", None))
        self.lblImageryLayer.setText(_translate("DlgPredict", "Imagery Layer", None))
        self.lblAddRawPredictions.setText(_translate("DlgPredict", "Add plain predictions", None))
        self.chkAddPlainPredictions.setText(_translate("DlgPredict", "Add the unprocessed predictions from the neural network", None))
        self.grpPreview.setTitle(_translate("DlgPredict", "Preview", None))
        self.lblPreviewText.setText(_translate("DlgPredict", "The following image will be sent to the backend. Make sure, there are no white areas visible, otherwise click the refresh button.", None))
        self.lblImage.setText(_translate("DlgPredict", "Preview is loading...", None))
        self.btnRefresh.setText(_translate("DlgPredict", "Refresh", None))
        self.btnPredict.setText(_translate("DlgPredict", "Predict", None))
        self.btnCancel.setText(_translate("DlgPredict", "Cancel", None))

