call pyuic4 -o dlg_about_qt4.py dlg_about.ui
call pyuic5 -o dlg_about_qt5.py dlg_about.ui


if NOT ["%errorlevel%"]==["0"] pause

exit