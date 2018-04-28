call pyuic4 -o dlg_about_qt4.py dlg_about.ui
call pyuic5 -o dlg_about_qt5.py dlg_about.ui

call pyuic4 -o dlg_settings_qt4.py dlg_settings.ui
call pyuic5 -o dlg_settings_qt5.py dlg_settings.ui


if NOT ["%errorlevel%"]==["0"] pause

exit