set PYTHONPATH=%~dp0
set CUDA_VISIBLE_DEVICES=''
python -V
python web/manage.py runserver