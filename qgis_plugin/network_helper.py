from future import standard_library
standard_library.install_aliases()
from .log_helper import warn, info, remove_key

from .qgis_2to3 import *


def url_exists(url):
    reply = get_async_reply(url, head_only=True)
    while not reply.isFinished():
        QApplication.processEvents()

    status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
    result = status == 200
    error = None
    if not status:
        error = reply.errorString()
    if status == 302:
        error = "Loading error: Moved Temporarily.\n\nURL incorrect? Missing or incorrect API key?"
    elif status == 404:
        error = "Loading error: Resource not found.\n\nURL incorrect?"
    elif error:
        error = "Loading error: {}\n\nURL incorrect?".format(error)

    return result, error


def get_async_reply(url, head_only=False):
    m = QgsNetworkAccessManager.instance()
    req = QNetworkRequest(QUrl(url))
    if head_only:
        reply = m.head(req)
    else:
        reply = m.get(req)
    return reply


def post_async(url, data, callback=None):
    m = QgsNetworkAccessManager.instance()
    req = QNetworkRequest(QUrl(url))
    reply = m.post(req, data)
    if callback:
        reply.finished.connect(callback)
        reply.error.connect(callback)
    return reply


def post(url, data):
    reply = post_async(url, data)
    while not reply.isFinished():
        if QApplication:
            QApplication.processEvents()
        else:
            return 400, None

    http_status_code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
    if http_status_code == 200:
        content = reply.readAll().data()
    else:
        if http_status_code is None:
            content = "Request failed: {}".format(reply.errorString())
        else:
            content = "Request failed: HTTP status {}".format(http_status_code)
        warn(content)
    return http_status_code, content


def load_url(url):
    reply = get_async_reply(url)
    while not reply.isFinished():
        QApplication.processEvents()

    http_status_code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
    if http_status_code == 200:
        content = reply.readAll().data()
    else:
        if http_status_code is None:
            content = "Request failed: {}".format(reply.errorString())
        else:
            content = "Request failed: HTTP status {}".format(http_status_code)
        warn(content)
    return http_status_code, content
