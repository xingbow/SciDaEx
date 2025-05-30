#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from app import app
from app.routes.app import create_app
from gevent.pywsgi import WSGIServer
from app.dataService import globalVariable as GV

app = create_app()
# app.debug = True
print("backend port: ", GV.backend_port)
http_server = WSGIServer(('localhost', GV.backend_port), app)
http_server.serve_forever()
