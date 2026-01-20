import secrets

import atexit
from apscheduler.schedulers.background import BackgroundScheduler

import flask
from flask import request, make_response, abort, url_for
from flask import redirect, render_template, session, jsonify
from flask import send_from_directory, send_file, Response


app = flask.Flask(__name__)
app.secret_key = secrets.token_hex(64) # session cookies encryption

##app.config['MAX_CONTENT_LENGTH'] = 50_000_000 # 50MB limit

@app.route("/", methods=['POST','GET'])
def page_main():
    match request.method:
        case "GET":
            return "MAINPAGE"#redirect("/")
        case "POST":
            return redirect("/")
    return redirect("/")


def generate_file_chunks(filename):
    with open(filename, 'rb') as file:
        while True:
            chunk = file.read(4096)  # Read 4096 bytes at a time
            if not chunk: break
            yield chunk

@app.route('/download/<file>')
def download(file):
    path = ""
    if path:
        return Response(generate_file_chunks(path), mimetype='application/octet-stream')
    return redirect("/")



##### ERROR HANDLERS
@app.errorhandler(404)
def page_not_found(error): return redirect("/")
@app.errorhandler(405)
def method_not_allowed(error): return redirect("/")
#####





##### SCHEDULED TASKS
##scheduler = BackgroundScheduler()
####job = scheduler.add_job(func=, trigger="interval", hours=1)
##scheduler.start()
##
##def safe_exit():
##    scheduler.shutdown()
##
##atexit.register(safe_exit)
#####

