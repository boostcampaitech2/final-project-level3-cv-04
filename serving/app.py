from flask import Flask, Response

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, Potato Farm Flask!'

@app.route('/webcam')
def webcam():
    return Response(open('./static/webcam.html').read(), mimetype="text/html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006, threaded=True)
