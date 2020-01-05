from flask import Flask, request, render_template, jsonify
import inference
import os
import tornado.wsgi
import tornado.httpserver

app = Flask(__name__)
REPO_DIRNAME = os.path.dirname(os.path.abspath(__file__))


news_model = inference.Inference('config.json')

@app.route("/")
def index():
    return render_template("template.html")

@app.route("/classify", methods=["POST", "GET"])
def classify():
    try:
        text = request.form['text']
        _, predict_category = news_model.predict(text)
        result = {
            "result": predict_category
        }
        result = {str(key): value for key, value in result.items()}
    except Exception as err:
        print('Classification error: ', err)
        return (False, 'Something went wrong when classify the text. Maybe try another one?')
    return render_template("template.html", input_text=text, predict_category=predict_category)
    # return jsonify(result=result)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    # app.run(debug=True)
    start_tornado(app)
