from flask import Flask, request, Response, render_template
from agent_workflow import run_agent_for_topic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run-agent", methods=["POST"])
def run_agent():
    data = request.json
    topic = data.get("topic")
    if not topic:
        return {"error": "Missing topic"}, 400

    def generate():
        for chunk in run_agent_for_topic(topic):
            yield chunk

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
