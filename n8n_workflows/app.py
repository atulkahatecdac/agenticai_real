from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# URL of our n8n webhook
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/support/message"

@app.route('/trigger-flow', methods=['POST'])
def trigger_flow():
    # Get form fields
    user_id = request.form.get('user_id')
    order_id = request.form.get('order_id')
    file = request.files.get('image')  # the uploaded image

    if not user_id or not order_id or not file:
        return jsonify({"error": "user_id, order_id, and image are required"}), 400

    # Prepare the multipart/form-data for n8n webhook
    files = {
        "file": (file.filename, file.stream, file.mimetype)
    }
    data = {
        "user_id": user_id,
        "order_id": order_id
    }

    # Send POST request to n8n webhook
    response = requests.post(N8N_WEBHOOK_URL, data=data, files=files)

    return jsonify({
        "status": "success",
        "n8n_response": response.text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
