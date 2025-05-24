from flask import Flask, request, jsonify
import base64
import json

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        payment_data = request.get_json()
        payment_token = payment_data.get('paymentMethodData', {}).get('tokenizationData', {}).get('token')
        if not payment_token:
            return jsonify({'message': 'No payment token received'}), 400
        
        # Simulate payment processing (replace with actual gateway API call)
        decoded_token = base64.b64decode(payment_token).decode('utf-8') if payment_token else ''
        print('Received payment token:', decoded_token)
        
        return jsonify({'message': 'Payment processed successfully'})
    except Exception as e:
        return jsonify({'message': f'Payment error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)