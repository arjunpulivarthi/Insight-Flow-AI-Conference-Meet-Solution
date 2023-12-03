from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import tensorflow
import os

app = Flask(__name__)

# Demo hat colours
person_hat_colors = {
    'Person 1': 'yellow',
    'Person 2': 'green',
    'Person 3': 'red',
    'Person 4': 'blue',
    'Person 5': 'white',
    'Person 6': 'black'
}

person_chart_data = {
    'Person 1': 100,
    'Person 2': 100,
    'Person 3': 100,
    'Person 4': 100,
    'Person 5': 100,
    'Person 6': 100
}

# Simulated user credentials for demonstration purposes
VALID_USERNAME = '123'
VALID_PASSWORD = '123'

@app.route('/')
def login():
    # Uncomment if running demo
    # return render_template('speech-to-text-demo.html')
    
    return render_template('login.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return render_template('dashboard.html', person_hat_colors=person_hat_colors, person_chart_data=person_chart_data)
    else:
        return redirect(url_for('login'))

# Route for the meeting page
@app.route('/meeting', methods=['POST','GET'])  
def meeting():
    return render_template('meeting.html')

# Route for returning icons for the meeting
@app.route('/icons/<path:filename>')
def serve_icon(filename):
    return send_from_directory('static/icons', filename)

# Receives text form the speech-to-text API
@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        text_data = request.json['textData']
        print('Received Text:', text_data)
        save_text_to_file(text_data)

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Routing the speech to text demonstration
@app.route('/speech_to_text', methods=['POST','GET'])
def speech_to_text():
    return render_template('speech_to_text.html')

# Saves the text to a file
def save_text_to_file(text):
    print('Saving text to file...')
    try:
        file_path = os.path.join('static', 'converted_text.txt')
        with open(file_path, 'a') as file:
            file.write(str(text) + '\n')
    except Exception as e:
        print(e)

# When the meeting ends, the text is sent to the model for analysis
@app.route('/end_meeting', methods=['POST','GET'])
def end_meeting():
    from transformers import TFRobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
    import numpy as np

    # Load model
    config = RobertaConfig.from_pretrained('models/roberta/config.json')
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=config.num_labels)
    weights_path = 'models/roberta/tf_model.h5'
    model.load_weights(weights_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    file_path = os.path.join('static', 'converted_text.txt')
    with open(file_path, 'r') as file:
        text = file.read()

        # Tokenize text
        inputs = tokenizer(text, return_tensors='tf')
        input_ids = inputs['input_ids']
        
        # Make predictions
        predictions = model(input_ids)

        # Get predicted label
        predicted_label = np.argmax(predictions[0], axis=-1).tolist()[0]
        print('Predicted Label:', predicted_label)
        
        emotion_to_hat = {
            'admiration': 'yellow',      # Yellow for optimistic
            'amusement': 'green',        # Green for creative
            'anger': 'red',              # Red for emotional
            'annoyance': 'black',        # Black for logic
            'approval': 'blue',          # Blue for leadership
            'caring': 'white',           # White for neutral and factual
            'confusion': 'black',        # Black for logic
            'curiosity': 'green',        # Green for creative
            'desire': 'yellow',          # Yellow for optimistic
            'disappointment': 'black',   # Black for logic
            'disapproval': 'black',      # Black for logic
            'disgust': 'black',          # Black for logic
            'embarrassment': 'white',    # White for neutral and factual
            'excitement': 'green',       # Green for creative
            'fear': 'red',               # Red for emotional
            'gratitude': 'yellow',       # Yellow for optimistic
            'grief': 'red',              # Red for emotional
            'joy': 'yellow',             # Yellow for optimistic
            'love': 'yellow',            # Yellow for optimistic
            'nervousness': 'red',        # Red for emotional
            'optimism': 'yellow',        # Yellow for optimistic
            'pride': 'blue',             # Blue for leadership
            'realization': 'green',      # Green for creative
            'relief': 'yellow',          # Yellow for optimistic
            'remorse': 'red',            # Red for emotional
            'sadness': 'red',            # Red for emotional
            'surprise': 'green',         # Green for creative
            'neutral': 'white',          # White for neutral and factual
        }

        # Get the color of the hat based on the predicted label
        hat_color = emotion_to_hat[config.id2label[predicted_label]]
        print('Hat Color:', hat_color)

        # Implement this when dashboard is ready
        # return render_template('dashboard.html', hat_color=hat_color)

if __name__ == '__main__':
    app.run(debug=True)
