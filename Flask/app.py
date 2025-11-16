"""
Simple Flask App with Plotly.js Visualization
==============================================

This app reads data from data_philly.csv and visualizes it using plotly.js.

How to run:
1. Install Flask: pip install -r requirements.txt
2. Run: python app.py
3. Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify
import csv
import os

app = Flask(__name__)

# Path to the CSV file
CSV_FILE = 'data_philly.csv'

def load_data():
    """
    Load data from the CSV file and return it as a list of dictionaries.
    """
    data = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    return data

@app.route('/')
def index():
    """
    Home page - displays the visualization
    """
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """
    API endpoint that returns the CSV data as JSON
    """
    data = load_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

