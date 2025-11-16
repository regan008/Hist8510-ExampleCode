# Simple Flask App with Plotly.js Visualization

This is a simple Flask application that visualizes Philadelphia venue data using Plotly.js.

## What this app does

- Reads data from `data_philly.csv`
- Displays three interactive visualizations:
  - **Location Map**: Scatter plot showing venue locations on a map
  - **Venues by Type**: Bar chart showing the distribution of venue types
  - **Venues by Year**: Bar chart showing the number of venues per year

## How to run

1. Install Flask:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

## Features

- Interactive Plotly.js visualizations
- Map visualization using OpenStreetMap
- Bar charts for categorical and temporal analysis
- Responsive design
- Clean, simple interface

## Files

- `app.py`: Main Flask application
- `templates/index.html`: HTML template with Plotly.js visualizations
- `data_philly.csv`: Data file with Philadelphia venue information
- `requirements.txt`: Python dependencies

## Key concepts demonstrated

- Flask routing and API endpoints
- CSV data loading
- JSON API for data access
- Plotly.js for interactive visualizations
- Client-side data processing and visualization

