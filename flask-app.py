from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import subprocess
import threading
import os

app = Flask(__name__)
app.secret_key = 'dsc180b'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_coordinates', methods=['POST'])
def add_coordinates():
    lat1 = request.form['lat1']
    lon1 = request.form['lon1']
    lat2 = request.form['lat2']
    lon2 = request.form['lon2']

    # Run the external script with user input
    thread = threading.Thread(target=run_pole_workflow, args=(lat1, lon1, lat2, lon2))
    thread.start()
    
    flash('Processing...', 'info')
    return redirect(url_for('index'))
   # return render_template('index.html')

def run_pole_workflow(lat1, lon1, lat2, lon2):
    # Build the command
    command = ['python3', 'entire_workflow/pole_workflow.py', f'{lat1},{lon1}', f'{lat2},{lon2}']

    try:
        # Run the command and capture the output
        print("Running Script")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        flash('Workflow completed successfully', 'success')
        return redirect('index')
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

if __name__ == '__main__':
    app.run(debug=True)
