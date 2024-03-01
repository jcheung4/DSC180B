from flask import Flask, render_template, request, jsonify
import subprocess
import threading

app = Flask(__name__)

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
    
    return render_template('index.html')

    #return jsonify({'status': 'Processing the request in the background'})

    #return render_template('index.html', result=result)

def run_pole_workflow(lat1, lon1, lat2, lon2):
    # Build the command
    command = ['python3', 'entire_workflow/pole_workflow.py', f'{lat1},{lon1}', f'{lat2},{lon2}']

    try:
        # Run the command and capture the output
        print("Running Script")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

if __name__ == '__main__':
    app.run(debug=True)
