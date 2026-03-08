from flask import Flask, render_template, request
from rl_model import train_q_learning, generate_folium_map, LOCATIONS

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    optimal_path = None
    map_ready = False
    
    if request.method == 'POST':
        start_node = int(request.form.get('start_node', 0))
        
        # Check if the user wants an open loop or closed loop
        return_to_start = 'return_start' in request.form
        
        selected_stops = request.form.getlist('stops')
        selected_ids = [int(stop_id) for stop_id in selected_stops]
        
        if start_node not in selected_ids:
            selected_ids.append(start_node)
        
        # Pass the toggle setting to the RL model
        optimal_path = train_q_learning(
            start_node_id=start_node, 
            selected_ids=selected_ids, 
            return_to_start=return_to_start
        )
        
        if len(optimal_path) > 1:
            generate_folium_map(optimal_path)
            map_ready = True
        
    return render_template('index.html', locations=LOCATIONS, path=optimal_path, map_ready=map_ready)

@app.route('/map')
def map_view():
    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)