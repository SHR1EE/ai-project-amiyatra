import numpy as np
import random
import requests
import folium
import os

LOCATIONS = {
    0: {"name": "Amity University (Start)", "coords": (28.5443, 77.3330)},
    1: {"name": "Sector 44 Food Market", "coords": (28.5540, 77.3350)},
    2: {"name": "Gardens Galleria Plaza", "coords": (28.5676, 77.3256)},
    3: {"name": "Okhla Bird Sanctuary", "coords": (28.5562, 77.3045)},
    4: {"name": "Noida Stadium Sports Complex", "coords": (28.5866, 77.3400)},
    5: {"name": "DLF Mall of India", "coords": (28.5677, 77.3211)},
    6: {"name": "Worlds of Wonder Water Park", "coords": (28.5638, 77.3209)},
    7: {"name": "Brahmaputra Street Market", "coords": (28.5800, 77.3330)},
    8: {"name": "Botanic Garden of Indian Republic", "coords": (28.5645, 77.3340)},
    9: {"name": "Rashtriya Dalit Prerna Sthal", "coords": (28.5724, 77.3005)},
    10: {"name": "Cafe Delhi Heights", "coords": (28.5680, 77.3215)},
    11: {"name": "ISKCON Temple Noida", "coords": (28.5989, 77.3415)},
    12: {"name": "Stupa 18 Art Gallery", "coords": (28.5303, 77.3670)},
    13: {"name": "PVR Superplex Logix City", "coords": (28.5740, 77.3522)},
    14: {"name": "Snow World Noida", "coords": (28.5675, 77.3210)},
    15: {"name": "Atta Market (Sector 27)", "coords": (28.5702, 77.3235)},
    16: {"name": "Paul French Bakery & Cafe", "coords": (28.5676, 77.3212)},
    17: {"name": "Noida Chess Academy", "coords": (28.5822, 77.3650)},
    18: {"name": "National Science Centre & Planetarium", "coords": (28.6132, 77.2455)},
    19: {"name": "Goonj NGO Processing Center", "coords": (28.5340, 77.2880)},
    20: {"name": "Noida Golf Course", "coords": (28.5638, 77.3465)},
    21: {"name": "Vedic Astrology Research Center", "coords": (28.5850, 77.3150)},
    22: {"name": "Bikanerwala Sector 18", "coords": (28.5710, 77.3245)},
    23: {"name": "KidZania Delhi NCR", "coords": (28.5630, 77.3210)},
    24: {"name": "Smaaash Entertainment", "coords": (28.5678, 77.3218)},
    25: {"name": "The Great India Place (GIP)", "coords": (28.5635, 77.3255)}
}

TOMTOM_API_KEY = "0W1yc6dHB2aNThJIXv4DJnWxBEa9e8ES"

def get_travel_time(coord1, coord2):
    """Get precise travel time using optimized routing parameters."""
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{coord1[0]},{coord1[1]}:{coord2[0]},{coord2[1]}/json?"
        f"routeType=fastest&traffic=true&travelMode=car&key={TOMTOM_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['routes'][0]['summary']['travelTimeInSeconds']
    except Exception:
        # Fallback straight-line distance heuristic if API fails
        dist = np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)
        return int(dist * 100000)

def get_traffic_color(coord):
    """Determine line color based on live traffic congestion ratio."""
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={coord[0]},{coord[1]}&unit=KMPH&key={TOMTOM_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return '#3498db' # Default blue
        
        data = response.json()
        segment = data.get('flowSegmentData', {})
        current_speed = segment.get('currentSpeed', 0)
        free_flow_speed = segment.get('freeFlowSpeed', 0)
        
        if free_flow_speed <= 0 or current_speed <= 0:
            return '#3498db' # Default blue
        
        ratio = current_speed / free_flow_speed
        if ratio >= 0.85:
            return '#2ecc71' # Green (Clear)
        elif ratio >= 0.5:
            return '#f39c12' # Orange (Moderate)
        else:
            return '#e74c3c' # Red (Heavy Traffic)
    except Exception as e:
        print(f"Traffic API exception: {str(e)}")
        return '#3498db' # Default blue

def get_route_leg_geometry(coord1, coord2):
    """Get the physical street geometry for a single leg of the journey."""
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{coord1[0]},{coord1[1]}:{coord2[0]},{coord2[1]}/json?"
        f"routeType=fastest&traffic=true&travelMode=car&key={TOMTOM_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('routes'):
            return [coord1, coord2]
            
        route_points = []
        for leg in data['routes'][0]['legs']:
            for point in leg['points']:
                route_points.append([point['latitude'], point['longitude']])
        return route_points
    except Exception as e:
        print(f"Routing API error: {str(e)}")
        return [coord1, coord2]

def build_environment(active_locations):
    n = len(active_locations)
    reward_matrix = np.full((n, n), -10000)
    for i in range(n):
        for j in range(n):
            if i != j:
                time = get_travel_time(active_locations[i]['coords'], active_locations[j]['coords'])
                reward_matrix[i][j] = -time
    return reward_matrix

def train_q_learning(start_node_id=0, selected_ids=None, return_to_start=True, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    if selected_ids is None or len(selected_ids) == 0:
        selected_ids = list(LOCATIONS.keys())
        
    if start_node_id not in selected_ids:
        selected_ids.insert(0, start_node_id)
        
    selected_ids = list(dict.fromkeys(selected_ids))
    active_locations = [LOCATIONS[node_id] for node_id in selected_ids]
    n = len(active_locations)
    local_start_node = selected_ids.index(start_node_id)
    
    if n <= 1:
        return [active_locations[0]]
        
    q_table = np.zeros((n, n))
    reward_matrix = build_environment(active_locations)
    
    for _ in range(episodes):
        state = random.randint(0, n-1)
        visited = [state]
        
        while len(visited) < n:
            if random.uniform(0, 1) < epsilon:
                action = random.choice([x for x in range(n) if x not in visited])
            else:
                q_values = q_table[state].copy()
                q_values[visited] = -np.inf
                action = np.argmax(q_values)
                if action in visited:
                    action = random.choice([x for x in range(n) if x not in visited])

            future_rewards = [q_table[action][x] for x in range(n) if x not in visited + [action]]
            next_max = np.max(future_rewards) if future_rewards else 0
            
            q_table[state, action] = q_table[state, action] + alpha * (
                reward_matrix[state, action] + gamma * next_max - q_table[state, action]
            )
            state = action
            visited.append(state)
            
    current_state = local_start_node
    optimal_path = [current_state]
    
    while len(optimal_path) < n:
        q_values = q_table[current_state].copy()
        q_values[optimal_path] = -np.inf
        next_state = np.argmax(q_values)
        optimal_path.append(next_state)
        current_state = next_state
        
    if return_to_start:
        optimal_path.append(local_start_node)
    
    return [active_locations[i] for i in optimal_path]

def generate_folium_map(optimal_path):
    start_coords = optimal_path[0]['coords']
    m = folium.Map(location=start_coords, zoom_start=13)
    
    # Raster traffic layer (set to show=False by default so it doesn't clutter)
    traffic_tile_url = f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}"
    folium.TileLayer(
        tiles=traffic_tile_url,
        attr="TomTom Traffic",
        name="Live Traffic Flow (Raster)",
        overlay=True,
        control=True,
        show=False 
    ).add_to(m)
    
    # 1. Add Markers
    for idx, node in enumerate(optimal_path):
        is_last_node = (idx == len(optimal_path) - 1)
        is_closed_loop = (optimal_path[0]['coords'] == optimal_path[-1]['coords'])
        
        if is_last_node and is_closed_loop and len(optimal_path) > 1:
            continue
            
        if idx == 0:
            color, icon_name, popup_text = 'red', 'home', f"Start: {node['name']}"
        elif is_last_node and not is_closed_loop:
            color, icon_name, popup_text = 'green', 'info-sign', f"End: {node['name']}"
        else:
            color, icon_name, popup_text = 'blue', 'info-sign', f"Stop {idx}: {node['name']}"

        folium.Marker(
            location=node['coords'],
            popup=popup_text,
            icon=folium.Icon(color=color, icon=icon_name)
        ).add_to(m)
        
    # 2. Draw Multi-Colored Path
    all_points = []
    for i in range(len(optimal_path) - 1):
        coord1 = optimal_path[i]['coords']
        coord2 = optimal_path[i+1]['coords']
        
        # Get geometry for this specific leg
        leg_coords = get_route_leg_geometry(coord1, coord2)
        all_points.extend(leg_coords)
        
        # Get traffic color for this leg
        leg_color = get_traffic_color(coord1)
        
        # Draw the colored segment
        folium.PolyLine(
            leg_coords, 
            color=leg_color, 
            weight=6, 
            opacity=0.9,
            tooltip="Live Traffic Segment"
        ).add_to(m)
    
    # Snap bounding box
    if all_points:
        m.fit_bounds(all_points)
    
    folium.LayerControl().add_to(m)
    
    os.makedirs('templates', exist_ok=True)
    m.save('templates/map.html')