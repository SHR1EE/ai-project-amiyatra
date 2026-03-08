📍 AMIyatra: AI-Powered Path & Traffic Optimizer
AMIyatra is a smart routing and traffic optimization engine designed for the Amity NCR region. It solves a dynamic, real-world variation of the Traveling Salesperson Problem (TSP) by deploying a Reinforcement Learning (RL) agent to find the optimal sequence of destinations while actively avoiding live traffic congestion.

🚀 Key Features
Reinforcement Learning Core: Utilizes a model-free Q-Learning algorithm to determine optimal multi-stop routes.

Live Traffic Integration: Connects to the TomTom Traffic API to fetch real-time currentSpeed and freeFlowSpeed data, actively calculating congestion penalties.

Dynamic Route Color-Coding: Maps are generated with segment-by-segment traffic visualization (Green for clear, Orange for moderate, Red for heavy traffic).

Interactive Mapping: Powered by Folium, featuring physical road-snapping geometry, bounding box camera focusing, and custom destination markers.

Customizable Itineraries: Supports open-loop (A to B) and closed-loop (return to start) routing across 25+ prominent locations in the Noida/NCR region.

🛠️ Technology Stack
Backend: Python 3, Flask

AI & Mathematics: NumPy, standard Reinforcement Learning paradigms

Frontend: HTML5, CSS3, JavaScript

Mapping & Visualization: Folium, Leaflet.js

External APIs: * TomTom Routing API

TomTom Traffic Flow Segment Data API

🧠 How the AI Works
The Markov Decision Process (MDP)
The routing problem is framed as an MDP where:

Agent: A virtual driver navigating the map.

State (s): The current location/node the agent occupies.

Action (a): The decision to transition to an unvisited location.

Reward (R): The negative travel time in seconds. By maximizing the reward, the agent inherently minimizes the total travel time.

Q-Learning & The Bellman Equation
The agent trains over 500 episodes using an ϵ-greedy exploration strategy (default ϵ=0.1). During training, the agent updates its knowledge base (the Q-Table) using the Bellman Equation:

Q(s,a)←Q(s,a)+α[R(s,a)+γ 
a 
′
 
max
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
Q(s,a): The current learned value of taking action a from state s.

α (Learning Rate = 0.1): Determines the weight of newly acquired information.

R(s,a): The immediate reward (live travel time penalty).

γ (Discount Factor = 0.9): Prioritizes long-term optimal routing over short-sighted immediate gains.

max 
a 
′
 
​
 Q(s 
′
 ,a 
′
 ): The maximum expected future reward from the next state.

⚙️ Installation & Setup
Prerequisites
Python 3.8+ installed on your system.

A valid TomTom Developer API Key.

1. Clone the Repository
Bash
git clone https://github.com/yourusername/amiyatra.git
cd amiyatra
2. Set Up a Virtual Environment (Recommended)
To avoid package conflicts (especially on macOS environments), create and activate a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure the API Key
Open rl_model.py and replace the placeholder with your actual TomTom API key:

Python
TOMTOM_API_KEY = "your_api_key_here"
5. Launch the Application
Start the Flask development server:

Bash
python3 app.py
Open your web browser and navigate to http://127.0.0.1:5000.

📂 Project Structure
Plaintext
amiyatra/
│
├── app.py                 # Flask server and application routing
├── rl_model.py            # Q-Learning logic, environment setup, and API calls
├── requirements.txt       # Python dependencies
└── templates/
    ├── index.html         # Frontend UI (Controls, Selectors, Display)
    └── map.html           # Auto-generated Folium map overlay
🔮 Future Roadmap
Deep Q-Networks (DQN): Transition from a discrete Q-Table to a neural network to scale the application to handle hundreds of simultaneous nodes.

Time-Window Constraints: Add functionality to account for location opening/closing hours.

Multi-Agent Routing: Expand the algorithm to manage a fleet of vehicles rather than a single agent.
