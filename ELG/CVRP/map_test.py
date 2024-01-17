import vrplib
import gmaps
import googlemaps
from ipywidgets.embed import embed_minimal_html
from flask import Flask, render_template

app = Flask(__name__)

def idx_to_coord_route(coord_list, sol):
    coord_route = []
    for idx in sol:
        coord_route.append(coord_list[idx])

    return coord_route


def get_google_maps_api_key():
    # Replace 'YOUR_API_KEY' with your actual API key
    return 'AIzaSyARJrp_WoQdboYmRwckwVaBA1WRGjkf8bQ'


def plot_route_on_google_maps(api_key, start_coord, end_coord, waypoints=None):
    gmaps_api = googlemaps.Client(key=api_key)

    # Get directions using Google Maps Directions API
    directions_result = gmaps_api.directions(
        start_coord,
        end_coord,
        mode="driving",
        waypoints=waypoints,
        optimize_waypoints=True
    )

    # Extract the route information
    route = directions_result[0]['legs'][0]['steps']

    # Extract coordinates from each step in the route
    coordinates = [(step['start_location']['lat'], step['start_location']['lng']) for step in route]
    coordinates.append((route[-1]['end_location']['lat'], route[-1]['end_location']['lng']))

    # Plot the route on Google Maps
    route_layer = gmaps.directions_layer(start_coord, end_coord, waypoints=coordinates)
    fig.add_layer(route_layer)

    return fig

@app.route('/')
def index():
    instance = vrplib.read_instance("ELG/CVRP/problem.vrp")

    sol = [0, 8, 2, 10, 1, 9, 0, 6, 7, 3, 0, 5, 4, 0]
    coord_list = instance["node_coord"]

    coord_route = idx_to_coord_route(coord_list, sol)

    # Replace these coordinates with your own
    start_coordinates = coord_route[0]  # San Francisco, CA
    end_coordinates = coord_route[-1]    # Los Angeles, CA

    api_key = get_google_maps_api_key()

    # Uncomment and modify the following line if you have waypoints
    waypoints = coord_route[1:-1]

    # fig = plot_route_on_google_maps(api_key, start_coordinates, end_coordinates, waypoints=waypoints_coordinates)
    fig = plot_route_on_google_maps(api_key, start_coordinates, end_coordinates, waypoints)

    save_map_as_html(fig, filename='route_map.html')

    return render_template('route_map.html', google_maps_api_key=api_key)


if __name__ == "__main__":
    app.run(debug=True)

