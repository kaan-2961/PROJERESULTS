import math
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import elkai  # Updated import
import os
import contextily as ctx  # for adding basemaps

# Global parameters (feel free to adjust)
TIME_LIMIT = 3      # Maximum allowed tour time (hours)
SPEED = 35          # km per hour
SERVICE_TIME = 0.05 # hours per stop
DISTANCE_COST_RATE = 2  # cost per km
HIRING_COST = 50        # cost per cluster

# -----------------------------
# Utility functions (TSP, haversine, etc.)
# -----------------------------
def haversine(coord1, coord2):
    """Calculate the haversine distance (in km) between two (lat, lon) pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth’s radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def tsp_solver_elkai(coordinates, customer_indices):
    """
    Solve the TSP using the elkai library.
    Returns the total travel time (hours) along the optimal route 
    plus the service time per customer.
    """
    if not customer_indices:
        return 0.0

    # Special handling for clusters with one customer (results in 2x2 matrix)
    if len(customer_indices) == 1:
        customer = coordinates[customer_indices[0]]
        distance = haversine(coordinates[0], customer)
        travel_time = (distance * 2) / SPEED  # round-trip
        return travel_time + SERVICE_TIME

    # Build a list of nodes: depot (index 0) and customer nodes.
    indices = [0] + customer_indices
    n = len(indices)
    # Create distance matrix using haversine
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = haversine(coordinates[indices[i]], coordinates[indices[j]])
    # Convert distance matrix to integers (in meters)
    int_dist_matrix = [[int(d * 1000) for d in row] for row in dist_matrix]
    # Solve TSP using elkai's integer solver
    route = elkai.solve_int_matrix(int_dist_matrix)
    # Recompute total distance using the original (float) distances
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i]][route[i + 1]]
    total_distance += dist_matrix[route[-1]][route[0]]
    # Add service time for each customer (exclude depot)
    service_time_total = SERVICE_TIME * len(customer_indices)
    travel_time = total_distance / SPEED
    return travel_time + service_time_total

def tsp_distance_elkai(coordinates, customer_indices):
    """
    Solve the TSP using the elkai library and return the total route distance (km).
    """
    if not customer_indices:
        return 0.0

    if len(customer_indices) == 1:
        customer = coordinates[customer_indices[0]]
        return haversine(coordinates[0], customer) * 2

    indices = [0] + customer_indices
    n = len(indices)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = haversine(coordinates[indices[i]], coordinates[indices[j]])
    int_dist_matrix = [[int(d * 1000) for d in row] for row in dist_matrix]
    route = elkai.solve_int_matrix(int_dist_matrix)
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i]][route[i + 1]]
    total_distance += dist_matrix[route[-1]][route[0]]
    return total_distance

def tsp_solver(coordinates, customer_indices):
    """
    A simple TSP solver using a nearest neighbor heuristic.
    The depot (index 0) is prepended and appended.
    Returns total travel time (hours), including SERVICE_TIME per customer.
    """
    if not customer_indices:
        return 0.0
    remaining = set(customer_indices)
    current = 0
    total_time = 0.0
    while remaining:
        next_node = min(remaining, key=lambda x: haversine(coordinates[current], coordinates[x]))
        travel_time = haversine(coordinates[current], coordinates[next_node]) / SPEED
        total_time += travel_time + SERVICE_TIME
        current = next_node
        remaining.remove(next_node)
    total_time += haversine(coordinates[current], coordinates[0]) / SPEED
    return total_time

def tsp_distance(coordinates, customer_indices):
    """
    Similar to tsp_solver but sums actual distances (in km) along the nearest-neighbor route.
    """
    if not customer_indices:
        return 0.0
    remaining = set(customer_indices)
    current = 0
    total_distance = 0.0
    while remaining:
        next_node = min(remaining, key=lambda x: haversine(coordinates[current], coordinates[x]))
        total_distance += haversine(coordinates[current], coordinates[next_node])
        current = next_node
        remaining.remove(next_node)
    total_distance += haversine(coordinates[current], coordinates[0])
    return total_distance

def compute_route_time_for_sequence(coordinates, sequence):
    """Compute the route time (in hours) for a given sequence (with depot at start and end)."""
    route = [0] + sequence + [0]
    total_time = 0.0
    for i in range(len(route)-1):
        total_time += haversine(coordinates[route[i]], coordinates[route[i+1]]) / SPEED
        if route[i+1] != 0:
            total_time += SERVICE_TIME
    return total_time

# -----------------------------
# Clustering functions (k-means, NN, rand NN)
# -----------------------------
class Cluster:
    """Stores cluster ID, method used, list of customer node indices, and TSP route time."""
    def __init__(self, cluster_id, method, nodes, route_time):
        self.cluster_id = cluster_id
        self.method = method
        self.nodes = nodes
        self.route_time = route_time

    def __repr__(self):
        return f"Cluster(id={self.cluster_id}, method={self.method}, nodes={self.nodes}, time={self.route_time:.2f})"

def kmeans_clusters(coordinates, k_range):
    clusters_list = []
    cluster_id = 0
    # Exclude depot (index 0)
    customers = np.array(coordinates[1:])
    customer_indices = np.arange(1, len(coordinates))
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(customers)
        labels = kmeans.labels_
        for label in range(k):
            cluster_nodes = customer_indices[labels == label].tolist()
            # Use the elkai-based TSP solver here
            route_time = tsp_solver_elkai(coordinates, cluster_nodes)
            if route_time > TIME_LIMIT and len(cluster_nodes) > 1:
                sub_customers = np.array([coordinates[i] for i in cluster_nodes])
                sub_indices = np.array(cluster_nodes)
                kmeans_sub = KMeans(n_clusters=2, random_state=42).fit(sub_customers)
                sub_labels = kmeans_sub.labels_
                for sub_label in [0, 1]:
                    sub_cluster_nodes = sub_indices[sub_labels == sub_label].tolist()
                    sub_route_time = tsp_solver_elkai(coordinates, sub_cluster_nodes)
                    clusters_list.append(Cluster(cluster_id, f"kmeans_split_from_{k}", sub_cluster_nodes, sub_route_time))
                    cluster_id += 1
            else:
                clusters_list.append(Cluster(cluster_id, f"kmeans_{k}", cluster_nodes, route_time))
                cluster_id += 1
    return clusters_list

def nn_clusters(coordinates):
    clusters_list = []
    cluster_id = 0
    unassigned = set(range(1, len(coordinates)))
    while unassigned:
        current_cluster = []
        improved = True
        while improved:
            best_candidate = None
            best_time = None
            for candidate in list(unassigned):
                test_seq = current_cluster + [candidate]
                t_time = compute_route_time_for_sequence(coordinates, test_seq)
                if t_time <= TIME_LIMIT:
                    if best_candidate is None or t_time < best_time:
                        best_candidate = candidate
                        best_time = t_time
            if best_candidate is not None:
                current_cluster.append(best_candidate)
                unassigned.remove(best_candidate)
            else:
                improved = False
        if current_cluster:
            route_time = tsp_solver(coordinates, current_cluster)
            clusters_list.append(Cluster(cluster_id, "nn", current_cluster, route_time))
            cluster_id += 1
        else:
            break
    return clusters_list

def rand_nn_clusters(coordinates):
    clusters_list = []
    cluster_id = 0
    unassigned = set(range(1, len(coordinates)))
    while unassigned:
        current_cluster = []
        improved = True
        while improved:
            candidates = []
            for candidate in list(unassigned):
                test_seq = current_cluster + [candidate]
                t_time = compute_route_time_for_sequence(coordinates, test_seq)
                if t_time <= TIME_LIMIT:
                    candidates.append((candidate, t_time))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                top_candidates = candidates[:3] if len(candidates) >= 3 else candidates
                chosen_candidate, chosen_time = random.choice(top_candidates)
                current_cluster.append(chosen_candidate)
                unassigned.remove(chosen_candidate)
            else:
                improved = False
        if current_cluster:
            route_time = tsp_solver(coordinates, current_cluster)
            clusters_list.append(Cluster(cluster_id, "rand_nn", current_cluster, route_time))
            cluster_id += 1
        else:
            break
    return clusters_list

# -----------------------------
# Set covering & overlap removal
# -----------------------------
def set_covering(clusters, num_customers):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Solver not created!")
        return None
    x = {}
    for cluster in clusters:
        x[cluster.cluster_id] = solver.BoolVar(f"x_{cluster.cluster_id}")
    for j in range(1, num_customers + 1):
        covering = [cluster for cluster in clusters if j in cluster.nodes]
        if covering:
            solver.Add(sum(x[cluster.cluster_id] for cluster in covering) >= 1)
        else:
            print(f"Customer {j} is not covered by any cluster!")
    solver.Minimize(sum(x[cluster.cluster_id] for cluster in clusters))
    status = solver.Solve()
    selected = []
    if status == pywraplp.Solver.OPTIMAL:
        for cluster in clusters:
            if x[cluster.cluster_id].solution_value() > 0.5:
                selected.append(cluster)
    else:
        print("No optimal solution found!")
    return selected

def remove_overlapping_nodes(clusters, coordinates):
    node_to_clusters = {}
    for cluster in clusters:
        for node in cluster.nodes:
            node_to_clusters.setdefault(node, []).append(cluster)
    for node, clust_list in node_to_clusters.items():
        if len(clust_list) > 1:
            best = min(clust_list, key=lambda c: c.route_time)
            for clust in clust_list:
                if clust != best and node in clust.nodes:
                    clust.nodes.remove(node)
                    if clust.method.startswith("kmeans"):
                        clust.route_time = tsp_solver_elkai(coordinates, clust.nodes)
                    else:
                        clust.route_time = compute_route_time_for_sequence(coordinates, clust.nodes)
    clusters = [clust for clust in clusters if clust.nodes]
    return clusters

def get_final_cluster_coordinates(final_clusters, coordinates):
    """Return nested list of coordinate pairs for each final cluster,
    ensuring that the depot (coordinates[0]) is the first element in every cluster.
    """
    nested = []
    for cluster in final_clusters:
        cluster_coords = [coordinates[0]] + [coordinates[node] for node in cluster.nodes]
        nested.append(cluster_coords)
    return nested

# -----------------------------
# Plotting functions
# -----------------------------
def plot_input_coordinates(coordinates, filename=None):
    plt.figure(figsize=(8, 6))
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    plt.scatter(lons[1:], lats[1:], color='blue', label='Customers', s=50)
    plt.scatter(coordinates[0][1], coordinates[0][0], color='red', marker='*', s=200, label='Depot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Original Input Coordinates')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_clusters(nested_list, depot=None, filename=None):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20', len(nested_list))
    for i, cluster_coords in enumerate(nested_list):
        lats = [coord[0] for coord in cluster_coords]
        lons = [coord[1] for coord in cluster_coords]
        plt.scatter(lons, lats, color=cmap(i), label=f'Cluster {i}', alpha=0.7, s=50)
    if depot:
        plt.scatter(depot[1], depot[0], color='red', marker='*', s=200, label='Depot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Final Clusters (Overlay)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

# -----------------------------
# Cost calculation and CSV writing for final clusters
# -----------------------------
def compute_cost_details(final_clusters, coordinates):
    data = []
    overall_cost = 0.0
    overall_distance = 0.0
    overall_time = 0.0
    for cluster in final_clusters:
        route_dist = tsp_distance(coordinates, cluster.nodes)
        travel_time = route_dist / SPEED + SERVICE_TIME * len(cluster.nodes)
        distance_cost = route_dist * DISTANCE_COST_RATE
        total_cost = distance_cost + HIRING_COST
        overall_cost += total_cost
        overall_distance += route_dist
        overall_time += travel_time
        data.append({
            'Cluster ID': cluster.cluster_id,
            'Method': cluster.method,
            'Num Nodes': len(cluster.nodes),
            'Distance (km)': round(route_dist, 2),
            'Time (hrs)': round(travel_time, 2),
            'Distance Cost': round(distance_cost, 2),
            'Hiring Cost': HIRING_COST,
            'Total Cluster Cost': round(total_cost, 2)
        })
    df = pd.DataFrame(data)
    return df, overall_distance, overall_cost, overall_time

def write_final_clusters_csv(final_clusters, coordinates, part_number):
    nested_list = get_final_cluster_coordinates(final_clusters, coordinates)
    df = pd.DataFrame({'Final Cluster Coordinates': [str(cluster) for cluster in nested_list]})
    csv_filename = f"set_covering_osm_data_part{part_number}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# -----------------------------
# Processing an input CSV file (one part)
# -----------------------------
def process_input_file(input_csv, part_number, k_range):
    # Read coordinates from CSV (assume two columns: lat, lon)
    df_in = pd.read_csv(input_csv)
    coordinates = df_in.values.tolist()
    
    # Generate and save input coordinates plot
    input_plot_file = f"input_coordinates_part{part_number}.png"
    plot_input_coordinates(coordinates, filename=input_plot_file)
    
    # Generate clusters from each method
    clusters_km = kmeans_clusters(coordinates, k_range)
    clusters_nn = nn_clusters(coordinates)
    clusters_rand = rand_nn_clusters(coordinates)
    all_clusters = clusters_km + clusters_nn + clusters_rand
    # Set covering over customer indices 1..n-1
    selected_clusters = set_covering(all_clusters, len(coordinates) - 1)
    # Remove overlapping nodes
    final_clusters = remove_overlapping_nodes(selected_clusters, coordinates)
    
    # Plot final clusters (overlay) and save image
    nested_list = get_final_cluster_coordinates(final_clusters, coordinates)
    clusters_plot_file = f"final_clusters_part{part_number}.png"
    plot_clusters(nested_list, depot=coordinates[0], filename=clusters_plot_file)
    
    # Compute cost details (including overall travel time) and overall cost & distance
    cost_df, overall_distance, overall_cost, overall_time = compute_cost_details(final_clusters, coordinates)
    # Write final clusters CSV file
    final_csv_file = write_final_clusters_csv(final_clusters, coordinates, part_number)
    
    summary = {
        'part_number': part_number,
        'input_csv': input_csv,
        'num_clusters': len(final_clusters),
        'overall_distance': overall_distance,
        'overall_cost': overall_cost,
        'overall_time': overall_time,
        'cost_df': cost_df,
        'methods': [cluster.method for cluster in final_clusters],
        'input_plot_file': input_plot_file,
        'clusters_plot_file': clusters_plot_file,
        'final_clusters_csv': final_csv_file
    }
    return summary

# -----------------------------
# Main processing: iterate over multiple input CSV files and create one Excel summary file.
# -----------------------------
def main_multi(input_csv_files, k_range):
    summaries = []
    for idx, csv_file in enumerate(input_csv_files, start=1):
        print(f"Processing file: {csv_file}")
        summary = process_input_file(csv_file, part_number=idx, k_range=k_range)
        summaries.append(summary)
    
    excel_filename = "set_covering_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for summary in summaries:
            sheet_name = f"PART {summary['part_number']}"
            summary_info = pd.DataFrame({
                'Input CSV': [summary['input_csv']],
                'Number of Clusters': [summary['num_clusters']],
                'Overall Distance (km)': [summary['overall_distance']],
                'Overall Cost ($)': [summary['overall_cost']],
                'Overall Time (hrs)': [summary['overall_time']]
            })
            summary_info.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            summary['cost_df'].to_excel(writer, sheet_name=sheet_name, startrow=5, index=False)
            
            worksheet = writer.sheets[sheet_name]
            worksheet.write('A12', f"Input Coordinates Plot (Part {summary['part_number']})")
            worksheet.insert_image('A13', summary['input_plot_file'], {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.write('H12', f"Final Clusters Plot (Part {summary['part_number']})")
            worksheet.insert_image('H13', summary['clusters_plot_file'], {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.write('A40', f"Final Clusters CSV File (Part {summary['part_number']})")
            worksheet.write('A41', summary['final_clusters_csv'])
    
    print(f"Excel summary file written: {excel_filename}")

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    input_csv_files = [
        "set_covering_osm_data_part1.csv",
        "set_covering_osm_data_part2.csv",
        "set_covering_osm_data_part3.csv",
        "set_covering_osm_data_part4.csv",
        "set_covering_osm_data_part5.csv",
        "set_covering_osm_data_part6.csv",
        "set_covering_osm_data_part7.csv",
        "set_covering_osm_data_part8.csv",
        "set_covering_osm_data_part9.csv",
        "set_covering_osm_data_part10.csv",
        "set_covering_osm_data_part11.csv",
        "set_covering_osm_data_part12.csv",
    ]
    # Adjust the k_range as needed (for k-means clustering)
    main_multi(input_csv_files, k_range=(17, 35))
