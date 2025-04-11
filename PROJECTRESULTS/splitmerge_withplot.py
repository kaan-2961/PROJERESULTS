import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import copy
import random
import math
import ast
from haversine import haversine
from sklearn.cluster import KMeans
import elkai
import os
import contextily as ctx  # for adding basemaps

# -----------------------------
# GLOBAL PARAMETERS & RUN SETTINGS
# -----------------------------
speed_km_per_hr = 35
service_time_hr = 0.05
tmax = 3
hiring_cost_per_cluster = 50
distance_cost_per_km = 2

# Initial run parameters (not used for final output)
max_merge_attempts_per_cluster = 20
max_iterations = 100
FIRST_RUN_NUM_RUNS = 50

# Second run parameters (final output)
SECOND_RUN_MAX_MERGE_ATTEMPTS = 30
SECOND_RUN_MAX_ITERATIONS = 150
SECOND_RUN_NUM_RUNS = 30

# -----------------------------
# SPLIT–MERGE CLUSTERING CLASSES & FUNCTIONS
# -----------------------------
class Cluster:
    def __init__(self, data_points, cluster_id=None, color=None):
        # data_points: list of [lat, lon] pairs (first point is the depot)
        self.data_points = np.array(data_points)
        self.centroid = self.calculate_centroid()
        self.id = cluster_id
        self.time = None
        self.cost = None
        self.nearest_cluster_ids = None
        self.tour = None  # TSP route (list of indices)
        self.total_distance = None
        self.merge_attempts_remaining = max_merge_attempts_per_cluster
        self.attempts_left = 4
        if color is None:
            self.color = np.random.rand(3,)
        else:
            self.color = color

    def calculate_centroid(self):
        if not self.data_points.size:
            return np.array([0, 0])
        return np.mean(self.data_points, axis=0)

    def update_centroid(self):
        self.centroid = self.calculate_centroid()

    def set_time(self, time_value):
        self.time = time_value

    def set_cost(self, cost_value):
        self.cost = cost_value

    def set_tsp_result(self, tour, total_distance):
        self.tour = tour
        self.total_distance = total_distance

    def calculate_nearest_clusters(self, all_clusters, num_nearest=6):
        distances_to_other_clusters = []
        for other_cluster in all_clusters:
            if other_cluster.id != self.id:
                distance = haversine(self.centroid, other_cluster.centroid)
                distances_to_other_clusters.append((distance, other_cluster.id))
        distances_to_other_clusters.sort(key=lambda item: item[0])
        nearest_ids = [distances_to_other_clusters[i][1] for i in range(min(num_nearest, len(distances_to_other_clusters)))]
        self.nearest_cluster_ids = nearest_ids

    def decrement_merge_attempts(self):
        self.merge_attempts_remaining -= 1

    def decrement_attempts_left(self):
        self.attempts_left -= 1

    def get_attempts_left(self):
        return self.attempts_left

    def split_cluster_kmeans(self, n_clusters=2):
        if len(self.data_points) < n_clusters:
            return None
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(self.data_points)
        sub_clusters_points = [self.data_points[cluster_labels == i] for i in range(n_clusters)]
        sub_clusters = [Cluster(points, cluster_id=f"{self.id}_Sub{i+1}", color=self.color)
                        for i, points in enumerate(sub_clusters_points) if len(points) > 0]
        return sub_clusters

    def __repr__(self):
        return f"Cluster(ID={self.id}, Centroid={self.centroid.round(2)}, Points={len(self.data_points)}, Time={self.time}, Cost={self.cost})"

def calculate_total_distance_cluster_obj(cluster_obj, tour):
    points = cluster_obj.data_points
    total_distance = 0
    if tour:
        for i in range(len(tour) - 1):
            total_distance += haversine(points[tour[i]], points[tour[i+1]])
    return total_distance

def calculate_total_time_cluster_obj(cluster_obj, tour, speed_km_per_hr, service_time_hr):
    total_distance = calculate_total_distance_cluster_obj(cluster_obj, tour)
    travel_time = total_distance / speed_km_per_hr
    total_time = travel_time + len(tour) * service_time_hr
    return total_time

def calculate_total_cost_cluster_obj(cluster_obj, total_distance, hiring_cost_per_cluster, distance_cost_per_km):
    return hiring_cost_per_cluster + (total_distance * distance_cost_per_km)

def solve_tsp_elkai_constrained_cluster_obj(cluster_obj, tmax, speed_km_per_hr, service_time_hr):
    points = cluster_obj.data_points
    n_points = len(points)
    if n_points < 2:
        return None, None, None
    if n_points == 2:
        d1 = haversine(points[0], points[1])
        total_distance = d1 + haversine(points[1], points[0])
        total_time = total_distance / speed_km_per_hr + service_time_hr * 3
        tour = [0, 1, 0]
        return tour, total_distance, total_time

    distance_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                distance_matrix[i, j] = haversine(points[i], points[j])
    distance_matrix_int = np.round(distance_matrix * 1000).astype(int)

    try:
        tour = elkai.solve_int_matrix(distance_matrix_int)
    except RuntimeError:
        return None, None, None

    if 0 in tour:
        idx = tour.index(0)
        tour = tour[idx:] + tour[:idx]
    if tour[-1] != 0:
        tour.append(0)

    total_distance = calculate_total_distance_cluster_obj(cluster_obj, tour)
    total_time = calculate_total_time_cluster_obj(cluster_obj, tour, speed_km_per_hr, service_time_hr)
    if total_time <= tmax:
        return tour, total_distance, total_time
    else:
        return None, total_distance, total_time

def attempt_cluster_merge(base_cluster, merge_candidate_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    if (base_cluster.time is not None and merge_candidate_cluster.time is not None) and \
       (base_cluster.time + merge_candidate_cluster.time) > tmax:
        return False
    merged_data_points = np.concatenate((base_cluster.data_points, merge_candidate_cluster.data_points), axis=0)
    temp_cluster = Cluster(merged_data_points, cluster_id=f"TEMP_MERGE_{base_cluster.id}_{merge_candidate_cluster.id}")
    tour_result, distance_result, time_result = solve_tsp_elkai_constrained_cluster_obj(
        temp_cluster, tmax, speed_km_per_hr, service_time_hr)
    if tour_result and time_result <= tmax:
        base_cluster.data_points = merged_data_points
        base_cluster.update_centroid()
        base_cluster.set_tsp_result(tour_result, distance_result)
        base_cluster.set_time(time_result)
        base_cluster.set_cost(calculate_total_cost_cluster_obj(base_cluster, distance_result, hiring_cost_per_cluster, distance_cost_per_km))
        return True
    else:
        return False

def explore_split_branches(current_clusters, sub_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    best_branch_clusters = copy.deepcopy(current_clusters)
    min_cost = sum(c.cost for c in get_valid_clusters(best_branch_clusters, tmax))

    def recursive_merge(clusters_to_merge, remaining_sub_clusters):
        nonlocal min_cost, best_branch_clusters
        if not remaining_sub_clusters:
            temp_clusters = copy.deepcopy(clusters_to_merge)
            temp_clusters = run_merging_iterations(temp_clusters, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
            valid_clusters = get_valid_clusters(temp_clusters, tmax)
            if valid_clusters:
                current_cost = sum(c.cost for c in valid_clusters)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_branch_clusters = copy.deepcopy(temp_clusters)
            return

        first_sub_cluster = remaining_sub_clusters[0]
        rest_sub_clusters = remaining_sub_clusters[1:]
        branch1_clusters = copy.deepcopy(clusters_to_merge)
        branch1_clusters.append(first_sub_cluster)
        recursive_merge(branch1_clusters, rest_sub_clusters)
        branch2_clusters = copy.deepcopy(clusters_to_merge)
        branch2_clusters.append(first_sub_cluster)
        recursive_merge(branch2_clusters, rest_sub_clusters)

    recursive_merge(copy.deepcopy(current_clusters), sub_clusters)
    return best_branch_clusters

def attempt_split_merge(base_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    original_data_points = np.copy(base_cluster.data_points)
    sub_clusters = base_cluster.split_cluster_kmeans(n_clusters=2)
    if not sub_clusters or len(sub_clusters) < 2:
        return False
    initial_clusters_state_before_split = copy.deepcopy(all_clusters)
    best_branch_result_clusters = explore_split_branches(
        initial_clusters_state_before_split, sub_clusters, speed_km_per_hr, service_time_hr, tmax, history_log)
    initial_cost = sum(c.cost for c in get_valid_clusters(initial_clusters_state_before_split, tmax))
    final_cost_after_split_merge = sum(c.cost for c in get_valid_clusters(best_branch_result_clusters, tmax))
    if final_cost_after_split_merge < initial_cost:
        all_clusters.clear()
        all_clusters.extend(best_branch_result_clusters)
        return True
    else:
        base_cluster.data_points = original_data_points
        base_cluster.update_centroid()
        return False

def run_merging_iterations(clusters, iteration_count, speed_km_per_hr, service_time_hr, tmax, history_log):
    clusters_merged_in_iteration = []
    random.shuffle(clusters)
    cluster_index = 0
    while cluster_index < len(clusters):
        base_cluster = clusters[cluster_index]
        if base_cluster in clusters_merged_in_iteration:
            cluster_index += 1
            continue
        if base_cluster.merge_attempts_remaining <= 0 or base_cluster.get_attempts_left() <= 0:
            cluster_index += 1
            continue
        base_cluster.calculate_nearest_clusters(clusters)
        nearest_cluster_ids = base_cluster.nearest_cluster_ids or []
        direct_merge_attempted_in_iteration = False
        for nearest_cluster_id in nearest_cluster_ids[:3]:
            merge_candidate_cluster = next((c for c in clusters if c.id == nearest_cluster_id), None)
            if merge_candidate_cluster and merge_candidate_cluster not in clusters_merged_in_iteration:
                if attempt_cluster_merge(base_cluster, merge_candidate_cluster, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                    clusters_merged_in_iteration.append(merge_candidate_cluster)
                    base_cluster.decrement_merge_attempts()
                    base_cluster.decrement_attempts_left()
                    merge_candidate_cluster.decrement_attempts_left()
                    direct_merge_attempted_in_iteration = True
                    break
                else:
                    base_cluster.decrement_merge_attempts()
                    base_cluster.decrement_attempts_left()
                    merge_candidate_cluster.decrement_attempts_left()
                    direct_merge_attempted_in_iteration = True
        if not direct_merge_attempted_in_iteration and len(base_cluster.data_points) > 1 and base_cluster.get_attempts_left() > 0:
            if attempt_split_merge(base_cluster, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                base_cluster.decrement_attempts_left()
            else:
                base_cluster.decrement_attempts_left()
        cluster_index += 1
    valid_clusters = [cluster for cluster in clusters if cluster not in clusters_merged_in_iteration]
    return valid_clusters

def get_valid_clusters(clusters, tmax):
    return [cluster for cluster in clusters if cluster.time is not None and cluster.time <= tmax]

def optimize_clustering(cluster_points_list, run_id):
    clusters_created = []
    initial_clusters = []
    # Use plt.get_cmap to avoid deprecation warnings
    cluster_colors = plt.get_cmap('tab20', len(cluster_points_list))
    history_log = []
    for i, points in enumerate(cluster_points_list):
        cluster = Cluster(points, cluster_id=f"Run{run_id}_Cluster_{i+1}", color=cluster_colors(i))
        clusters_created.append(cluster)
        initial_clusters.append(cluster)
    for cluster_instance in clusters_created:
        tour_result, distance_result, time_result = solve_tsp_elkai_constrained_cluster_obj(
            cluster_instance, tmax, speed_km_per_hr, service_time_hr)
        if tour_result:
            cluster_instance.set_tsp_result(tour_result, distance_result)
            cluster_instance.set_time(time_result)
            cluster_instance.set_cost(calculate_total_cost_cluster_obj(
                cluster_instance, distance_result, hiring_cost_per_cluster, distance_cost_per_km))
    initial_valid_clusters = get_valid_clusters(clusters_created, tmax)
    current_clusters = list(initial_valid_clusters)
    for iteration in range(max_iterations):
        current_clusters = run_merging_iterations(current_clusters, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
    final_valid_clusters = get_valid_clusters(current_clusters, tmax)
    final_cost = sum(c.cost for c in final_valid_clusters)
    final_time = sum(c.time for c in final_valid_clusters)
    initial_metrics = (len(initial_valid_clusters),
                       sum(c.cost for c in initial_valid_clusters),
                       sum(c.time for c in initial_valid_clusters))
    final_metrics = (len(final_valid_clusters), final_cost, final_time)
    return initial_metrics, final_metrics, final_valid_clusters

# -----------------------------
# PLOTTING FUNCTIONS (with real-life basemap)
# -----------------------------
def plot_nodes_map(clusters, filename=None, title='Nodes Only Plot'):
    """
    Plot each cluster's nodes in a distinct color on a real-life black-and-white map.
    The depot (first node) is marked with a red star.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.get_cmap('tab20', len(clusters))
    legend_handles = []
    all_lons = []
    all_lats = []
    for idx, cluster in enumerate(clusters):
        if len(cluster.data_points) == 0:
            continue
        # Data_points are [lat, lon]
        lats = cluster.data_points[:, 0]
        lons = cluster.data_points[:, 1]
        all_lons.extend(lons)
        all_lats.extend(lats)
        color = cmap(idx)
        ax.scatter(lons, lats, color=color, s=50)
        ax.scatter(lons[0], lats[0], marker='*', color='red', s=150)
        legend_handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='None', label=cluster.id))
    if all_lons and all_lats:
        margin_lon = (max(all_lons) - min(all_lons)) * 0.1 or 0.1
        margin_lat = (max(all_lats) - min(all_lats)) * 0.1 or 0.1
        min_lon, max_lon = min(all_lons) - margin_lon, max(all_lons) + margin_lon
        min_lat, max_lat = min(all_lats) - margin_lat, max(all_lats) + margin_lat
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        extent = [min_lon, max_lon, min_lat, max_lat]
    else:
        extent = None
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Stamen.Toner, extent=extent)
    except Exception as e:
        print("Error adding basemap:", e)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_routes_map(clusters, filename=None, title='Final Routes Plot'):
    """
    Plot final TSP routes on a real-life black-and-white map.
    Each cluster’s computed route is drawn with arrows; the depot is marked with a red star and the route is closed.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.get_cmap('tab20', len(clusters))
    legend_handles = []
    all_lons = []
    all_lats = []
    for idx, cluster in enumerate(clusters):
        if cluster.tour is None or len(cluster.tour) < 2:
            continue
        route_indices = cluster.tour
        route_coords = [cluster.data_points[i] for i in route_indices]
        lats = [pt[0] for pt in route_coords]
        lons = [pt[1] for pt in route_coords]
        all_lons.extend(lons)
        all_lats.extend(lats)
        color = cmap(idx)
        for i in range(len(lons) - 1):
            start = (lons[i], lats[i])
            end = (lons[i+1], lats[i+1])
            ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=4)
        ax.scatter(lons, lats, color='black', s=50, zorder=5)
        ax.scatter(lons[0], lats[0], marker='*', color='red', s=150, zorder=6)
        legend_handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='-', label=cluster.id))
    if all_lons and all_lats:
        margin_lon = (max(all_lons) - min(all_lons)) * 0.1 or 0.1
        margin_lat = (max(all_lats) - min(all_lats)) * 0.1 or 0.1
        min_lon, max_lon = min(all_lons) - margin_lon, max(all_lons) + margin_lon
        min_lat, max_lat = min(all_lats) - margin_lat, max(all_lats) + margin_lat
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        extent = [min_lon, max_lon, min_lat, max_lat]
    else:
        extent = None
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Stamen.Toner, extent=extent)
    except Exception as e:
        print("Error adding basemap:", e)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()

# -----------------------------
# UTILITY FUNCTIONS FOR COST DETAILS & CSV OUTPUT
# -----------------------------
def compute_cost_details(final_clusters):
    data = []
    overall_cost = 0.0
    overall_time = 0.0
    overall_distance = 0.0
    for cluster in final_clusters:
        route_cost = cluster.cost if cluster.cost is not None else 0
        overall_cost += route_cost
        overall_time += cluster.time if cluster.time is not None else 0
        distance = cluster.total_distance if cluster.total_distance is not None else 0
        overall_distance += distance
        data.append({
            'Cluster ID': cluster.id,
            'Num Points': len(cluster.data_points),
            'Distance (km)': round(distance, 2),
            'Time (hrs)': round(cluster.time, 2) if cluster.time is not None else None,
            'Cost ($)': round(cluster.cost, 2) if cluster.cost is not None else None
        })
    df = pd.DataFrame(data)
    return df, overall_distance, overall_cost, overall_time

def write_final_clusters_csv(final_clusters, part_number):
    nested_list = []
    for cluster in final_clusters:
        nested_list.append(str(cluster.data_points.tolist()))
    df = pd.DataFrame({'Final Cluster Coordinates': nested_list})
    csv_filename = f"split_merge_summary_part{part_number}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# -----------------------------
# PROCESSING SINGLE INPUT CSV FILE (SPLIT–MERGE APPROACH)
# -----------------------------
def process_input_file_split_merge(input_csv, part_number):
    df_in = pd.read_csv(input_csv)
    if df_in.columns[0].strip() == "Final Cluster Coordinates":
        coordinate_strings = df_in["Final Cluster Coordinates"].tolist()
        coordinates = [ast.literal_eval(cell) for cell in coordinate_strings if isinstance(cell, str)]
    else:
        coordinates = df_in.values.tolist()
    if not coordinates:
        print(f"No valid coordinates found in {input_csv}.")
        return {}
    # Create nodes-only plot from CSV data.
    cluster_points_list = coordinates
    nodes_plot_file = f"nodes_only_plot_part{part_number}.png"
    clusters_for_nodes = []
    colors = plt.get_cmap('tab20', len(cluster_points_list))
    for i, points in enumerate(cluster_points_list):
        clusters_for_nodes.append(Cluster(points, cluster_id=f"Cluster_{i+1}", color=colors(i)))
    plot_nodes_map(clusters_for_nodes, filename=nodes_plot_file, title='Nodes Only Plot (CSV Clusters)')
    
    # ----- FINAL RUN (Second Run) to compute TSP routes and final clusters -----
    best_final_cost = float('inf')
    best_run_metrics = None
    best_run_clusters = None
    all_runs_initial_metrics = []
    for run_number in range(SECOND_RUN_NUM_RUNS):
        initial_metrics, final_metrics, final_clusters = optimize_clustering(cluster_points_list, run_number + 100)
        all_runs_initial_metrics.append(initial_metrics)
        print(f"\n=== Run {run_number + 1} for file {input_csv} ===")
        print(f"  INITIAL Clusters: {initial_metrics[0]}, Total Cost: ${initial_metrics[1]:.2f}, Total Time: {initial_metrics[2]:.2f} hrs")
        print(f"  FINAL Clusters: {final_metrics[0]}, Total Cost: ${final_metrics[1]:.2f}, Total Time: {final_metrics[2]:.2f} hrs")
        if final_metrics[1] < best_final_cost:
            best_final_cost = final_metrics[1]
            best_run_metrics = final_metrics
            best_run_clusters = final_clusters
    final_routes_plot_file = f"final_routes_plot_part{part_number}.png"
    if best_run_clusters:
        plot_routes_map(best_run_clusters, filename=final_routes_plot_file, title='Final Routes Plot (TSP Solutions)')
        cost_df, overall_distance, overall_cost, overall_time = compute_cost_details(best_run_clusters)
    else:
        cost_df = pd.DataFrame()
        overall_distance = 0
        overall_cost = 0
        overall_time = 0
        final_routes_plot_file = ""
    
    final_clusters_to_write = best_run_clusters if best_run_clusters else None
    final_clusters_csv = write_final_clusters_csv(final_clusters_to_write, part_number) if final_clusters_to_write else ""
    
    summary = {
        'part_number': part_number,
        'input_csv': input_csv,
        'nodes_plot_file': nodes_plot_file,
        'final_routes_plot_file': final_routes_plot_file,
        'final_run': {
            'num_final_clusters': best_run_metrics[0] if best_run_metrics else 0,
            'overall_distance': overall_distance,
            'overall_cost': overall_cost,
            'overall_time': overall_time,
            'cost_df': cost_df,
            'all_runs_initial_metrics': all_runs_initial_metrics,
            'best_run_metrics': best_run_metrics
        },
        'final_clusters_csv': final_clusters_csv
    }
    return summary

# -----------------------------
# MAIN FUNCTION: PROCESS MULTIPLE CSV FILES AND CREATE EXCEL SUMMARY
# -----------------------------
def main_multi_2(input_csv_files):
    summaries = []
    for idx, csv_file in enumerate(input_csv_files, start=1):
        print(f"\nProcessing file: {csv_file}")
        summary = process_input_file_split_merge(csv_file, part_number=idx)
        if summary:
            summaries.append(summary)
    excel_filename = "split_merge_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for summary in summaries:
            sheet_name = f"PART {summary['part_number']}"
            overall_info = pd.DataFrame({
                'Input CSV': [summary['input_csv']],
                'Final Clusters': [summary['final_run']['num_final_clusters']],
                'Overall Distance (km)': [summary['final_run']['overall_distance']],
                'Overall Cost ($)': [summary['final_run']['overall_cost']],
                'Overall Time (hrs)': [summary['final_run']['overall_time']]
            })
            overall_info.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            summary['final_run']['cost_df'].to_excel(writer, sheet_name=sheet_name, startrow=5, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.write('A12', f"Nodes Only Plot (Part {summary['part_number']})")
            worksheet.insert_image('A13', summary['nodes_plot_file'], {'x_scale': 0.5, 'y_scale': 0.5})
            if summary['final_routes_plot_file']:
                worksheet.write('H12', f"Final Routes Plot (Part {summary['part_number']})")
                worksheet.insert_image('H13', summary['final_routes_plot_file'], {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.write('A40', f"Final Clusters CSV File (Part {summary['part_number']})")
            worksheet.write('A41', summary['final_clusters_csv'])
    print(f"\nExcel summary file written: {excel_filename}")

# -----------------------------
# MAIN BLOCK
# -----------------------------
if __name__ == '__main__':
    input_csv_files = [
        "split_merge_osm_data_part1.csv",
        "split_merge_osm_data_part2.csv",
        "split_merge_osm_data_part3.csv",
        "split_merge_osm_data_part4.csv",
        "split_merge_osm_data_part5.csv",
        "split_merge_osm_data_part6.csv",
        "split_merge_osm_data_part7.csv",
        "split_merge_osm_data_part8.csv",
        "split_merge_osm_data_part9.csv",
        "split_merge_osm_data_part10.csv",
        "split_merge_osm_data_part11.csv",
        "split_merge_osm_data_part12.csv",
    ]
    main_multi_2(input_csv_files)
