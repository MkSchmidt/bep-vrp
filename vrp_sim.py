import pandas as pd
import math
import numpy as np
from typing import Optional
import networkx as nx
from itertools import pairwise

class TrafficSim:
    def __init__(self, edges: pd.DataFrame, flows: pd.DataFrame, nodes: Optional[pd.DataFrame] = None):
        self.G = nx.Graph()
        for row in edges.to_dict('records'):
            self.G.add_edge(row["init_node"], row["term_node"], volume=0, **row)
        for row in flows.to_dict("records"):
            self.G.edges[row["from"], row["to"]]["volume"] = row["volume"]
        if nodes is not None:
            for node in nodes.to_dict("records"):
                self.G.nodes[node["node"]]["coordinates"] = node["x"], node["y"]
    
    def get_edge_congestion_time(self, source, dest, t):
        travel_time = self._get_edge_travel_time(source, dest, t)
        free_flow_time = self.G.edges[source, dest]["free_flow_time"]
        return travel_time - free_flow_time

    def get_route_travel_time(self, route: list[int], t_start: float = 3600*8) -> float:
        _, head = self._dynamic_dijkstra(route[0], route[1], t_start)
        if len(route) > 2:
            tail = self.get_route_travel_time(route[1:], t_start + head)
        else:
            tail = 0
        return head + tail

    def get_solution_travel_time(self, solution: list[list[int]], t_start: int = 3600*8) -> tuple[float, float]:
        travel_times = [ self.get_route_travel_time(route, t_start) for route in solution ]
        total = sum(travel_times)
        maximum = max(travel_times)
        return total, maximum

    def _get_edge_travel_time(self, source: int, dest: int, t: float) -> float:
        attrs = self.G.edges[source, dest]
        free_time = attrs.get("free_flow_time")
        B = attrs.get("b")
        capacity = attrs.get("capacity")
        #critical_density = self._get_critical_density(source, dest)
        #density = self._get_density(source, dest, t)
        flow = self._get_flow(source, dest, t)
        #if density <= critical_density:
        #    travel_time = free_time
        #else:
            # travel_time = free_time + (density - critical_density) * B
        travel_time = free_time * (1 + B*(flow/capacity)**attrs["power"])
        return travel_time

    def _dynamic_dijkstra(self, start, end, start_t) -> tuple[list[int], float]:
        nx.set_node_attributes(self.G, math.inf, "arrival_time")
        nx.set_node_attributes(self.G, None, "previous")
        self.G.nodes[start]["arrival_time"] = start_t
        
        Q = set(self.G.nodes)
        
        def get_node_sequence(graph, end_node):
            previous = graph.nodes[end_node]["previous"]
            if previous is None:
                return [end_node]
            return get_node_sequence(graph, previous) + [end_node]

        while len(Q) > 0:
            minimum_distance_node = list(Q)[0]
            minimum_distance = self.G.nodes[minimum_distance_node]["arrival_time"]
            for node_index in Q:
                node_distance = self.G.nodes[node_index]["arrival_time"]
                if node_distance < minimum_distance:
                    minimum_distance = node_distance
                    minimum_distance_node = node_index
            if minimum_distance_node == end:
                return get_node_sequence(self.G, end), self.G.nodes[end]["arrival_time"] - start_t
            Q.remove(minimum_distance_node)
            t = self.G.nodes[minimum_distance_node]["arrival_time"]
            for successor in self.G.neighbors(minimum_distance_node):
                if successor not in Q: continue
                travel_time = self._get_edge_travel_time(minimum_distance_node, successor, t)
                potential_time = minimum_distance + travel_time
                if potential_time < self.G.nodes[successor]["arrival_time"]:
                    self.G.nodes[successor]["arrival_time"] = potential_time
                    self.G.nodes[successor]["previous"] = minimum_distance_node
    
    def _get_density(self, source: int, dest: int, t: float) -> float:
        attrs = self.G.edges[source, dest]
        length = attrs.get("length")
        free_time = attrs.get("free_flow_time")
        flow = self._get_flow(source, dest, t)
        free_speed = (length / free_time)
        return flow / free_speed

    def _get_flow(self, source: int, dest: int, t: float) -> float:
        volume = self.G.edges[source, dest].get("volume") or 0
        return volume * demand(t)
    
    def _get_critical_density(self, source: int, dest: int) -> float:
        attrs = self.G.edges[source, dest]
        capacity = attrs.get("capacity")
        free_time = attrs.get("free_flow_time")
        length = attrs.get("length")
        if not (length and free_time):
            ff_speed = 50 * 3.6
        else:
            ff_speed = length / free_time
        return capacity / ff_speed

    # Demand function
#   def _demand(self, t: float) -> float:
#       low, medium, high = 0.1, 0.5, 1.1
#       demands = np.array([ low, low, high, high, medium, medium, high, high, low, low ])
#       times =   np.array([ 0,   4,   8.5,  10,   12,     16.5,   18,   20,   22,  24  ]) * 3600
#       partial_sums = (times[1:] - times[:-1]) * (demands[1:] + demands[:-1]) / 2 / (3600 * 24)
#       normalized_demands = demands / np.sum(partial_sums)
#       return np.interp(t, times, normalized_demands)

class VRP:
    def __init__(self, depot: int, customers: list[int]):
        self.customers = set(customers)
        self.depot = depot

    def check_solution_valid(self, solution: list[list[int]]) -> bool:
        customers_visited = set()

        for route in solution:
            if route[0] != self.depot or route[-1] != self.depot:
                return False
            route_set = set(route)
            if len(customers_visited & route_set) > 0:
                return False
            customers_visited = customers_visited & route_set - {self.depot,}

        return len(self.customers ^ customers_visited) == 0

period_breaks = np.array([0,   4,   8.5,  12,     16.5,   18,   22,   24]) * 3600
low_demand, medium_demand, high_demand = 0.1, 0.5, 1.1
demands = np.array([low_demand, low_demand, high_demand, medium_demand, medium_demand, high_demand, low_demand, low_demand])

def demand(t: float) -> float:
    
    # Helper function to calculate quarter-power decay parameters
    def get_quarter_power_params(t1, y1, t2, y2):
        """Calculate parameters a and b for y = a(b-t)^(1/4) given two points"""
        if abs(y1) == abs(y2):
            # Fallback to linear if same absolute values
            return None, None
        
        # Using explicit formulas: b = (y1^4*t2 - y2^4*t1)/(y1^4 - y2^4)
        y1_4 = y1**4
        y2_4 = y2**4
        b = (y1_4 * t2 - y2_4 * t1) / (y1_4 - y2_4)
        a = y1 / (b - t1)**(1/4) if (b - t1) > 0 else None
        
        return a, b
    
    t = t % (24*3600)

    # Determine which segment we're in
    segment_idx = np.searchsorted(period_breaks, t, side='right') - 1
    segment_idx = max(0, min(segment_idx, len(period_breaks) - 2))
    
    t1, t2 = period_breaks[segment_idx], period_breaks[segment_idx + 1]
    y1, y2 = demands[segment_idx], demands[segment_idx + 1]
    
    if y1 > y2:
        # Use quarter-power decay for declining periods
        a, b = get_quarter_power_params(t1, y1, t2, y2)
        
        if a is not None and b is not None and (b - t) > 0:
            demand_value = a * (b - t)**(1/4)
        else:
            # Fallback to linear interpolation if parameters are invalid
            demand_value = np.interp(t, [t1, t2], [y1, y2])
    else:
        # Use linear interpolation for all other segments
        demand_value = np.interp(t, [t1, t2], [y1, y2])
    
    # Calculate exact normalization using analytical integration
    def integrate_quarter_power(a, b, t1, t2):
        """Analytically integrate a(b-t)^(1/4) from t1 to t2"""
        # Integral of a(b-t)^(1/4) = -4a/5 * (b-t)^(5/4) + C
        def antiderivative(t):
            return -4*a/5 * (b - t)**(5/4)
        
        return antiderivative(t2) - antiderivative(t1)
    
    def integrate_linear(y1, y2, t1, t2):
        """Integrate linear interpolation from t1 to t2"""
        return (t2 - t1) * (y1 + y2) / 2
    
    # Calculate the total integral over 24 hours for normalization
    total_integral = 0
    
    for i in range(len(period_breaks) - 1):
        t_start, t_end = period_breaks[i], period_breaks[i + 1]
        y_start, y_end = demands[i], demands[i + 1]
        
        # Check if this segment uses quarter-power decay
        morning_decline = (t_start == 8.5 * 3600 and t_end == 12 * 3600)
        evening_decline = (t_start == 18 * 3600 and t_end == 22 * 3600)
        
        if (morning_decline or evening_decline) and y_start > y_end:
            # Use analytical integration for quarter-power decay
            a_seg, b_seg = get_quarter_power_params(t_start, y_start, t_end, y_end)
            
            if a_seg is not None and b_seg is not None:
                segment_integral = integrate_quarter_power(a_seg, b_seg, t_start, t_end)
            else:
                # Fallback to linear integration
                segment_integral = integrate_linear(y_start, y_end, t_start, t_end)
        else:
            # Use linear integration for other segments
            segment_integral = integrate_linear(y_start, y_end, t_start, t_end)
        
        total_integral += segment_integral
    
    # Normalize so the total integral over 24 hours equals 1
    normalization_factor = (24 * 3600) / total_integral
    
    return demand_value * normalization_factor
