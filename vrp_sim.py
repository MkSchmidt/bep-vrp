import pandas as pd
import numpy as np
from typing import Optional
import networkx as nx
from itertools import pairwise

class TrafficSim:
    def __init__(self, edges: pd.DataFrame, flows: pd.DataFrame):
        self.G = nx.DiGraph()
        for row in edges.to_dict('records'):
            self.G.add_edge(row["init_node"], row["term_node"], volume=0, **row)
        for row in flows.to_dict("records"):
            self.G.edges[row["from"], row["to"]]["volume"] = row["volume"]
            self.G.edges[row["to"], row["from"]]["volume"] = row["volume"]
    
    def get_route_travel_time(self, route: list[int], t_start: float = 3600*8) -> float:
        head = self._get_edge_travel_time(route[0], route[1], t_start)
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
        critical_density = self._get_critical_density(source, dest)
        density = self._get_density(source, dest, t)
        flow = self._get_flow(source, dest, t)
        if density <= critical_density:
            travel_time = free_time
        else:
            # travel_time = free_time + (density - critical_density) * B
            travel_time = free_time * (1 + (B*flow/capacity)**4)
        return travel_time
    
    def _get_density(self, source: int, dest: int, t: float) -> float:
        attrs = self.G.edges[source, dest]
        length = attrs.get("length")
        free_time = attrs.get("free_flow_time")
        flow = self._get_flow(source, dest, t)
        free_speed = (length / free_time)
        return flow / free_speed

    def _get_flow(self, source: int, dest: int, t: float) -> float:
        volume = self.G.edges[source, dest].get("volume") or 0
        return volume * self._demand(t)
    
    def _get_critical_density(self, source: int, dest: int) -> float:
        attrs = self.G.edges[source, dest]
        capacity = attrs.get("capacity")
        free_time = attrs.get("free_flow_time")
        length = attrs.get("length")
        ff_speed = length / free_time or 50 * 3.6
        return capacity / ff_speed

    # Demand function
    def _demand(self, t: float) -> float:
        low, medium, high = 0.1, 0.5, 1.1
        demands = np.array([ low, low, high, high, medium, medium, high, high, low, low ])
        times =   np.array([ 0,   4,   8.5,  10,   12,     16.5,   18,   20,   22,  24  ]) * 3600    
        
        return np.interp(t, times, demands)

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
            customers_visited = customers_visited & route_set - {depot,}

        return len(self.customers ^ customers_visited) == 0
