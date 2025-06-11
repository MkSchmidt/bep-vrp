import random
import numpy as np
from sklearn.cluster import KMeans

class BSOLNS:
    def __init__(self,
                 travel_time_fn,
                 demands,
                 vehicle_capacity,
                 start_time=0.0,
                 pop_size=50,
                 n_clusters=5,
                 ideas_per_cluster=5,
                 max_iter=100,
                 remove_rate=0.2):
        """
        travel_time_fn: function(u, v, depart_time) → travel time on edge (u→v)
        demands: list of customer demands (index 1 → first customer)
        vehicle_capacity: numeric capacity
        start_time: simulation depart time for evaluation
        pop_size: number of solutions in population
        n_clusters: clusters for Brain Storm
        ideas_per_cluster: ideas per cluster per iter
        max_iter: total iterations
        remove_rate: fraction to remove in destroy phase
        """
        self.travel_time = travel_time_fn
        self.demands = demands
        self.capacity = vehicle_capacity
        self.start_time = start_time

        self.pop_size = pop_size
        self.k = n_clusters
        self.ideas_per_cluster = ideas_per_cluster
        self.max_iter = max_iter
        self.remove_rate = remove_rate
        self.num_customers = len(demands)

# In the BSOLNS class:

    def evaluate(self, solution, start_time=None):
        """
        Simulate a solution by summing the duration of each route.
        """
        total_duration = 0.0
        for route in solution:
            total_duration += self._evaluate_route_cost(route)
        
        # The cost is the sum of all travel times.
        return total_duration

    def _evaluate_route_cost(self, route):
        """Calculates the total travel time for a single route."""
        # This function is now the core of all evaluations and is correct for this objective.
        t = self.start_time
        duration = 0.0
        prev = 0
        for cust in route:
            travel = self.travel_time(prev, cust, t)
            duration += travel
            t += travel # Note: t is local to this route simulation
            prev = cust
        duration += self.travel_time(prev, 0, t)
        return duration

    def greedy_initial_solution(self, start_time=None):
        """Build a first solution by repeatedly inserting the nearest feasible customer."""
        t0 = self.start_time if start_time is None else start_time
        unvisited = set(range(1, self.num_customers + 1))
        solution = []

        while unvisited:
            load = 0
            route = []
            curr = 0
            t = t0

            while True:
                feas = [c for c in unvisited if load + self.demands[c-1] <= self.capacity]
                if not feas:
                    break
                # pick the one with least travel_time(curr→c) at time t
                next_c = min(feas, key=lambda c: self.travel_time(curr, c, t))
                t += self.travel_time(curr, next_c, t)
                route.append(next_c)
                unvisited.remove(next_c)
                load += self.demands[next_c-1]
                curr = next_c

            solution.append(route)

        return solution

    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            sol = self.greedy_initial_solution()
            cost = self.evaluate(sol)
            pop.append({'sol': sol, 'cost': cost})
        return pop

    def cluster_population(self, population):
        costs = np.array([p['cost'] for p in population]).reshape(-1, 1)
        km = KMeans(n_clusters=self.k, n_init=10)
        labels = km.fit_predict(costs)
        clusters = [[] for _ in range(self.k)]
        for idx, lbl in enumerate(labels):
            clusters[lbl].append(population[idx])
        return clusters

# In the BSOLNS class, replace your old crossover with this one.

    def crossover(self, parent1, parent2):
        """
        Best Cost Route Crossover (BCRC).
        Inherits the best routes from both parents to create a new, valid solution.
        """
        # 1. Collect all routes from both parents and calculate their individual costs
        all_routes = parent1['sol'] + parent2['sol']
        routes_with_costs = []
        for r in all_routes:
            if r: # Ignore empty routes
                routes_with_costs.append({'route': r, 'cost': self._evaluate_route_cost(r)})

        # 2. Sort routes by cost
        routes_with_costs.sort(key=lambda x: x['cost'])

        # 3. Build a new solution by greedily selecting the best non-conflicting routes
        new_solution = []
        customers_served = set()
        for item in routes_with_costs:
            route = item['route']
            # Check if any customer in this route has already been served
            if not any(c in customers_served for c in route):
                new_solution.append(route)
                for c in route:
                    customers_served.add(c)

        # 4. Find unserved customers and repair the solution
        all_customers = set(range(1, self.num_customers + 1))
        unserved = list(all_customers - customers_served)
        random.shuffle(unserved)

        # 5. Insert unserved customers using the same logic as destroy_repair
        for c in unserved:
            best_insertion_cost = float('inf')
            best_insertion_spot = None

            for i, r in enumerate(new_solution):
                load = sum(self.demands[x - 1] for x in r)
                if load + self.demands[c - 1] <= self.capacity:
                    for pos in range(len(r) + 1):
                        tmp_route = r[:pos] + [c] + r[pos:]
                        cost = self._evaluate_route_cost(tmp_route)
                        if cost < best_insertion_cost:
                            best_insertion_cost = cost
                            best_insertion_spot = (i, pos)

            cost_of_new_route = self._evaluate_route_cost([c])

            if best_insertion_spot is not None and best_insertion_cost <= cost_of_new_route:
                route_idx, pos = best_insertion_spot
                new_solution[route_idx].insert(pos, c)
            else:
                new_solution.append([c])
                
        return new_solution

    def destroy_repair(self, sol):
        flat = [c for route in sol for c in route]
        if not flat: return [] # Handle empty solution case
        n_remove = max(1, int(self.remove_rate * len(flat)))
        to_remove = set(random.sample(flat, n_remove))

        new_routes = []
        removed = []
        for r in sol:
            kept = [c for c in r if c not in to_remove]
            if kept:
                new_routes.append(kept)
            removed.extend([c for c in r if c in to_remove])

        random.shuffle(removed)

        for c in removed:
            best_insertion_cost = float('inf')
            best_insertion_spot = None 

            # Option 1: Try inserting 'c' into an existing route
            for i, r in enumerate(new_routes):
                load = sum(self.demands[x - 1] for x in r)
                if load + self.demands[c - 1] <= self.capacity:
                    for pos in range(len(r) + 1):
                        # Create a temporary route with 'c' inserted
                        tmp_route = r[:pos] + [c] + r[pos:]
                        
                        # Evaluate the cost of ONLY this modified route
                        cost = self._evaluate_route_cost(tmp_route)

                        if cost < best_insertion_cost:
                            best_insertion_cost = cost
                            best_insertion_spot = (i, pos)

            cost_of_new_route = self._evaluate_route_cost([c])

            # Decide: Insert into existing route, or create a new one
            if best_insertion_spot is not None and best_insertion_cost <= cost_of_new_route:
                route_idx, pos = best_insertion_spot
                new_routes[route_idx].insert(pos, c)
            else:
                new_routes.append([c])
        
        return new_routes


    def select_new_population(self, candidates):
        return sorted(candidates, key=lambda x: x['cost'])[:self.pop_size]

    def run(self):
        population = self.initialize_population()
        best = min(population, key=lambda x: x['cost'])
        best_history = []

        for it in range(self.max_iter):
            clusters = self.cluster_population(population)
            new_ideas = []

            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                for _ in range(self.ideas_per_cluster):
                    p1, p2 = random.sample(cluster, 2)
                    idea = self.crossover(p1, p2)
                    improved = self.destroy_repair(idea)
                    cost = self.evaluate(improved)
                    new_ideas.append({'sol': improved, 'cost': cost})

            population = self.select_new_population(population + new_ideas)
            current_best = min(population, key=lambda x: x['cost'])
            if current_best['cost'] < best['cost']:
                best = current_best
            best_history.append(current_best["cost"])

            print(f"Iter {it+1}/{self.max_iter}, Best Cost: {best['cost']:.2f}")

        return best, best_history
