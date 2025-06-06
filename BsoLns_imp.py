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

    def evaluate(self, solution, start_time=None):
        """Simulate a solution under time‐dependent travel times."""
        t = self.start_time if start_time is None else start_time
        for route in solution:
            prev = 0
            for cust in route:
                t += self.travel_time(prev, cust, t)
                prev = cust
            t += self.travel_time(prev, 0, t)
        return t - (self.start_time if start_time is None else start_time)

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

    def crossover(self, parent1, parent2):
        r1, r2 = parent1['sol'], parent2['sol']
        cut1, cut2 = len(r1)//2, len(r2)//2
        new = r1[:cut1] + r2[cut2:]
        all_c = set(range(1, self.num_customers+1))
        used = {c for route in new for c in route}
        missing = list(all_c - used)
        random.shuffle(missing)
        for c in missing:
            random.choice(new).append(c)
        return new

    def destroy_repair(self, sol):
        flat = [c for route in sol for c in route]
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
            best_cost = float('inf')
            best_route, best_pos = None, None
            for i, r in enumerate(new_routes):
                load = sum(self.demands[x-1] for x in r)
                if load + self.demands[c-1] > self.capacity:
                    continue
                for pos in range(len(r)+1):
                    tmp_route = r[:pos] + [c] + r[pos:]
                    tmp_sol = new_routes[:i] + [tmp_route] + new_routes[i+1:]
                    cost = self.evaluate(tmp_sol)
                    if cost < best_cost:
                        best_cost, best_route, best_pos = cost, i, pos
            if best_route is None:
                new_routes.append([c])
            else:
                new_routes[best_route].insert(best_pos, c)
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
