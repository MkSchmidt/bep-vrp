import random
import bisect

class GA_DP:
    def __init__(self,
                 # ────────────────────────────────────────────────────────────────────────────
                 # 1. Graph + traffic/demand representation
                 # ────────────────────────────────────────────────────────────────────────────
                 travel_time_fn,
                     # function(u: int, v: int, depart_time: float) → float
                     #     Returns travel time (minutes) from u→v when departing at depart_time.
                 travel_distance_fn,
                     # function(u: int, v: int) → float
                     #     Returns distance (km) from u→v.
                 demands,
                     # dict[int → float]: customer node ID → demand. Depot is given separately.
                 num_vehicles,
                     # int: total number of identical vehicles.
                 vehicle_capacity,
                     # float: payload capacity (kg) per vehicle.
                 time_windows=None,
                     # dict[int → (earliest, latest)] in minutes. (unused if empty)
                 # ────────────────────────────────────────────────────────────────────────────
                 # 2. Time‐period discretization & emission model
                 # ────────────────────────────────────────────────────────────────────────────
                 period_breaks=None,
                     # list[float]: time‐points in minutes. Required for the DP indices.
                 emission_fn=None,  # not used if travel_time only
                     # function(weight_kg, speed_kmh) → emission rate. (ignored here).
                 # ────────────────────────────────────────────────────────────────────────────
                 # 3. Genetic‐Algorithm hyperparameters
                 # ────────────────────────────────────────────────────────────────────────────
                 pop_size=50,
                 max_gens=200,
                 tournament_size=2,
                 crossover_rate=0.9,
                 mutation_rate=0.2,
                 elite_count=2,
                 start_time=0.0,
                 # ────────────────────────────────────────────────────────────────────────────
                 # 4. Depot node ID
                 depot_node_id=0):
        """
        GA + Time‐indexed DP for VRP.

        travel_time_fn: (u: int, v: int, depart_time: float) → float (minutes)
        travel_distance_fn: (u: int, v: int) → float (km)
        demands: dict[node_id → demand]
        num_vehicles: number of vehicles
        vehicle_capacity: capacity per vehicle
        period_breaks: sorted list of time‐breakpoints in minutes
        start_time: departure time for all vehicles (in minutes)
        depot_node_id: actual graph node ID of depot
        """
        # 1. Graph + demand
        self.travel_time = travel_time_fn
        self.travel_distance = travel_distance_fn
        self.demands = demands
        self.customer_ids = sorted(demands.keys())
        self.N = len(self.customer_ids)
        self.num_vehicles = num_vehicles
        self.capacity = vehicle_capacity
        self.time_windows = {} if time_windows is None else time_windows
        self.depot_node_id = depot_node_id

        # 2. Time‐period discretization (required for DP)
        if period_breaks is None:
            raise ValueError("period_breaks must be provided.")
        self.period_breaks = period_breaks

        # 2b. Emission function (not used here)
        self.emission_fn = emission_fn

        # 3. GA hyperparameters
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.start_time = start_time

        # 4. Compute fixed splits so that route sizes differ by at most 1
        s = self.N // self.num_vehicles
        r = self.N % self.num_vehicles
        sizes = [s+1] * r + [s] * (self.num_vehicles - r)
        splits = []
        cum = 0
        for size in sizes[:-1]:
            cum += size
            splits.append(cum)
        self.fixed_splits = splits

    def run(self):
        """
        Execute the GA‐DP search. Returns:
            best_solution: (giant_tour: list[int], split_indices: list[int])
            best_cost: float (total travel time in minutes)
        """
        population = self._initialize_population()
        fitnesses = [self._evaluate(sol) for sol in population]

        best_cost = min(fitnesses)
        best_solution = population[fitnesses.index(best_cost)]

        for gen in range(1, self.max_gens + 1):
            new_population = []
            # Elitism: copy the best `elite_count` individuals
            sorted_indices = sorted(range(self.pop_size), key=lambda i: fitnesses[i])
            for i in sorted_indices[: self.elite_count]:
                new_population.append(population[i])

            # Fill the rest by crossover & mutation
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                if random.random() < self.crossover_rate:
                    child = self._ordered_crossover(parent1, parent2)
                else:
                    child = (parent1[0][:], self.fixed_splits[:])
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            fitnesses = [self._evaluate(sol) for sol in population]

            gen_best_cost = min(fitnesses)
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_solution = population[fitnesses.index(gen_best_cost)]

            avg_f = sum(fitnesses) / len(fitnesses)
            print(f"Gen {gen}: Best = {gen_best_cost:.2f}, Avg = {avg_f:.2f}")

        return best_solution, best_cost

    def _initialize_population(self):
        """
        Each individual = (giant_tour: permutation of customers, split_indices = self.fixed_splits)
        """
        population = []
        for _ in range(self.pop_size):
            tour = random.sample(self.customer_ids, self.N)
            splits = self.fixed_splits[:]
            population.append((tour, splits))
        return population

    def _evaluate(self, solution):
        """
        Evaluate total travel time (minutes) for all vehicle routes in solution,
        splitting the giant tour according to self.fixed_splits.
        """
        giant_tour, split_indices = solution
        # 1. Split giant_tour into routes
        routes = []
        prev = 0
        for split in split_indices:
            routes.append(giant_tour[prev:split])
            prev = split
        routes.append(giant_tour[prev:])
        while len(routes) < self.num_vehicles:
            routes.append([])

        total_travel_time = 0.0

        for route in routes:
            if not route:
                continue
            # 2. Enforce capacity (each route size ≤ capacity)
            if len(route) > self.capacity:
                return float("inf")
            # 3. Run DP to compute minimum travel time for that route
            travel_time_for_route, _ = self._dynamic_programming(route)
            total_travel_time += travel_time_for_route

        return total_travel_time

    def _dynamic_programming(self, route):
        """
        Time‐indexed DP that returns (min_total_travel_time, schedule).
        - route: list of customer IDs (ints).
        """
        # Replace zeros with actual depot_node_id
        nodes = [self.depot_node_id] + route + [self.depot_node_id]
        n = len(nodes)
        T = len(self.period_breaks) - 1
        INF = float("inf")

        # DP[k][p] = minimum cumulative travel time when we have served nodes[0..k]
        DP = [[INF] * T for _ in range(n)]
        predecessor = [[None] * T for _ in range(n)]

        # Initialization: at k=0 (depot), any p with break ≥ start_time has cost = 0
        for p in range(T):
            if self.period_breaks[p] >= self.start_time:
                DP[0][p] = 0.0

        # Main DP loop
        for k in range(1, n):
            u = nodes[k - 1]
            v = nodes[k]
            for p_prev in range(T):
                cost_so_far = DP[k - 1][p_prev]
                if cost_so_far == INF:
                    continue
                depart_time = self.period_breaks[p_prev]

                # 1) travel time from u → v departing at depart_time
                travel_t = self.travel_time(u, v, depart_time)
                if travel_t == INF:
                    continue
                arrival_time = depart_time + travel_t
                if arrival_time > self.period_breaks[-1]:
                    continue

                # 2) find period index for arrival_time
                idx = bisect.bisect_right(self.period_breaks, arrival_time) - 1
                if idx < 0:
                    continue
                p_curr = min(idx, T - 1)

                # 3) new cumulative cost
                new_cost = cost_so_far + travel_t
                if new_cost < DP[k][p_curr]:
                    DP[k][p_curr] = new_cost
                    predecessor[k][p_curr] = p_prev

        # Find best end‐state at k = n - 1
        min_time = INF
        best_p_end = None
        for p in range(T):
            if DP[n - 1][p] < min_time:
                min_time = DP[n - 1][p]
                best_p_end = p

        if best_p_end is None:
            return INF, {}

        # Reconstruct schedule by backtracking (not strictly needed here)
        schedule = {}
        k = n - 1
        p = best_p_end
        while k >= 0:
            node = nodes[k]
            arr_time = self.period_breaks[p]
            dep_time = arr_time if k < n - 1 else None
            schedule[node] = (arr_time, dep_time)
            prev_p = predecessor[k][p]
            k -= 1
            p = prev_p if prev_p is not None else 0

        return min_time, schedule

    def _tournament_selection(self, population, fitnesses):
        """
        Select one parent via tournament (size = self.tournament_size).
        """
        best = None
        best_fit = float("inf")
        for _ in range(self.tournament_size):
            i = random.randrange(len(population))
            if fitnesses[i] <= best_fit:
                best_fit = fitnesses[i]
                best = population[i]
        return (best[0][:], best[1][:])

    def _ordered_crossover(self, parentA, parentB):
        """
        Crossover on the giant_tour only; splits remain fixed.
        """
        tourA, _ = parentA
        tourB, _ = parentB
        size = len(tourA)
        a, b = sorted(random.sample(range(size), 2))
        child_tour = [None] * size

        # Copy subsequence from A[a:b]
        for i in range(a, b):
            child_tour[i] = tourA[i]
        # Fill remaining from B
        fill_positions = [i for i in range(size) if i < a or i >= b]
        ptr = 0
        for gene in tourB:
            if gene not in tourA[a:b]:
                child_tour[fill_positions[ptr]] = gene
                ptr += 1

        child_splits = self.fixed_splits[:]
        return (child_tour, child_splits)

    def _mutate(self, solution):
        """
        Mutation: swap two genes in the giant_tour; splits remain fixed.
        """
        tour, _ = solution
        splits = self.fixed_splits[:]
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return (tour, splits)
