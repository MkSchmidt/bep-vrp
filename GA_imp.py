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
                 demands_dict,
                     # dict[int → float]: customer node ID → demand. Depot is assumed ID 0.
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
                     # list[float]: time‐points in minutes. (Still required for the DP indices.)
                 emission_fn=None,  # not used if travel_time only
                     # function(weight_kg, speed_kmh) → emission rate. (ignored in pure travel-time mode.)
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
                 depot_node_id=0):
        """
        (Modified so that splits are fixed based on equal‐division of N customers across V vehicles.)
        """
        # 1. Graph + demand
        self.travel_time = travel_time_fn
        self.demands = demands_dict
        self.customer_ids = sorted(demands_dict.keys())
        self.N = len(self.customer_ids)
        self.num_vehicles = num_vehicles
        self.capacity = vehicle_capacity
        self.time_windows = {} if time_windows is None else time_windows
        self.depot_node_id = depot_node_id

        # 2. Time‐period discretization (still required)
        if period_breaks is None:
            raise ValueError("period_breaks must be provided.")
        self.period_breaks = period_breaks

        # emission_fn kept for signature compatibility; not used if we accumulate travel_time
        self.emission_fn = emission_fn

        # 3. GA hyperparameters
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.start_time = start_time
        

        # ────────────────────────────────────────────────────────────────────────────
        # 4. Compute fixed splits so that routes sizes differ by at most 1
        # ────────────────────────────────────────────────────────────────────────────
        # Let N = total customers, V = num_vehicles.
        # Compute s = N // V, r = N % V. Then r routes have size (s+1), and (V-r) have size s.
        s = self.N // self.num_vehicles
        r = self.N % self.num_vehicles
        # Build a list of sizes: first r vehicles → size (s+1); next (V-r) vehicles → size s
        sizes = [s+1] * r + [s] * (self.num_vehicles - r)
        # Now build split indices = cumulative sums of sizes, but skip the last (equal N)
        splits = []
        cum = 0
        for size in sizes[:-1]:
            cum += size
            splits.append(cum)
        self.fixed_splits = splits
        # e.g. if N=7, V=2 → s=3, r=1 → sizes=[4,3] → splits=[4]
        #      if N=9, V=3 → s=3, r=0 → sizes=[3,3,3] → splits=[3,6]

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
            # Elitism: copy the best elite_count individuals
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
                    # Copy parent1’s tour; splits are always fixed so we ignore parent1[1]
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
            # Every chromosome uses the same fixed splits
            splits = self.fixed_splits[:]
            population.append((tour, splits))
        return population

    def _evaluate(self, solution):
        """
        Evaluate total travel time (minutes) for all vehicle routes in solution,
        splitting the giant tour according to self.fixed_splits.
        """
        giant_tour, split_indices = solution
        # 1. Split giant_tour into routes using fixed_splits
        routes = []
        prev = 0
        for split in split_indices:
            routes.append(giant_tour[prev:split])
            prev = split
        routes.append(giant_tour[prev:])
        # If fewer than num_vehicles (shouldn’t happen), pad with empty
        while len(routes) < self.num_vehicles:
            routes.append([])

        total_travel_time = 0.0

        for v_idx, route in enumerate(routes):
            if not route:
                continue

            # 2. Enforce capacity (each route size ≤ capacity)
            #    If pure “customers count,” capacity was set to ceil(N/V), so route length ≤ capacity.
            if len(route) > self.capacity:
                # infeasible split → infinite cost
                return float("inf")

            # 3. Run DP to compute minimum travel time (minutes) for that subroute
            travel_time_for_route, _ = self._dynamic_programming(route, v_idx)
            total_travel_time += travel_time_for_route

        return total_travel_time

    def _dynamic_programming(self, route, vehicle_idx):
        """
        Time‐indexed DP that returns (min_total_travel_time, schedule).
        - route: list of customer IDs (ints).
        - schedule: dict[node → (arrival_min, depart_min)] (may be used downstream).
        """
        nodes = [self.depot_node_id] + route + [self.depot_node_id]
        n = len(nodes)
        T = len(self.period_breaks) - 1

        INF = float("inf")
        # DP[k][p] = minimum cumulative travel time when we have just served nodes[0..k]
        # and we arrived in period p.
        DP = [[INF]*T for _ in range(n)]
        predecessor = [[None]*T for _ in range(n)]

        # Initialization: at k=0 (depot), any p with period_breaks[p] ≥ start_time has cost=0
        for p in range(T):
            if self.period_breaks[p] >= self.start_time:
                DP[0][p] = 0.0

        # Main DP loop
        for k in range(1, n):
            u = nodes[k-1]
            v = nodes[k]
            for p_prev in range(T):
                cost_so_far = DP[k-1][p_prev]
                if cost_so_far == INF:
                    continue
                depart_time = self.period_breaks[p_prev]

                # 1) travel time (minutes) from u→v if we depart at depart_time:
                travel_t = self.travel_time(u, v, depart_time)
                if travel_t == INF:
                    continue
                arrival_time = depart_time + travel_t
                if arrival_time > self.period_breaks[-1]:
                    continue

                # 2) determine period index p_curr for arrival_time
                idx = bisect.bisect_right(self.period_breaks, arrival_time) - 1
                if idx < 0:
                    continue
                p_curr = min(idx, T-1)

                # 3) cost increment = travel_t (minutes)
                new_cost = cost_so_far + travel_t

                if new_cost < DP[k][p_curr]:
                    DP[k][p_curr] = new_cost
                    predecessor[k][p_curr] = p_prev

        # Find best end‐state at k = n-1
        min_time = INF
        best_p_end = None
        for p in range(T):
            if DP[n-1][p] < min_time:
                min_time = DP[n-1][p]
                best_p_end = p

        if best_p_end is None:
            return INF, {}

        # Reconstruct schedule by backtracking
        schedule = {}
        k = n - 1
        p = best_p_end
        while k >= 0:
            node = nodes[k]
            arr_time = self.period_breaks[p]
            dep_time = arr_time if k < n-1 else None
            schedule[node] = (arr_time, dep_time)
            prev_p = predecessor[k][p]
            k -= 1
            p = prev_p if prev_p is not None else 0

        return min_time, schedule

    def _tournament_selection(self, population, fitnesses):
        """
        Always select at least one individual, even if all fitnesses are equal/infinite.
        """
        best = None
        best_fit = float("inf")
        for _ in range(self.tournament_size):
            i = random.randrange(len(population))
            # tie‐breaking: use ≤ so best gets set even if fitnesses[i] == best_fit
            if fitnesses[i] <= best_fit:
                best_fit = fitnesses[i]
                best = population[i]
        return (best[0][:], best[1][:])

    def _ordered_crossover(self, parentA, parentB):
        """
        Crossover only on the giant_tour; enforce child_splits = fixed_splits.
        """
        tourA, _ = parentA
        tourB, _ = parentB
        size = len(tourA)
        a, b = sorted(random.sample(range(size), 2))
        child_tour = [None]*size
        # Copy subsequence from A[a:b]
        for i in range(a, b):
            child_tour[i] = tourA[i]
        fill_positions = [i for i in range(size) if i < a or i >= b]
        ptr = 0
        for gene in tourB:
            if gene not in tourA[a:b]:
                child_tour[fill_positions[ptr]] = gene
                ptr += 1
        # Splits are always fixed
        child_splits = self.fixed_splits[:]
        return (child_tour, child_splits)

    def _mutate(self, solution):
        """
        Mutation only swaps two genes in the giant_tour; splits remain fixed.
        """
        tour, _ = solution
        splits = self.fixed_splits[:]  # force splits back to fixed
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return (tour, splits)