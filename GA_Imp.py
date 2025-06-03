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
                     # dict[int → float]: customer node ID → demand (kg). Depot is assumed ID 0.
                 num_vehicles,
                     # int: total number of identical vehicles.
                 vehicle_capacity,
                     # float: payload capacity (kg) per vehicle.
                 time_windows=None,
                     # dict[int → (earliest: float, latest: float)], optional customer time windows in minutes.
                 # ────────────────────────────────────────────────────────────────────────────
                 # 2. Time‐period discretization & emission model
                 # ────────────────────────────────────────────────────────────────────────────
                 period_breaks=None,
                     # list[float]: sorted time‐points (minutes) defining discrete periods [p[i], p[i+1]).
                 emission_fn=None,
                     # function(weight: float, speed: float) → float: emission rate (g/min or g/h).
                 # ────────────────────────────────────────────────────────────────────────────
                 # 3. Genetic‐Algorithm hyperparameters
                 # ────────────────────────────────────────────────────────────────────────────
                 pop_size=50,
                 max_gens=200,
                 tournament_size=2,
                 crossover_rate=0.9,
                 mutation_rate=0.2,
                 elite_count=2,
                 start_time=0.0):
        """
        Initialize the GA‐DP algorithm (Xiao & Konak, 2017).

        1. Graph + traffic/demand inputs:
           - travel_time_fn(u, v, depart_time) → minutes
             (time‐dependent travel‐time oracle).
           - travel_distance_fn(u, v) → km.
           - demands: {customer_node_id: demand_kg}, depot is node 0 (demand=0).
           - num_vehicles: total identical vehicles.
           - vehicle_capacity: payload capacity (kg) per vehicle.
           - time_windows: {node_id: (earliest_min, latest_min)}, optional.

        2. Time‐period discretization & emission model:
           - period_breaks: sorted list of time‐points in minutes (e.g., [0,5,10,…,1440]).
           - emission_fn(weight_kg, speed_kmh) → emission rate (g/min or g/h).

        3. GA hyperparameters:
           - pop_size: population size (default 50).
           - max_gens: number of generations (default 200).
           - tournament_size: tournament size for parent selection (default 2).
           - crossover_rate: probability of crossover (default 0.9).
           - mutation_rate: probability of mutation per offspring (default 0.2).
           - elite_count: how many top solutions to carry over unchanged (default 2).
           - start_time: simulation start time in minutes (default 0.0).
        """
        # 1. Graph + demand
        self.travel_time = travel_time_fn
        self.travel_distance = travel_distance_fn
        self.demands = demands  # dict: customer_id → demand
        self.customer_ids = sorted(demands.keys())  # list of customer node IDs
        self.N = len(self.customer_ids)  # number of customers
        self.num_vehicles = num_vehicles
        self.capacity = vehicle_capacity
        self.time_windows = {} if time_windows is None else time_windows

        # 2. Time‐period & emission
        if period_breaks is None or emission_fn is None:
            raise ValueError("period_breaks and emission_fn must be provided.")
        self.period_breaks = period_breaks  # list of M time knots; forms T = M-1 periods
        self.emission_fn = emission_fn

        # 3. GA hyperparameters
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.start_time = start_time

    def run(self):
        """
        Execute the GA‐DP search. Returns:
            best_solution: (giant_tour: list[int], split_indices: list[int])
            best_cost: float (total CO₂ emission + tardiness penalties)
        """
        # Initialize population: list of (giant_tour, split_indices)
        population = self._initialize_population()
        # Evaluate fitnesses
        fitnesses = [self._evaluate(sol) for sol in population]

        best_cost = min(fitnesses)
        best_solution = population[fitnesses.index(best_cost)]

        for gen in range(1, self.max_gens + 1):
            new_population = []

            # Elitism: carry forward elite_count best solutions
            sorted_indices = sorted(range(self.pop_size), key=lambda i: fitnesses[i])
            for i in sorted_indices[: self.elite_count]:
                new_population.append(population[i])

            # Generate rest of new population
            while len(new_population) < self.pop_size:
                # Parent selection via tournament
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._ordered_crossover(parent1, parent2)
                else:
                    # Clone parent1
                    child = (parent1[0][:], parent1[1][:])

                # Mutation
                child = self._mutate(child)

                new_population.append(child)

            # Evaluate new population
            population = new_population
            fitnesses = [self._evaluate(sol) for sol in population]

            # Update best
            gen_best_cost = min(fitnesses)
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_solution = population[fitnesses.index(gen_best_cost)]

            # (Optional) print generation stats
            avg_f = sum(fitnesses) / len(fitnesses)
            print(f"Gen {gen}: Best = {gen_best_cost:.2f}, Avg = {avg_f:.2f}")

        return best_solution, best_cost

    def _initialize_population(self):
        """
        Create initial population of size pop_size.
        Each individual = (giant_tour: list of customer IDs, split_indices: list of V-1 splits)
        """
        population = []
        for _ in range(self.pop_size):
            # Random permutation of all customers
            tour = random.sample(self.customer_ids, self.N)
            # Choose V-1 split points from [1..N-1]
            if self.num_vehicles > 1:
                splits = sorted(random.sample(range(1, self.N), self.num_vehicles - 1))
            else:
                splits = []
            population.append((tour, splits))
        return population

    def _evaluate(self, solution):
        giant_tour, split_indices = solution
        # Build per-vehicle routes
        routes = []
        prev = 0
        for split in split_indices:
            routes.append(giant_tour[prev:split])
            prev = split
        routes.append(giant_tour[prev:])  # last vehicle
        # If fewer than num_vehicles splits, pad with empty routes
        while len(routes) < self.num_vehicles:
            routes.append([])

        total_emission = 0.0
        total_tardiness = 0.0
        # Evaluate each route via DP subroutine
        for v_idx, route in enumerate(routes):
            if not route:
                continue
            min_em, schedule = self._dynamic_programming(route, v_idx)
            total_emission += min_em
            # Tardiness: for each customer, if arrival > due, add (arrival - due)
            for cust in route:
                arr_time = schedule.get(cust, (None, None))[0]
                if cust in self.time_windows:
                    due = self.time_windows[cust][1]
                    if arr_time is not None and arr_time > due:
                        total_tardiness += (arr_time - due)
        # Combine objectives (you may weight tardiness if desired)
        return total_emission + total_tardiness

    def _dynamic_programming(self, route, vehicle_idx):
        nodes = [0] + route + [0]
        n = len(nodes) 
        total_payload = sum(self.demands[c] for c in route)
        payload_on_leg = []
        running = total_payload
        for c in route:
            payload_on_leg.append(running)
            running -= self.demands[c]
        p_breaks = self.period_breaks
        M = len(p_breaks)
        T = M - 1  

        INF = float("inf")
        DP = [[INF] * T for _ in range(n)]
        predecessor = [[None] * T for _ in range(n)]
        for p in range(T):
            if p_breaks[p] >= self.start_time:
                DP[0][p] = 0.0

        for k in range(1, n):
            u = nodes[k - 1]
            v = nodes[k]
            for p_prev in range(T):
                cost_so_far = DP[k - 1][p_prev]
                if cost_so_far == INF:
                    continue
                depart_time = p_breaks[p_prev]

                travel_t = self.travel_time(u, v, depart_time)
                arrival_time = depart_time + travel_t

                if arrival_time > p_breaks[-1]:
                    continue

                idx = bisect.bisect_right(p_breaks, arrival_time) - 1
                if idx < 0:
                    continue
                p_curr = min(idx, T - 1)

                dist = self.travel_distance(u, v)
                travel_hours = travel_t / 60.0
                if travel_hours <= 0:
                    continue
                speed = dist / travel_hours  # km/h
                weight = payload_on_leg[k - 1] if (k - 1) < len(payload_on_leg) else 0
                rate = self.emission_fn(weight, speed)
                emission_arc = rate * travel_hours
                new_cost = cost_so_far + emission_arc
                if new_cost < DP[k][p_curr]:
                    DP[k][p_curr] = new_cost
                    predecessor[k][p_curr] = p_prev

        min_em = INF
        best_p_end = None
        for p in range(T):
            if DP[n - 1][p] < min_em:
                min_em = DP[n - 1][p]
                best_p_end = p

        schedule = {} 
        if best_p_end is None:

            return INF, {}
        k = n - 1
        p = best_p_end
        while k >= 0:
            node = nodes[k]
            arr_time = p_breaks[p]
            dep_time = arr_time if k < n - 1 else None
            schedule[node] = (arr_time, dep_time)
            prev_p = predecessor[k][p]
            k -= 1
            p = prev_p if prev_p is not None else 0

        return min_em, schedule

    def _tournament_selection(self, population, fitnesses):
        best = None
        best_fit = float("inf")
        for _ in range(self.tournament_size):
            i = random.randrange(len(population))
            if fitnesses[i] < best_fit:
                best_fit = fitnesses[i]
                best = population[i]
        return (best[0][:], best[1][:])

    def _ordered_crossover(self, parentA, parentB):
        tourA, splitsA = parentA
        tourB, splitsB = parentB
        size = len(tourA)

        # 1. Random cut points a < b
        a, b = sorted(random.sample(range(size), 2))
        child_tour = [None] * size

        # 2. Copy subsequence from A[a:b] into child
        for i in range(a, b):
            child_tour[i] = tourA[i]

        # 3. Fill remaining positions from B in order
        fill_positions = [i for i in range(size) if i < a or i >= b]
        ptr = 0
        for gene in tourB:
            if gene not in child_tour[a:b]:
                child_tour[fill_positions[ptr]] = gene
                ptr += 1

        # 4. Split inheritance: pick splits from A or B at random
        if random.random() < 0.5:
            child_splits = splitsA[:]
        else:
            child_splits = splitsB[:]
        # Ensure splits are valid: sorted, in 1..N-1
        child_splits = sorted([s for s in child_splits if 1 <= s < size])

        return (child_tour, child_splits)

    def _mutate(self, solution):
        tour, splits = solution

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]

        if splits and (random.random() < (self.mutation_rate / 2)):
            idx = random.randrange(len(splits))
            shift = random.choice([-1, 1])
            new_split = splits[idx] + shift
            if 1 <= new_split < len(tour) and new_split not in splits:
                splits[idx] = new_split
                splits.sort()

        return (tour, splits)
