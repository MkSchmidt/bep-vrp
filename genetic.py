from __future__ import annotations
import random
import networkx as nx
from typing import Self
#from graph_sim import get_route_cost

get_route_cost = lambda x, y: 1.

Node = int          # All nodes represented as ints
# A route is a list of customers visited, and it is implied that the
# first and last stop is the depot. The depot node is left out of the list
Route = list[Node]

class CVRP:
    def __init__(self, dense_graph: nx.Graph, depot: Node, capacity: int):
        self.dense_graph = dense_graph
        self.depot = depot
        self.capacity = capacity
        self.customers: set[Node] = set(dense_graph.nodes) ^ {depot}

    def check_solution_valid(self, solution: Solution) -> bool:
        list1 = solution.as_delimited(depot=self.depot)
        set1 = set(list1)

        # missing elements
        missing_elements = set(self.customers) - set1
        if missing_elements:
            return False

        # repeated elements
        if len(set1) < len(list1) - len(solution.as_routes()):
            return False

        # capacity constraint
        for route in solution.as_routes():
            if len(route) > self.capacity or len(route) == 0:
                return False

        return True

    def make_solution_valid(self, solution: Solution) -> Solution:
        list1 = solution.as_delimited(depot=self.depot)
        customer_list = [node for node in list1 if node != self.depot]

        seen: set[Node] = set()
        duplicates: list[int] = []
        fixed_list: list[Node] = []
        for idx, node in enumerate(customer_list):
            if node not in seen:
                seen.add(node)
                fixed_list.append(node)
            else:
                duplicates.append(idx)

        missing = list(self.customers - set(fixed_list))

        for i in range(min(len(missing), len(duplicates))):
            fixed_list.insert(duplicates[i], missing[i])

        if len(missing) > len(duplicates):
            fixed_list.extend(missing[len(duplicates):])

        new_routes: list[Route] = []
        current_route: Route = []
        for node in fixed_list:
            current_route.append(node)
            if len(current_route) >= self.capacity:
                new_routes.append(current_route)
                current_route = []
        if current_route:
            new_routes.append(current_route)

        return Solution.from_routes(new_routes)

class GeneticAlgorithm:
    def __init__(self, cvrp: CVRP):
        self.max_generations: int = 1000
        population_size: int = 100
        self.pick_proportion: float = 0.2
        self.k: int = 3

        self.cvrp: CVRP = cvrp
        self.population: list[Individual] = [Individual(cvrp) for _ in range(population_size)]

    def get_solution_cost(self, solution: Solution) -> float:
        delimited_route = solution.as_delimited(delimiter=self.cvrp.depot)
        return get_route_cost(self.cvrp.dense_graph, delimited_route)

    def evolve(self) -> None:
        for _ in range(self.max_generations):
            self.population.sort(key=lambda ind: self.get_solution_cost(ind.solution))
            survivors = self.population[:int(self.pick_proportion * len(self.population))]
            new_generation = survivors.copy()

            while len(new_generation) < len(self.population):
                parent1, parent2 = random.sample(survivors, 2)
                offspring = Individual(self.cvrp)
                offspring.solution = parent1.solution
                offspring.apply_solution_operators(parent2.solution)
                offspring.apply_parameter_operators()
                offspring.apply_operator_operators(parent2)
                new_generation.append(offspring)

            self.population = new_generation

class Individual:
    def __init__(self, cvrp: CVRP):
        self.cvrp: CVRP = cvrp

        self.crossover_rate: float = random.choice([0.2, 0.4, 0.6, 0.8])
        self.mutation_rate: float = random.choice([0.3, 0.5, 0.7, 0.9])

        self.crossover_op = self.order_based_crossover
        self.mutation_op = self.reverse_mutation
        self.improvement_op = self.swap_improvement

        remaining_customers = cvrp.customers.copy()
        routes: list[Route] = []
        while remaining_customers:
            route_length = min(cvrp.capacity, len(remaining_customers))
            route: Route = random.sample(list(remaining_customers), k=route_length)
            routes.append(route)
            remaining_customers -= set(route)
        self.solution: Solution = Solution.from_routes(routes)

    def apply_solution_operators(self, other_solution: Solution) -> None:
        offspring = self.crossover_op(other_solution)
        offspring = self.mutation_op(offspring)
        offspring = self.improvement_op(offspring)
        self.solution = offspring

    def apply_parameter_operators(self) -> None:
        self.crossover_rate *= random.uniform(0.9, 1.1)
        self.mutation_rate *= random.uniform(0.9, 1.1)
        self.crossover_rate = min(max(self.crossover_rate, 0.1), 1.0)
        self.mutation_rate = min(max(self.mutation_rate, 0.1), 1.0)

    def apply_operator_operators(self, other: Self) -> None:
        if random.random() < 0.5:
            self.crossover_op, other.crossover_op = other.crossover_op, self.crossover_op
        if random.random() < 0.5:
            self.mutation_op, other.mutation_op = other.mutation_op, self.mutation_op
        if random.random() < 0.5:
            self.improvement_op, other.improvement_op = other.improvement_op, self.improvement_op

    def order_based_crossover(self, parent2: Solution) -> Solution:
        list1 = self.solution.as_delimited(depot=self.cvrp.depot)
        list2 = parent2.as_delimited(depot=self.cvrp.depot)
        assert len(list1) == len(list2)
        start = random.randrange(len(list1) + 1)
        stop = random.randrange(start, len(list1) + 1)
        combined_list = list1[0:start] + list2[start:stop] + list1[stop:]
        return Solution.from_delimited(combined_list, self.cvrp.depot)

    def reverse_mutation(self, solution: Solution) -> Solution:
        routes = solution.as_routes()
        if not routes:
            return solution
        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]
        if len(route) < 2:
            return solution
        i, j = sorted(random.sample(range(len(route)), 2))
        route[i:j + 1] = reversed(route[i:j + 1])
        return Solution.from_routes(routes)

    def swap_improvement(self, solution: Solution) -> Solution:
        routes = solution.as_routes()
        for route in routes:
            if len(route) < 2:
                continue
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return Solution.from_routes(routes)

class Solution:
    @staticmethod
    def from_routes(routes: list[Route]) -> Self:
        solution = Solution()
        solution.routes: list[Route] = routes
        solution.depot: Node | None = None
        return solution
    
    @staticmethod
    def from_delimited(nodes: list[Node], depot: Node) -> Self:
        solution = Solution()
        solution.routes = []
        solution.depot = depot
        read_head = 0
        while read_head < len(nodes):
            route: Route = []
            while read_head < len(nodes) and nodes[read_head] != depot:
                route.append(nodes[read_head])
                read_head += 1
            read_head += 1
            if route:
                solution.routes.append(route)
        return solution

    def as_routes(self) -> list[Route]:
        return self.routes

    def as_delimited(self, depot: Node | None = None) -> list[Node]:
        if depot is None:
            depot = self.depot
        assert depot is not None, "Depot (delimiter) must be set"
        result: list[Node] = [depot]
        for route in self.routes:
            result += route + [depot]
        return result

