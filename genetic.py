import random
import networkx as nx

class GeneticAlgorithm:
    def __init__(self, dense_graph, depot_node, capacity=10):
        self.max_generations = 1000
        population_size = 30
        self.k = 3


        self.depot_node = depot_node
        self.customers = set(dense_graph.nodes) ^ {depot_node,}
        self.capacity = capacity

        self.population = [ Individual(self.customers, capacity) for i in range(population_size) ]

class Individual:
    def __init__(self, customers: set, capacity: int):
        self.crossover_rate = random.choice([0.2, 0.4, 0.6, 0.8])
        self.mutation_rate = random.choice([0.3, 0.5, 0.7, 0.9])

        self.crossover_op = None
        self.mutation_op = None
        self.improvement_op = None

        remaining_customers = customers.copy()
        self.routes = []
        while len(remaining_customers) > 0:
            route_length = min(capacity, len(remaining_customers))
            route = random.sample(list(remaining_customers), k=route_length)
            self.routes.append(route)
            remaining_customers = remaining_customers ^ set(route)

class Solution:
    def from_routes(routes):
        solution = Solution()
        solution.routes = routes
        solution.delimiter = None
        return solution

    def from_delimited(nodes, delimiter):
        solution = Solution()
        solution.routes = []
        solution.delimiter = delimiter
        read_head = 0
        while read_head < len(nodes):
            route = []
            while nodes[read_head] != delimiter:
                route.append(nodes[read_head])
                read_head += 1
            read_head += 1
            if len(route) > 0:
                solution.routes.append(route)

        return solution

    def as_routes(self):
        return self.routes

    def as_delimited(self, delimiter=None):
        delimiter = delimiter or self.delimiter
        assert delimiter is not None, "Delimiter must be set"
        result = [delimiter]
        for route in self.routes:
            result = result + route + [delimiter]
        return result

def order_based_crossover(solution1, solution2):
    list1 = solution1.as_delimited(delimiter="DEPOT")
    list2 = solution2.as_delimited(delimiter="DEPOT")
    assert len(list1) == len(list2)

    start = random.randrange(len(list1)+1)
    stop = random.randrange(start, len(list1)+1)
    
    combined_list = list1[0:start] + list2[start:stop] + list1[stop:]

    return Solution.from_delimited(combined_list, "DEPOT")

def route_based_crossover(solution1, solution2, k=3):
    solution1_costs = [ get_route_cost(route) for route in solution1.as_routes() ]
    solution2_costs = [ get_route_cost(route) for route in solution2.as_routes() ]
    # TODO: select k fittest routes and add them to the next solution. The paper doesn't
    # define fitness for a route though???? Only for a complete solution.
    

def make_solution_feasible(solution, nodes):
    list1 = solution.as_delimited(delimiter="DEPOT")
    missing_elements = set(nodes) ^ set(list1) ^ {"DEPOT",}
    missing_element_iter = iter(missing_elements)
    
    new_list = []
    for node in list1:
        if node in new_list and node != "DEPOT":
            new_list.append(next(missing_element_iter))
        else:
            new_list.append(node)

    return Solution.from_delimited(new_list, delimiter="DEPOT")


if __name__ == "__main__":
    G = nx.DiGraph([(1,2), (2,1), (1, 3), (3, 1), (2, 3), (3, 2)])
    alg = GeneticAlgorithm(G, 1)

    sol1 = Solution.from_routes([[1, 2, 5], [3, 2, 3, 6]])

    sol2 = make_solution_feasible(sol1, range(1, 8))
