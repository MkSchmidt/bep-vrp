import random
import networkx as nx
from graph_sim import get_route_cost

type Node = int             # Nodes represented as ints
type Route = list[int]      # Routes represented as list of nodes, not including depot

class GeneticAlgorithm:
    def __init__(self, dense_graph, depot: Node, capacity: int = 10):
        self.max_generations = 1000
        population_size = 30
        self.k: int = 3

        self.depot: Node = depot
        self.dense_graph = dense_graph
        self.customers: set[Node] = set(dense_graph.nodes) ^ {depot,}
        self.capacity: int = capacity

        self.population = [ Individual(self.customers, capacity) for i in range(population_size) ]
    
    def get_solution_cost(self, solution: Solution) -> float:
        delimited_route = solution.as_delimited(delimiter=self.depot)
        return get_route_cost(self.dense_graph, delimited_route)

class Individual:
    def __init__(self, customers: set[Node], capacity: int):
        self.crossover_rate = random.choice([0.2, 0.4, 0.6, 0.8])
        self.mutation_rate = random.choice([0.3, 0.5, 0.7, 0.9])

        self.crossover_op = random.choice([
            order_based_crossover, route_based_crossover, swap_based_crossover
        ])
        self.mutation_op = random.choice([
            random_remove_mutation, worst_remove_mutation, reverse_mutation
        ])
        self.improvement_op = random.choice([
            swap_improvement, single_move_improvement, double_move_improvement
        ])

        remaining_customers = customers.copy()
        routes: list[Route] = []
        while len(remaining_customers) > 0:
            route_length = min(capacity, len(remaining_customers))
            route: Route = random.sample(list(remaining_customers), k=route_length)
            routes.append(route)
            remaining_customers = remaining_customers ^ set(route)
        self.solution = Solution.from_routes(routes)
    
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

## Cross-over operators
def order_based_crossover(solution1, solution2) -> Solution:
    list1 = solution1.as_delimited(delimiter="DEPOT")
    list2 = solution2.as_delimited(delimiter="DEPOT")
    assert len(list1) == len(list2)

    start = random.randrange(len(list1)+1)
    stop = random.randrange(start, len(list1)+1)
    
    combined_list = list1[0:start] + list2[start:stop] + list1[stop:]

    return Solution.from_delimited(combined_list, "DEPOT")

def route_based_crossover(solution1, solution2, k=3) -> Solution:
    solution1_costs = [ get_route_cost(route) for route in solution1.as_routes() ]
    solution2_costs = [ get_route_cost(route) for route in solution2.as_routes() ]
    # TODO: select k fittest routes and add them to the next solution. The paper doesn't
    # define fitness for a route though???? Only for a complete solution.

def swap_based_crossover(solution1, solution2, k: int = 2) -> Solution:
    ## TODO: Randomly select k routes from both solutions and swap the selected routes
    #  between two solutions. Acoording to the paper you have to remove duplicate customers
    #  and re-insert them into the best possible position??? Idk how you do that.
    pass


### Mutation operators
def random_remove_mutation(solution: Solution, N: int = 3) -> Solution:
    ## TODO: remove N customers at random and re-assign them to different positions
    pass

def worst_remove_mutation(solution: Solution, N: int = 3) -> Solution:
    ## TODO: remove N customers with largest cost savings and reassign them to best cost
    #  saving positions.
    pass

def reverse_mutation(solution: Solution) -> Solution:
    ## TODO: select one route at random and shuffle customer orders.
    pass


### Improvement operators
def swap_improvement(solution: Solution) -> Solution:
    ## TODO: select two customers from 2 different routes and swap them. If the swap leads
    #  to lower cost, keep the new solution.
    pass

def single_move_improvement(solution: Solution) -> Solution:
    ## TODO: move a single customer to a different route. Keep new solution if it's better.
    pass

def double_move_improvement(solution: Solution) -> Solution:
    ## TODO: select two customers and move to a different route. Keep new solution if better.
    pass

def make_solution_feasible(solution, nodes) -> Solution:
    ## TODO: deal with capacity constraints.
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
