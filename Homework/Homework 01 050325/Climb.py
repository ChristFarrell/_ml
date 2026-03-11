import random

# 1. GENERATE DATA FOR 10 CITIES (Random Coordinates)
random.seed(42)  # Ensures consistent coordinates across runs
city_locations = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(10)}

def calculate_distance(c1, c2):
    (x1, y1), (x2, y2) = city_locations[c1], city_locations[c2]
    # Standard Euclidean distance formula
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

class TSPSolution:
    def __init__(self, route):
        self.route = route

    def height(self):
        """Condition: Total Distance * -1 (Higher height = shorter distance)"""
        total = 0
        for i in range(len(self.route) - 1):
            total += calculate_distance(self.route[i], self.route[i+1])
        total += calculate_distance(self.route[-1], self.route[0])  # Return to start
        return -total

    def neighbor(self):
        """Condition: 2-opt swap (a,b)(c,d) -> (a,d)(b,c)"""
        new_route = list(self.route)
        # Select two random indices
        i, j = sorted(random.sample(range(len(new_route)), 2))
        # Reverse the segment between i and j
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return TSPSolution(new_route)

    def __str__(self):
        return f"Route: {self.route} | Distance: {abs(self.height()):.2f} km"

# 2. MAIN FUNCTION WITH PROGRESS LOGGING
def hill_climbing(initial_solution, max_gens, max_fails):
    print(f"STARTING POINT: {initial_solution}\n" + "="*50)
    fails = 0
    current_best = initial_solution

    for gens in range(max_gens):
        new_solution = current_best.neighbor()
        
        # Check if the neighbor is 'higher' (shorter distance)
        if new_solution.height() > current_best.height():
            current_best = new_solution
            fails = 0
            # Log progress every time a better route is found
            print(f"Gen {gens:4}: Improvement! -> Distance: {abs(current_best.height()):.2f} km")
        else:
            fails += 1
        
        if fails >= max_fails:
            print(f"\nStopped at Gen {gens}: No better route found after {max_fails} attempts.")
            break
            
    print("="*50 + f"\nFINAL SOLUTION: {current_best}")
    return current_best

# 3. RUN THE PROGRAM
# Initial state: sequential order 0-1-2-...-9
initial_route = list(range(10))
starting_solution = TSPSolution(initial_route)

# Allowing 5000 generations and 500 maximum failures
hill_climbing(starting_solution, max_gens=5000, max_fails=500)