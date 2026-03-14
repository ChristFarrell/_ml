# NOTES

## [Homework 1](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%2001%20050326/Climb.py)

This homework was getting helped by AI for help understanding.<br>
Link for AI : <br>

This project works by using climbing techniques to find the fastest final distance. At first we started by making 10 random seed from 0 until 100 in coordinat of x and y. The program also will calculate the distance for each seed.
```python
random.seed(42)  # Ensures consistent coordinates across runs
city_locations = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(10)}

def calculate_distance(c1, c2):
    (x1, y1), (x2, y2) = city_locations[c1], city_locations[c2]
    # Standard Euclidean distance formula
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
```

After that program will trying the some method. The first method is using Hill Climbing. Hill Climbing Essentially, the program wants to find the highest "peak." The second method is neighbor (2-opt). The program selects two sections of road at random, then rotates their order so that the "crossed" roads become "straight"
```python
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
```

After that, the engine started when program by starting with a random route, for example, from house 1 to 10. The program will then:
- Continue swapping the order of the houses.
- Check the distance, if the swapping results in a shorter distance.
- Give up: If the program has tried swapping up to 500 times but no other route options are found.

## [Homework 2](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%2002%20120326/Manual%20Calculation%20of%20The%20Inverse%20Transit%20Algorithm.jpeg)
Problem 1: $f(x, y, z) = (x \cdot y) + z$<br>
In this computational graph, we first perform a Forward Pass where the inputs $x=1$ and $y=2$ are multiplied to produce an intermediate value of $2$, which is then added to $z=3$ to reach a final output of $5$.
| Part | Operation / Variable | Chain Rule Formula | Final Gradient |
| :--- | :--- | :--- | :--- |
| **Addition** | $P + z$ | $\frac{\partial f}{\partial P}$ and $\frac{\partial f}{\partial z}$ | **1** |
| **z** | Input $z$ | $\frac{\partial f}{\partial z}$ | **1** |
| **Multiplication** | $x * y$ | $\frac{\partial f}{\partial P}$ | **1** |
| **x** | Input $x$ | $\frac{\partial f}{\partial P} \cdot y$ | **2** |
| **y** | Input $y$ | $\frac{\partial f}{\partial P} \cdot x$ | **1** |

Problem 2: $f(x, y, z, t) = ((x \cdot y) + z) \cdot t$<br>
In the Forward Pass, the product of $x(1)$ and $y(2)$ results in $2$, which is added to $z(3)$ to get a sub-total of $5$, and finally multiplied by $t(4)$ to yield a total of $20$.
| Part | Operation / Variable | Chain Rule Formula | Final Gradient |
| :--- | :--- | :--- | :--- |
| **Final Multi** | $Q * t$ | $\frac{\partial f}{\partial Q}$ and $\frac{\partial f}{\partial t}$ | $Q \rightarrow 4, t \rightarrow 5$ |
| **t** | Input $t$ | $\frac{\partial f}{\partial t}$ | **5** |
| **Addition** | $P + z$ | $\frac{\partial f}{\partial Q} \cdot 1$ | **4** |
| **z** | Input $z$ | $\frac{\partial f}{\partial z}$ | **4** |
| **First Multi** | $x * y$ | $\frac{\partial f}{\partial P}$ | **4** |
| **x** | Input $x$ | $\frac{\partial f}{\partial P} \cdot y$ | **8** |
| **y** | Input $y$ | $\frac{\partial f}{\partial P} \cdot x$ | **4** |