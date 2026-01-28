import time
import random
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (useful if we run frontend separately, though we serve static)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class BenchmarkRequest(BaseModel):
    num_items: int
    capacity: int

class AlgorithmResult(BaseModel):
    name: str
    value: int
    time_ms: float
    status_color: str

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.ratio = value / weight

# --- ALGORITHMS ---

def solve_greedy(items: List[Item], capacity: int) -> int:
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
    total_value = 0
    current_weight = 0
    for item in sorted_items:
        if current_weight + item.weight <= capacity:
            current_weight += item.weight
            total_value += item.value
    return total_value

def solve_dp(items: List[Item], capacity: int) -> int:
    n = len(items)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i-1].weight <= w:
                dp[i][w] = max(items[i-1].value + dp[i-1][w-items[i-1].weight], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

def solve_backtracking(items: List[Item], capacity: int) -> int:
    # Using Branch and Bound for efficiency, otherwise plain recursion is too slow
    n = len(items)
    items = sorted(items, key=lambda x: x.ratio, reverse=True) # Sort for better pruning
    max_profit = 0

    class Node:
        def __init__(self, level, profit, weight, bound):
            self.level = level
            self.profit = profit
            self.weight = weight
            self.bound = bound

    def bound(u, n, capacity, items):
        if u.weight >= capacity:
            return 0
        profit_bound = u.profit
        j = u.level + 1
        totweight = u.weight
        while j < n and totweight + items[j].weight <= capacity:
            totweight += items[j].weight
            profit_bound += items[j].value
            j += 1
        if j < n:
            profit_bound += (capacity - totweight) * items[j].ratio
        return profit_bound

    Q = []
    u = Node(-1, 0, 0, 0)
    v = Node(-1, 0, 0, 0)
    u.bound = bound(u, n, capacity, items)
    Q.append(u)

    start_time = time.time()

    while Q:
        # Check timeout for very large inputs if needed, though branch and bound is fast
        if time.time() - start_time > 2.0: # 2 second safety timeout
            return max_profit 

        u = Q.pop(0) # BFS 
        # Better to use Priority Queue (Best First Search) for performance, but simple BFS/Stack is okay for this size
        
        if u.level == -1:
            v.level = 0
        if u.level == n-1:
            continue
            
        v.level = u.level + 1

        # Case 1: taking the item
        v.weight = u.weight + items[v.level].weight
        v.profit = u.profit + items[v.level].value
        
        if v.weight <= capacity and v.profit > max_profit:
            max_profit = v.profit
        
        v.bound = bound(v, n, capacity, items)
        
        if v.bound > max_profit:
            Q.append(Node(v.level, v.profit, v.weight, v.bound))

        # Case 2: not taking the item
        v.weight = u.weight
        v.profit = u.profit
        v.bound = bound(v, n, capacity, items) # Recalculate bound for child
        
        if v.bound > max_profit:
            Q.append(Node(v.level, v.profit, v.weight, v.bound))

    return max_profit

def solve_genetic(items: List[Item], capacity: int) -> int:
    population_size = 50
    generations = 50
    n = len(items)
    
    def fitness(chromosome):
        current_weight = 0
        current_value = 0
        for i, bit in enumerate(chromosome):
            if bit:
                current_weight += items[i].weight
                current_value += items[i].value
        if current_weight > capacity:
            return 0
        return current_value

    population = [[random.choice([0, 1]) for _ in range(n)] for _ in range(population_size)]
    
    for _ in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        if fitness(population[0]) == 0: # All invalid or empty initial
             population = [[random.choice([0, 1]) for _ in range(n)] for _ in range(population_size)] # Restart
             continue

        new_generation = population[:2] # Elitism
        
        while len(new_generation) < population_size:
            parent1 = random.choice(population[:10])
            parent2 = random.choice(population[:10])
            crossover_point = random.randint(0, n-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            # Mutation
            if random.random() < 0.1:
                idx = random.randint(0, n-1)
                child[idx] = 1 - child[idx]
                
            new_generation.append(child)
        population = new_generation

    best_chromo = max(population, key=fitness)
    return fitness(best_chromo)

# --- API ---

@app.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    # Generate Data
    items = [
        Item(random.randint(1, 20), random.randint(10, 100))
        for _ in range(request.num_items)
    ]
    
    results = []
    
    # 1. Greedy
    start = time.perf_counter()
    val_greedy = solve_greedy(items, request.capacity)
    dur_greedy = (time.perf_counter() - start) * 1000
    results.append({
        "name": "Greedy (Approximation)",
        "value": val_greedy,
        "time_ms": round(dur_greedy, 2),
        "status_color": "text-orange-500" # Frontend will map this
    })
    
    # 2. Dynamic Programming
    start = time.perf_counter()
    val_dp = solve_dp(items, request.capacity)
    dur_dp = (time.perf_counter() - start) * 1000
    results.append({
        "name": "Dynamic Programming (Exact)",
        "value": val_dp,
        "time_ms": round(dur_dp, 2),
        "status_color": "text-teal-500"
    })
    
    # 3. Genetic
    start = time.perf_counter()
    val_genetic = solve_genetic(items, request.capacity)
    dur_genetic = (time.perf_counter() - start) * 1000
    results.append({
        "name": "Genetic Algorithm",
        "value": val_genetic,
        "time_ms": round(dur_genetic, 2),
        "status_color": "text-purple-500"
    })

    # 4. Backtracking
    start = time.perf_counter()
    val_bt = solve_backtracking(items, request.capacity)
    dur_bt = (time.perf_counter() - start) * 1000
    results.append({
        "name": "Backtracking (Branch & Bound)",
        "value": val_bt,
        "time_ms": round(dur_bt, 2),
        "status_color": "text-red-500"
    })
    
    return results

@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
