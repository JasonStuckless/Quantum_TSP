import numpy as np
import itertools
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Function to compute the Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

# Function to compute and return the pairwise distances between cities
def compute_distances(city_coords):
    num_cities = len(city_coords)
    return {
        (i, j): euclidean_distance(city_coords[i], city_coords[j])
        for i in range(num_cities) for j in range(num_cities) if i != j
    }

# Function to create the oracle for Grover's search algorithm
def create_oracle(num_qubits):
    oracle = QuantumCircuit(num_qubits)
    
    # Set the binary index for the oracle
    binary_index = format(0, f'0{num_qubits}b')
    
    # Apply X gates based on the binary index
    for i, bit in enumerate(reversed(binary_index)):
        if bit == '1':
            oracle.x(i)

    # Grover's oracle part: Apply H, then multi-controlled X (MCX), and H again
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)

    # Apply X gates again based on the binary index
    for i, bit in enumerate(reversed(binary_index)):
        if bit == '1':
            oracle.x(i)
    
    return oracle

# Function to create the diffuser (also called the Grover diffusion operator)
def create_diffuser(num_qubits):
    diffuser = QuantumCircuit(num_qubits)
    diffuser.h(range(num_qubits)) # Apply Hadamard gates to all qubits
    diffuser.x(range(num_qubits)) # Apply X gates to all qubits
    diffuser.h(num_qubits - 1) # Apply Hadamard to the last qubit
    diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1) # Apply multi-controlled X (MCX) gate
    diffuser.h(num_qubits - 1)  # Apply Hadamard and X gates again
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits)) # Apply Hadamard gates to all qubits
    
    return diffuser

# Function to implement Grover's Algorithm for solving the Traveling Salesman Problem (TSP)
def grovers_algorithm_tsp(num_cities, coordinates):
    start_time = time.time()

    city_coords = np.array(coordinates)
    distances = compute_distances(city_coords)
    
    # Generate all possible routes (permutations of cities)
    all_routes = list(itertools.permutations(range(num_cities)))

    num_qubits = int(np.ceil(np.log2(len(all_routes))))
    backend = AerSimulator()
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    # Create oracle and diffuser circuits
    oracle = create_oracle(num_qubits)
    diffuser = create_diffuser(num_qubits)

    # Estimate the number of Grover iterations required for the algorithm
    grover_iterations = int(np.pi / 4 * np.sqrt(len(all_routes)))
    
    # Apply Grover's operator (oracle + diffuser) for the required number of iterations
    for _ in range(grover_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure_all() # Measure all qubits in the circuit
    transpiled = transpile(qc, backend)
    
    # Run the simulation on the backend
    job = backend.run(transpiled, shots=1024)
    result = job.result()
    
    # Get the result counts
    counts = result.get_counts()

    # Filter valid results
    valid_results = [int(key, 2) for key in counts if int(key, 2) < len(all_routes)]
    
    if not valid_results:
        return float("inf"), []

    # Find the best route based on the maximum number of occurrences
    best_index = max(valid_results, key=lambda idx: counts[format(idx, f'0{num_qubits}b')])
    best_route = all_routes[best_index]

    # Calculate the total distance for the best route
    total_distance = sum(
        distances[(best_route[i], best_route[i + 1])] for i in range(len(best_route) - 1)
    )
    total_distance += distances[(best_route[-1], best_route[0])]
    
    # Print the results
    total_time = time.time() - start_time
    print(f"[Grover] Best tour found: {list(best_route)}")
    print(f"[Grover] Tour distance: {total_distance:.2f}")
    print(f"[Grover] Total execution time: {total_time:.2f} seconds")

    return total_distance, list(best_route)
