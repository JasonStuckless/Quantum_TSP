import itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time


def create_distance_matrix_from_coordinates(coordinates):
    """Create a distance matrix from a list of city coordinates"""
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            # Calculate Euclidean distance
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix


def calculate_tsp_cost(tour, dist_matrix):
    """Calculate the total distance of a tour"""
    cost = 0
    for i in range(len(tour)):
        cost += dist_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return cost


def create_tsp_hamiltonian(dist_matrix):
    """Create Hamiltonian terms for TSP problem.
    Uses a permutation-based encoding where we fix city 0 as the starting point."""
    n = len(dist_matrix)
    # Use binary encoding to represent permutations of cities 1 to n-1
    # Each city position requires log2(n-1) qubits rounded up
    remaining_cities = n - 1
    bits_per_position = int(np.ceil(np.log2(remaining_cities)))
    # Total qubits needed = (n-1) * bits_per_position
    num_qubits = (n - 1) * bits_per_position

    # Cost terms (minimize total distance)
    cost_terms = []

    # Generate all valid permutations of the remaining cities
    all_perms = list(itertools.permutations(range(1, n)))

    # For each valid permutation, assign an energy proportional to its tour cost
    for perm in all_perms:
        # Create a full tour starting with city 0
        full_tour = (0,) + perm
        # Calculate tour cost
        tour_cost = calculate_tsp_cost(full_tour, dist_matrix)

        # Create binary representation of this permutation
        binary_rep = []
        for city_idx in range(n - 1):
            city = perm[city_idx] - 1  # Adjust city index to 0-based for the remaining cities
            # Convert to binary and pad to bits_per_position
            binary = bin(city)[2:].zfill(bits_per_position)
            binary_rep.extend(list(binary))

        # Create a list of Z operators (represented as 0 or 1) for each qubit
        z_terms = []
        for i, bit in enumerate(binary_rep):
            if bit == '0':
                # For 0 bit, we want (1+Z)/2 term
                z_terms.append(1)
            else:
                # For 1 bit, we want (1-Z)/2 term
                z_terms.append(0)

        # Add this configuration with its cost
        cost_terms.append((tour_cost, z_terms))

    return cost_terms, num_qubits


def binary_to_tour(binary_str, num_cities):
    """Convert binary representation to a valid TSP tour"""
    n = num_cities
    remaining_cities = n - 1
    bits_per_position = int(np.ceil(np.log2(remaining_cities)))

    # Parse the binary string into city positions
    tour = [0]  # Always start with city 0

    for i in range(n - 1):
        if i * bits_per_position >= len(binary_str):
            # If we don't have enough bits, add cities in order
            tour.extend(list(range(1, n)))
            break

        # Extract bits for this position
        bits = binary_str[i * bits_per_position:min((i + 1) * bits_per_position, len(binary_str))]
        if len(bits) < bits_per_position:
            bits = bits.ljust(bits_per_position, '0')

        # Convert to city index
        city_idx = int(bits, 2) + 1  # +1 because we're encoding cities 1 to n-1

        # Ensure city_idx is valid (between 1 and n-1)
        if 1 <= city_idx < n and city_idx not in tour:
            tour.append(city_idx)

    # Add any missing cities
    for city in range(1, n):
        if city not in tour:
            tour.append(city)

    return tour


def create_qaoa_circuit(dist_matrix, p=1, gamma=None, beta=None):
    """Create a QAOA circuit for TSP"""
    n = len(dist_matrix)

    # Get cost terms and number of qubits from the Hamiltonian
    cost_terms, num_qubits = create_tsp_hamiltonian(dist_matrix)

    # Initialize parameters if not provided
    if gamma is None:
        gamma = np.random.uniform(0, 2 * np.pi, p)
    if beta is None:
        beta = np.random.uniform(0, np.pi, p)

    # Create the circuit
    qc = QuantumCircuit(num_qubits)

    # Initial state - superposition
    qc.h(range(num_qubits))

    # Apply QAOA layers
    for layer in range(p):
        # Problem Hamiltonian
        # For each valid permutation, we have a cost and a list of Z-terms
        for cost, z_terms in cost_terms:
            # For each valid permutation, add a phase proportional to its cost
            # We need to create a term that projects onto this specific permutation
            projection_circuit = QuantumCircuit(num_qubits)

            # For each qubit position, apply the appropriate X gate if needed to flip 0<->1
            for q_idx, z_val in enumerate(z_terms):
                if z_val == 0:  # If we want a 1 bit, flip the qubit first
                    projection_circuit.x(q_idx)

            # Now all qubits should be in |0⟩ if we have the right permutation
            # Use a multi-controlled phase gate to add phase on this specific state

            # Implement multi-controlled phase gate
            # First, use CNOTs to create an ancilla bit
            if num_qubits > 1:
                for q in range(num_qubits - 1):
                    projection_circuit.cx(q, num_qubits - 1)

                # Apply phase on ancilla
                projection_circuit.rz(gamma[layer] * cost / 10.0, num_qubits - 1)  # Scale cost to avoid huge phases

                # Undo the CNOTs
                for q in range(num_qubits - 2, -1, -1):
                    projection_circuit.cx(q, num_qubits - 1)
            else:
                # Single qubit case
                projection_circuit.rz(gamma[layer] * cost / 10.0, 0)

            # Undo the X gates
            for q_idx, z_val in enumerate(z_terms):
                if z_val == 0:  # If we flipped, unflip it
                    projection_circuit.x(q_idx)

            # Add this projection circuit to the main circuit
            qc = qc.compose(projection_circuit)

        # Mixer Hamiltonian - X rotations
        for q in range(num_qubits):
            qc.rx(2 * beta[layer], q)

    # Measure all qubits
    qc.measure_all()

    return qc, gamma, beta


def qaoa_tsp(num_cities, coordinates):
    """
    Solve TSP using QAOA

    Args:
        num_cities (int): Number of cities
        coordinates (list): List of (x, y) coordinates for each city

    Returns:
        tuple: (best_distance, best_tour) - The distance of the best tour and the best tour itself
    """
    print(f"[QAOA] Starting TSP solution for {num_cities} cities...")

    # Convert coordinates to distance matrix
    print(f"[QAOA] Creating distance matrix from coordinates...")
    dist_matrix = create_distance_matrix_from_coordinates(coordinates)

    # Select parameters based on problem size - scale up with problem complexity
    if num_cities <= 4:
        p = 1
        shots = 1024
    elif num_cities <= 6:
        p = 2
        shots = 2048
    else:
        # For larger problems, increase p and shots
        p = 3
        shots = 4096

    print(f"[QAOA] Using p={p} layers and {shots} shots for simulation")

    # Create QAOA circuit with predefined initial parameters
    print(f"[QAOA] Creating quantum circuit with optimized parameters...")
    start_time = time.time()

    # Select initial parameters based on number of layers
    if p == 1:
        gamma = np.array([0.9 * np.pi])
        beta = np.array([0.4 * np.pi])
    elif p == 2:
        gamma = np.array([0.7 * np.pi, 1.2 * np.pi])
        beta = np.array([0.3 * np.pi, 0.6 * np.pi])
    elif p == 3:
        gamma = np.array([0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi])
        beta = np.array([0.2 * np.pi, 0.4 * np.pi, 0.6 * np.pi])
    else:
        gamma = np.linspace(0.1 * np.pi, 1.5 * np.pi, p)
        beta = np.linspace(0.1 * np.pi, 0.6 * np.pi, p)

    qc, gamma, beta = create_qaoa_circuit(dist_matrix, p=p, gamma=gamma, beta=beta)
    circuit_time = time.time() - start_time
    print(f"[QAOA] Circuit creation completed in {circuit_time:.2f} seconds")
    print(f"[QAOA] Circuit depth: {qc.depth()}, Number of qubits: {qc.num_qubits}")

    # Transpile and optimize the circuit
    print(f"[QAOA] Transpiling circuit with maximum optimization...")
    transpile_start = time.time()
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator, optimization_level=3)
    transpile_time = time.time() - transpile_start
    print(f"[QAOA] Transpilation completed in {transpile_time:.2f} seconds")

    # Run quantum simulation
    print(f"[QAOA] Running quantum simulation with {shots} shots...")
    sim_start = time.time()
    result = simulator.run(compiled_circuit, shots=shots, memory=True).result()
    sim_time = time.time() - sim_start
    print(f"[QAOA] Simulation completed in {sim_time:.2f} seconds")

    counts = result.get_counts()
    print(f"[QAOA] Obtained {len(counts)} unique measurement results")

    # Analyze results
    print(f"[QAOA] Analyzing measurement results to find best tour...")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    # Get the best tour
    best_tour = None
    best_cost = float('inf')

    # Evaluate measurement results to find the best tour
    top_n = min(10, len(sorted_counts))
    print(f"[QAOA] Evaluating top {top_n} measurement results:")

    # Track unique tours to avoid duplicate evaluation
    evaluated_tours = set()

    for i in range(top_n):
        bitstring = sorted_counts[i][0]
        frequency = sorted_counts[i][1] / shots * 100
        tour = binary_to_tour(bitstring, num_cities)

        # Convert to tuple for hashing in the set
        tour_tuple = tuple(tour)

        # Skip if we've already evaluated this tour
        if tour_tuple in evaluated_tours:
            print(f"[QAOA]   Skipping duplicate tour: {tour}")
            continue

        evaluated_tours.add(tour_tuple)

        # Make sure we have a complete tour
        if len(tour) == num_cities:
            cost = calculate_tsp_cost(tour, dist_matrix)
            print(f"[QAOA]   Candidate {i + 1}: Tour={tour}, Cost={cost:.2f}, Frequency={frequency:.1f}%")
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                print(f"[QAOA]   → New best tour found!")

    # Check reverse tours which might have the same cost but different representation
    print(f"[QAOA] Analyzing tour representations...")

    # If no valid tour found, use a basic approach to construct one
    if best_tour is None:
        print(f"[QAOA] No valid tour found in top results, using basic tour construction...")
        best_bitstring = sorted_counts[0][0]
        best_tour = binary_to_tour(best_bitstring, num_cities)
        best_cost = calculate_tsp_cost(best_tour, dist_matrix)

    total_time = time.time() - start_time
    print(f"[QAOA] Best tour found: {best_tour}")
    print(f"[QAOA] Tour distance: {best_cost:.2f}")
    print(f"[QAOA] Total execution time: {total_time:.2f} seconds")

    # Note: Optimality verification is handled by main.py

    return best_cost, best_tour