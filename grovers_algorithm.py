
import numpy as np
import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

def compute_distances(city_coords):
    num_cities = len(city_coords)
    return {
        (i, j): euclidean_distance(city_coords[i], city_coords[j])
        for i in range(num_cities) for j in range(num_cities) if i != j
    }

def create_oracle(num_qubits):
    oracle = QuantumCircuit(num_qubits)
    binary_index = format(0, f'0{num_qubits}b')
    for i, bit in enumerate(reversed(binary_index)):
        if bit == '1':
            oracle.x(i)
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    for i, bit in enumerate(reversed(binary_index)):
        if bit == '1':
            oracle.x(i)
    return oracle

def create_diffuser(num_qubits):
    diffuser = QuantumCircuit(num_qubits)
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.h(num_qubits - 1)
    diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    return diffuser

def grovers_algorithm_tsp(num_cities, coordinates):
    city_coords = np.array(coordinates)
    distances = compute_distances(city_coords)
    all_routes = list(itertools.permutations(range(num_cities)))

    num_qubits = int(np.ceil(np.log2(len(all_routes))))
    backend = AerSimulator()

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    oracle = create_oracle(num_qubits)
    diffuser = create_diffuser(num_qubits)

    grover_iterations = int(np.pi / 4 * np.sqrt(len(all_routes)))
    for _ in range(grover_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure_all()
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=1024)
    result = job.result()
    counts = result.get_counts()

    valid_results = [int(key, 2) for key in counts if int(key, 2) < len(all_routes)]
    if not valid_results:
        return float("inf"), []

    best_index = max(valid_results, key=lambda idx: counts[format(idx, f'0{num_qubits}b')])
    best_route = all_routes[best_index]

    total_distance = sum(
        distances[(best_route[i], best_route[i + 1])] for i in range(len(best_route) - 1)
    )
    total_distance += distances[(best_route[-1], best_route[0])]

    return total_distance, list(best_route)
