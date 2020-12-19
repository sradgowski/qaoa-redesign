import cirq
import matplotlib.pyplot as plt
import numpy as np
import qutip
import random
import seaborn as sns
import sympy
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"


class ROD():
    def __init__(self, N, layers, method, approach="collective"):
        if N % 2 != 0:
            raise ValueError("Number of qubits must be an even number.")

        self.N = N
        self.nrows = int(N / 2)
        self.ncols = 2

        if layers < 1:
            raise ValueError("Layers must be a positive integer.")
        self.layers = layers
        if method not in ["BFGS", "Nelder-Mead", "Powell", "CG"]:
            raise ValueError("Optimization method must be from approved list.")
        self.method = method
        if approach not in ["collective", "individual"]:
            raise ValueError("Layers approach method must be from approved list.")
        self.approach = approach
        self.qubits = [[cirq.GridQubit(i, j) for i in range(self.nrows)] \
            for j in range(self.ncols)]
        
        p = list(range(N))
        self.p_cyclic_perm = [[p[i - j] for i in range(N)] for j in range(N)]

    def beta_layer(self, beta):
        """Generator for U(β, B) mixing Hamiltonian layer of QAOA."""
        for row in self.qubits:
            for qubit in row:
                yield cirq.X(qubit)**beta

    def gamma_layer(self, gamma):
        """Generator for U(γ, C) cost Hamiltonian layer of QAOA."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if i < self.nrows - 1:
                    yield cirq.ZZ(self.qubits[i][j], self.qubits[i + 1][j])**gamma
                if j < self.ncols - 1:
                    yield cirq.ZZ(self.qubits[i][j], self.qubits[i][j + 1])**gamma
                
                yield cirq.Z(self.qubits[i][j])**(gamma)

    def qaoa(self):
        """Returns a QAOA circuit."""
        circuit = cirq.Circuit()
        symbols = []
        for i in range(self.layers):
            symbols.append(sympy.Symbol("γ" + str(i + 1)))
            symbols.append(sympy.Symbol("β" + str(i + 1)))

        circuit.append(cirq.H.on_each(*[q for row in self.qubits for q in row]))

        for i in range(self.layers):
            circuit.append(self.gamma_layer(symbols[2 * i]))
            circuit.append(self.beta_layer(symbols[2 * i + 1]))

        return circuit

    def cost_hamiltonian(self):
        """Generates the Hamiltonian for spins on a ring"""
        ZZ = qutip.tensor(qutip.sigmaz(), qutip.sigmaz())
        H_p = qutip.qeye(2)
        for k in range(self.N - 3):
            H_p = qutip.tensor(qutip.qeye(2), H_p)

        H1 = (1 - qutip.tensor(ZZ, H_p)) / 2
        H0 = sum([H1.permute(self.p_cyclic_perm[i]) for i in range(self.N)])
        H = self.N - H0
        
        eigenvalues_H0, _ = H0.eigenstates()

        return H, H0, eigenvalues_H0

    def mixer_hamiltonian(self):
        """Generates the Mixer Hamiltonian."""
        H_mix = qutip.sigmax()
        for k in range(self.N - 1):
            H_mix = qutip.tensor(qutip.qeye(2), H_mix)

        return sum([H_mix.permute(self.p_cyclic_perm[i]) for i in range(self.N)])

    def vector_input(self):
        """Generates the Vector Input."""
        vec_0 = (qutip.basis(2, 0) + qutip.basis(2, 1)) / np.sqrt(2)
        vec_input = vec_0
        for k in range(self.N - 1):
            vec_input = qutip.tensor(vec_input, vec_0)

        return vec_input

    def optimization(self):
        """Computes the optimal energy from circuit."""

        H, H0, _ = self.cost_hamiltonian()
        H_mix = self.mixer_hamiltonian()
        vec_input = self.vector_input()

        # Optimize layers simultaneously
        if self.approach == "collective":
            def cost(angles):
                D = 1
                for i in range(self.layers)[::-1]:
                    D *= (1j * angles[2 * i] * H_mix).expm()
                    D *= (1j * angles[2 * i + 1] * H0).expm()

                # Cost = |<ψ|U' H U|ψ>|
                vec_var = (D * vec_input)
                return abs((vec_var.dag() * H * vec_var).tr())

            angles = []
            print("\n\n")
            for i in range(2 * self.layers):
                angle = random.random()
                print(f"Initialized angle {i + 1}: {angle}")
                angles.append(angle)

            print(f"\nOptimizing angles with {self.method}...\n")
            results = minimize(cost, angles, method=self.method)
            for i in range(2 * self.layers):
                print(f"Optimized angle {i + 1}: {results.x[i]}")

            return results.x

        # Optimize layers individually
        else:
            all_angles = []
            print("\n\n")

            def cost(angles):
                U = (1j * angles[0] * H_mix).expm()
                U *= (1j * angles[1] * H0).expm()

                # Cost = |<ψ|U' H U|ψ>| 
                vec_var = (U * vec_input)
                return abs((vec_var.dag() * H * vec_var).tr())
            
            for i in range(self.layers):
                new_angles = [random.random(), random.random()]
                print(f"Initialized Gamma {i + 1}: {new_angles[0]}")
                print(f"Initialized Beta {i + 1}: {new_angles[1]}")

                results = minimize(cost, new_angles, method=self.method)
                U1 = (1j * results.x[0] * H_mix).expm()
                U2 = (1j * results.x[1] * H0).expm()
                vec_input = U1 * U2 * vec_input

                all_angles.append(results.x[0])
                all_angles.append(results.x[1])

            print("\n")
            print(f"Optimizing angles with {self.method}...\n")
            for i in range(self.layers):
                print(f"Optimized Gamma {i + 1}: {all_angles[2 * i]}")
                print(f"Optimized Beta {i + 1}: {all_angles[2 * i + 1]}")

            return all_angles

    def final_energy(self):
        _, H0, eigenvalues = self.cost_hamiltonian()
        H_mix = self.mixer_hamiltonian()
        vec_input = self.vector_input()
        results = self.optimization()

        D = 1
        for i in range(self.layers)[::-1]:
            D *= (1j * results[2 * i] * H_mix).expm()
            D *= (1j * results[2 * i + 1] * H0).expm()

        vec_var = (D * vec_input)

        return abs((vec_var.dag() * H0 * vec_var).tr())/eigenvalues[-1]

    def analyze(self):
        print(f"\n********** QAOA with {self.N} Qubits, {self.layers} Layers **********")
        # print(self.qaoa())
        if self.approach == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 3)
        print(f"\nFinal Energy: {energy}\n")
        return(energy)


class PowerIteration(ROD):
    def optimization(self):
        """Computes the optimal energy from circuit."""

        H, H0, _ = self.cost_hamiltonian()
        H_mix = self.mixer_hamiltonian()
        vec_input = self.vector_input()

        # Optimize layers simultaneously
        if self.approach == "collective":
            def cost(angles):
                U = 1
                for i in range(self.layers)[::-1]:
                    U *= (1j * angles[2 * i] * H_mix).expm()
                    U *= (1j * angles[2 * i + 1] * H0).expm()

                # Cost = |sqrt(<ψ|H^2|ψ>)| - |<ψ|U' H|ψ>|

                vec_var = U * vec_input
                term_one = (vec_input.dag() * (H**2) * vec_input).tr()
                term_two = (vec_var.dag() * H * vec_input).tr()
                return abs(abs(np.sqrt(term_one)) - abs(term_two))

            angles = []
            print("\n\n")
            for i in range(2 * self.layers):
                angle = random.random()
                print(f"Initialized angle {i + 1}: {angle}")
                angles.append(angle)

            print(f"\nOptimizing angles with {self.method}...\n")
            results = minimize(cost, angles, method=self.method)
            for i in range(2 * self.layers):
                print(f"Optimized angle {i + 1}: {results.x[i]}")

            return results.x

        # Optimize layers individually
        else:
            all_angles = []
            print("\n\n")

            def cost(angles):
                U = (1j * angles[0] * H_mix).expm()
                U *= (1j * angles[1] * H0).expm()

                # Cost = |sqrt(<ψ|H^2|ψ>)| - |<ψ|U' H|ψ>| 
                vec_var = (U * vec_input)
                term_one = (vec_input.dag() * (H**2) * vec_input).tr()
                term_two = (vec_var.dag() * H * vec_input).tr()
                return abs(abs(np.sqrt(term_one)) - abs(term_two))
            
            for i in range(self.layers):
                new_angles = [random.random(), random.random()]
                print(f"Initialized Gamma {i + 1}: {new_angles[0]}")
                print(f"Initialized Beta {i + 1}: {new_angles[1]}")

                results = minimize(cost, new_angles, method=self.method)
                U1 = (1j * results.x[0] * H_mix).expm()
                U2 = (1j * results.x[1] * H0).expm()
                vec_input = U1 * U2 * vec_input

                all_angles.append(results.x[0])
                all_angles.append(results.x[1])

            print("\n")
            print(f"Optimizing angles with {self.method}...\n")
            for i in range(self.layers):
                print(f"Optimized Gamma {i + 1}: {all_angles[2 * i]}")
class ROD():
    def __init__(self, N, layers, approach, method="iterative"):
        self.N = N
        if N % 2 == 0:
            self.nrows = int(N / 2)
            self.ncols = 2
        else:
            self.nrows = N
            self.ncols = 1

        if layers < 1:
            raise ValueError("Layers must be a positive integer.")
        self.layers = layers
        if approach not in ["BFGS", "Nelder-Mead", "Powell", "CG"]:
            raise ValueError("Optimization approach must be from approved list.")
        self.approach = approach
        if method not in ["collective", "iterative"]:
            raise ValueError("Layers method approach must be from approved list.")
        self.method = method
        self.qubits = [[cirq.GridQubit(i, j) for i in range(self.nrows)] \
            for j in range(self.ncols)]
        
        p = list(range(N))
        self.cyclic_perm = [[p[i - j] for i in range(N)] for j in range(N)]

    def beta_layer(self, beta):
        """Generator for U(beta, B) mixing Hamiltonian layer of QAOA."""
        for row in self.qubits:
            for qubit in row:
                yield cirq.X(qubit)**beta

    def gamma_layer(self, gamma):
        """Generator for U(gamma, C) cost Hamiltonian layer of QAOA."""
        for i in range(self.nrows):
            for j in range(self.ncols):
                if i < self.nrows - 1:
                    yield cirq.ZZ(self.qubits[i][j], self.qubits[i + 1][j])**gamma
                if j < self.ncols - 1:
                    yield cirq.ZZ(self.qubits[i][j], self.qubits[i][j + 1])**gamma
                yield cirq.Z(self.qubits[i][j])**(gamma)

    def qaoa(self):
        """Returns a QAOA circuit."""
        circuit = cirq.Circuit()
        symbols = []
        for i in range(self.layers):
            symbols.append(sympy.Symbol("\u03B3" + str(i + 1)))
            symbols.append(sympy.Symbol("\u03B2" + str(i + 1)))

        circuit.append(cirq.H.on_each(*[q for row in self.qubits for q in row]))
        for i in range(self.layers):
            circuit.append(self.gamma_layer(symbols[2 * i]))
            circuit.append(self.beta_layer(symbols[2 * i + 1]))

        return circuit

    def cost_hamiltonian(self):
        """Generates the Cost Hamiltonian for spins on a ring."""
        
        ZZ = qutip.tensor(qutip.sigmaz(), qutip.sigmaz())
        H_p = qutip.qeye(2)
        for k in range(self.N - 3):
            H_p = qutip.tensor(qutip.qeye(2), H_p)

        H1 = (1 - qutip.tensor(ZZ, H_p)) / 2
        H = [H1.permute(cycle) for cycle in self.cyclic_perm]
        return sum(H)

    def mixer_hamiltonian(self):
        """Generates the Mixer Hamiltonian."""
        
        H_mix = qutip.sigmax()
        for k in range(self.N - 1):
            H_mix = qutip.tensor(qutip.qeye(2), H_mix)

        H_B = [H_mix.permute(self.cyclic_perm[i]) for i in range(self.N)]
        return sum(H_B)

    def vector_input(self):
        """Generates the Vector Input."""
        vec_0 = (qutip.basis(2, 0) + qutip.basis(2, 1)) / np.sqrt(2)
        vec_input = vec_0
        for k in range(self.N - 1):
            vec_input = qutip.tensor(vec_input, vec_0)

        return vec_input

    def optimization(self):
        """Computes the optimal energy from the circuit."""

        H = self.cost_hamiltonian()
        H_B = self.mixer_hamiltonian()
        vec_input = self.vector_input()

        # Optimize layers simultaneously
        if self.method == "collective":
            def cost(angles):
                U = 1
                for i in range(self.layers)[::-1]:
                    U *= (1j * angles[2 * i] * H_B).expm()
                    U *= (1j * angles[2 * i + 1] * H).expm()

                # Cost = |<psi|U' H U|psi>|
                
                vec_var = (U * vec_input)
                return -abs((vec_var.dag() * H * vec_var).tr())

            angles = []
            print("\n\n")
            for i in range(2 * self.layers):
                angle = random.random()
                print(f"Initialized angle {i + 1}: {angle}")
                angles.append(angle)

            print(f"\nOptimizing angles with {self.approach}...\n")
            results = minimize(cost, angles, method=self.approach)
            for i in range(2 * self.layers):
                print(f"Optimized angle {i + 1}: {results.x[i]}")

            return results.x
        
        # Optimize layers individually
        else:
            all_angles = []
            print("\n\n")

            def cost(angles):
                U = (1j * angles[0] * H_B).expm()
                U *= (1j * angles[1] * H).expm()

                # Cost = |<psi|U' H U|psi>| 
                
                vec_var = (U * vec_input)
                return -abs((vec_var.dag() * H * vec_var).tr())
            
            for i in range(self.layers):
                new_angles = [random.random(), random.random()]
                print(f"Initialized Gamma {i + 1}: {new_angles[0]}")
                print(f"Initialized Beta {i + 1}: {new_angles[1]}")

                results = minimize(cost, new_angles, method=self.approach)
                U1 = (1j * results.x[0] * H_B).expm()
                U2 = (1j * results.x[1] * H).expm()
                vec_input = U1 * U2 * vec_input

                all_angles.append(results.x[0])
                all_angles.append(results.x[1])

            print("\n")
            print(f"Optimizing angles with {self.approach}...\n")
            for i in range(self.layers):
                print(f"Optimized Gamma {i + 1}: {all_angles[2 * i]}")
                print(f"Optimized Beta {i + 1}: {all_angles[2 * i + 1]}")

            return all_angles

    def final_energy(self):
        """Calculates the final energy ratio from learned angles."""
        
        H = self.cost_hamiltonian()
        eigenvalues, _ = H.eigenstates()
        
        H_B = self.mixer_hamiltonian()
        vec_input = self.vector_input()
        results = self.optimization()

        U = 1
        for i in range(self.layers)[::-1]:
            U *= (1j * results[2 * i] * H_B).expm()
            U *= (1j * results[2 * i + 1] * H).expm()

        vec_var = (U * vec_input)

        # Returns the ratio of the found solution/the optimal solution
        return abs((vec_var.dag() * H * vec_var).tr())/eigenvalues[-1]

    def analyze(self):
        print(f"\n********** QAOA with {self.N} Qubits, {self.layers} Layers **********")
        print(self.qaoa())
        if self.method == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 4)
        print(f"\nFinal Energy Ratio: {energy}\n")
        return(energy)


class MaxCut(ROD):
    def __init__(self, min_conx, max_conx, density, *args, **kwargs):
        if min_conx > max_conx:
            raise ValueError("Maximum connections must be at least minimum connections.")
        self.min_conx = min_conx
        self.max_conx = max_conx

        # Density = avg number of connections per node
        if density < min_conx or density > max_conx:
            raise ValueError("Density must be between the minimum and maximum connections.")
        self.density = density
        super().__init__(*args, **kwargs)

        self.graph_edges = self.make_graph()

    def make_graph(self):
        """Produces an random edge dictionary for a regular graph within specifications."""
        
        nodes = list(range(self.N))
        edges = {
            node: [] for node in nodes
        }
        edges_count = {
            node: 0 for node in nodes
        }

        # Add minimum value of connections to each point:
        for i in range(self.min_conx):
            for node in nodes:
                if edges_count[node] == self.max_conx:
                    pass
                else:
                    while True:
                        x = np.random.choice(nodes, size=1)[0]
                        if x != node and edges_count[x] < self.max_conx:
                            if x not in edges[node]:
                                break
                    
                    edges[node].append(x)
                    edges[x].append(node)
                    edges_count[node] += 1
                    edges_count[x] += 1

        current_density = sum(edges_count.values())/self.N
        
        # If at or above current density, it's finished
        if current_density >= self.density:
            return edges

        # Otherwise, add extra connections for density:
        for i in range(round(self.N * (self.density - current_density)/2)):
            while True:
                pair = np.random.choice(nodes, size=2, replace=False)
                a = pair[0]
                b = pair[1]
                
                if a in edges[b]:
                    continue

                if edges_count[a] < self.max_conx and edges_count[b] < self.max_conx:
                    break
            
            edges[a].append(b)
            edges[b].append(a)
            edges_count[a] += 1
            edges_count[b] += 1

        return edges

    def cost_hamiltonian(self):
        """Generates the Cost Hamiltonian for spins on a max-cut graph."""

        Zs = []
        for i in range(self.N):
            if i == 0:
                ZZ = qutip.tensor(qutip.sigmaz(), qutip.qeye(2))
                for i in range(self.N - 2):
                    ZZ = qutip.tensor(ZZ, qutip.qeye(2))
                Zs.append(ZZ)

            else:
                ZZ = qutip.qeye(2)
                for j in range(1, self.N):
                    if i == j:
                        ZZ = qutip.tensor(ZZ, qutip.sigmaz())
                    else:
                        ZZ = qutip.tensor(ZZ, qutip.qeye(2))
                Zs.append(ZZ)

        H = 0
        all_edges = self.graph_edges
        for node, edges in all_edges.items():
            Zi = Zs[node]
            for edge in edges:
                Zj = Zs[edge]
                H += (1 - (Zi * Zj)) / 2

        return H

    def analyze(self):
        title = f"Max-Cut {self.max_conx} with {self.N} Qubits, {self.layers} Layers"
        print(f"\n\n********** {title} **********")
        print(self.qaoa())
        print(self.graph_edges)
        if self.method == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 4)
        print(f"\nFinal Energy Ratio: {energy}\n")
        return(energy)


class ROD_PI(ROD):
    def optimization(self):
        """Computes the optimal energy from the circuit."""

        H = self.cost_hamiltonian()
        H_B = self.mixer_hamiltonian()
        vec_input = self.vector_input()

        # Optimize layers simultaneously
        if self.method == "collective":
            def cost(angles):
                U = 1
                for i in range(self.layers)[::-1]:
                    U *= (1j * angles[2 * i] * H_B).expm()
                    U *= (1j * angles[2 * i + 1] * H).expm()

                # Cost = |sqrt(<psi|H^2|psi>)| - |<psi|U' H|psi>|
                vec_var = U * vec_input
                term_one = (vec_input.dag() * (H**2) * vec_input).tr()
                term_two = (vec_var.dag() * H * vec_input).tr()
                return -abs(abs(np.sqrt(term_one)) + abs(term_two))

            angles = []
            print("\n\n")
            for i in range(2 * self.layers):
                angle = random.random()
                print(f"Initialized angle {i + 1}: {angle}")
                angles.append(angle)

            print(f"\nOptimizing angles with {self.approach}...\n")
            results = minimize(cost, angles, method=self.approach)
            for i in range(2 * self.layers):
                print(f"Optimized angle {i + 1}: {results.x[i]}")

            return results.x

        # Optimize layers individually
        else:
            all_angles = []
            print("\n\n")

            def cost(angles):
                U = (1j * angles[0] * H_B).expm()
                U *= (1j * angles[1] * H).expm()

                # Cost = |sqrt(<psi|H^2|psi>)| - |<psi|U' H|psi>| 
                vec_var = (U * vec_input)
                term_one = (vec_input.dag() * (H**2) * vec_input).tr()
                term_two = (vec_var.dag() * H * vec_input).tr()
                return -abs(abs(np.sqrt(term_one)) + abs(term_two))
            
            for i in range(self.layers):
                new_angles = [random.random(), random.random()]
                print(f"Initialized Gamma {i + 1}: {new_angles[0]}")
                print(f"Initialized Beta {i + 1}: {new_angles[1]}")

                results = minimize(cost, new_angles, method=self.approach)
                U1 = (1j * results.x[0] * H_B).expm()
                U2 = (1j * results.x[1] * H).expm()
                vec_input = U1 * U2 * vec_input

                all_angles.append(results.x[0])
                all_angles.append(results.x[1])

            print("\n")
            print(f"Optimizing angles with {self.approach}...\n")
            for i in range(self.layers):
                print(f"Optimized Gamma {i + 1}: {all_angles[2 * i]}")
                print(f"Optimized Beta {i + 1}: {all_angles[2 * i + 1]}")

            return all_angles

    def analyze(self):
        title = f"Power Iteration QAOA with {self.N} Qubits, {self.layers} Layers"
        print(f"\n******* {title} *******")
        print(self.qaoa())
        if self.method == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 4)
        print(f"\nFinal Energy Ratio: {energy}\n")
        return(energy)
        

class MaxCut_PI(MaxCut, ROD_PI):
    def __init__(self, *args, **kwargs):
        super(MaxCut).__init__(*args, **kwargs)
        
    def cost_hamiltonian(self):
        return super(MaxCut).cost_hamiltonian()

    def optimization(self):
        return super(ROD_PI).optimization()

    def analyze(self):
        title = f"P.I. Max-Cut {self.max_conx} with {self.N} Qubits, {self.layers} Layers"
        print(f"\n\n********** {title} **********")
        print(self.graph_edges)
        print(self.qaoa())
        if self.method == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 4)
        print(f"\nFinal Energy Ratio: {energy}\n")
        return(energy)
    
    
class Simulation():
    def __init__(self, layers, method, approach):
        self.layers = layers
        self.method = method
        self.approach = approach
        if approach not in ["collective", "individual"]:
            raise ValueError("Approach must be collective or individual.")

    def energies(self):
        # Y variable: energies
        energies = []

        # Categorical variable: layers
        for layers in range(self.layers):
            this_layer = []

            # X variable: number of qubits
            for N in range(4, 9, 2):
                circuit = ROD(N, layers + 1, self.method, approach=self.approach)
                energy = circuit.analyze()
                this_layer.append(energy)

            energies.append(this_layer)

        return energies

    def graph(self):
        energies = self.energies()
        labels = [f"{i + 1} Layers" for i in range(self.layers)]
        labels[0] = "1 Layer"
        plt.style.use("seaborn-darkgrid")
        ax = plt.subplot()
        for i in range(self.layers):
            plt.plot(range(4, 9, 2), energies[i])

        approach = "Collectively" if self.approach == "collective" else "Individually"
        plt.legend(labels, prop={"family": "Times New Roman"})
        plt.title(f"Optimization Library: {self.method}", fontsize=12)
        plt.suptitle(f"Ring of Disagrees QAOA, Optimized {approach}", fontsize=14)
        plt.xlabel("Number of Qubits")
        plt.ylabel("Final Energy State")
        ax.set_xticks(range(4, 9, 2))
        plt.show()
        

if __name__ == "__main__":
    """Defines a grid of (N = ncols x nrows) qubits.
    Args:
        N: number of qubits
        layers: number of Cost Hamiltonian + Mixer Hamiltonian Layers
        method: which optimization method to use from the following options:
            ["BFGS", "Nelder-Mead", "Powell", "CG"]
        
    Kwargs:    
        approach: either "collective" (minimizing all angles simultaneously) 
            or "individual" (minimizing two angles at a time, then plugging 
            that circuit back in) -> Default "collective"
    """

    # Results from ROD Simulation:
    # Qubits (4, 6, 8), Layers Collectively:
    # 1 Layer: (0.75, 0.75, 0.75); 2 Layers: (1, 0.833, 0.833)
    # 3 Layers: (1, 1, 0.875); 4 Layers: (1, 1, 1)

    # Qubits (4, 6, 8), Layers Individually:
    # 1 Layer: (0.75, 0.75, 0.75); 2 Layers: (0.913, 0.765, 0.765)
    # 3 Layers: (0.953, 0.767, 0.766); 4 Layers: (0.956, 0.767, 0.766)
    # 5 Layers: (0.956, 0.767, 0.766)

    circuit = MaxCut(1, 3, 2, 4, 3, "Powell")
    circuit.analyze()
