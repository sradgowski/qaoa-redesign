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
                print(f"Optimized Beta {i + 1}: {all_angles[2 * i + 1]}")

            return all_angles

    def final_energy(self):
        H, H0, eigenvalues_H0 = self.cost_hamiltonian()
        H_mix = self.mixer_hamiltonian()
        vec_input = self.vector_input()
        results = self.optimization()

        U = 1
        for i in range(self.layers)[::-1]:
            U *= (1j * results[2 * i] * H_mix).expm()
            U *= (1j * results[2 * i + 1] * H0).expm()

        vec_var = (U * vec_input)
        eigenvalue_H = self.N - eigenvalues_H0[0]

        return abs((vec_var.dag() * H * vec_var).tr())/eigenvalue_H

    def analyze(self):
        print(f"\n******* Power Iteration QAOA with {self.N} Qubits, {self.layers} Layers *******")
        # print(self.qaoa())
        if self.approach == "collective":
            print(f"\n\n********** Optimizing Layers Simultaneously **********")
        else:
            print(f"\n\n********** Optimizing Layers Individually **********")
        
        energy = round(self.final_energy(), 3)
        print(f"\nFinal Energy: {energy}\n")
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

    #circuit = PowerIteration(4, 3, "Powell", "individual")
    #circuit.analyze()
    sim = Simulation(4, "Powell", "individual")
    sim.graph()