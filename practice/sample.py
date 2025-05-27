from qiskit import QuantumCircuit, Aer, execute, IBMQ
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_gate_map
from qiskit.quantum_info import state_fidelity
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Circuit Construction
qc = QuantumCircuit(3, 2)
# Prepare arbitrary state on sender qubit (qubit 0)
theta, phi, lam = np.pi/3, np.pi/4, np.pi/2
qc.u3(theta, phi, lam, 0)
# Entangle qubits 1 & 2 into Bell state
qc.h(1)
qc.cx(1, 2)
# Bell measurement on sender side
qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])

# 2. Execution on Simulators
backend_sv = Aer.get_backend('statevector_simulator')
result_sv = execute(qc, backend_sv).result()
state_before = result_sv.get_statevector()

# Apply correction conditionally on classical bits
qc_corr = qc.copy()
qc_corr.x(2).c_if(qc.clbits[1], 1)
qc_corr.z(2).c_if(qc.clbits[0], 1)

backend_qasm = Aer.get_backend('qasm_simulator')
result_qasm = execute(qc, backend_qasm, shots=1024).result()
counts = result_qasm.get_counts()

# 3. Visualization
# Circuit diagram
qc.draw('mpl')
plt.show()

# Bloch vector before teleportation
plot_bloch_multivector(state_before)
plt.title('Before Teleportation')
plt.show()

# Bloch vector after teleportation
state_after = execute(qc_corr, backend_sv).result().get_statevector()
plot_bloch_multivector(state_after)
plt.title('After Teleportation')
plt.show()

# Measurement histogram
plot_histogram(counts)
plt.show()

# 4. Result Analysis
fidelity = state_fidelity(state_before, state_after)
print(f'Fidelity: {fidelity:.4f}')

# 5. Tabulate Results for Multiple Input States
results = []
test_angles = [
    (0, 0, 0),               # |0>
    (np.pi/2, 0, 0),         # |+>
    (np.pi/3, np.pi/4, 0)    # arbitrary
]
for th, ph, la in test_angles:
    qc2 = QuantumCircuit(3, 2)
    qc2.u3(th, ph, la, 0)
    qc2.h(1); qc2.cx(1, 2)
    qc2.cx(0, 1); qc2.h(0)
    qc2.measure([0,1],[0,1])
    state0 = execute(qc2, backend_sv).result().get_statevector()
    qc2_corr = qc2.copy()
    qc2_corr.x(2).c_if(qc2.clbits[1], 1)
    qc2_corr.z(2).c_if(qc2.clbits[0], 1)
    state_after0 = execute(qc2_corr, backend_sv).result().get_statevector()
    results.append((th, ph, la, state_fidelity(state0, state_after0)))

df = pd.DataFrame(results, columns=['θ','φ','λ','Fidelity'])
print(df)

# 6. Real-Device Coupling Map
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend_real = provider.get_backend('ibmq_quito')
plot_gate_map(backend_real)
plt.show()
