import torch

from QRL.learning.approximate.quantum_model import get_quantum_neural_network
num_qubits=4
n_layers=6
enable_rx=True
enable_ry=True
enable_rz=True

model=get_quantum_neural_network(num_qubits,n_layers,False,True,True)

