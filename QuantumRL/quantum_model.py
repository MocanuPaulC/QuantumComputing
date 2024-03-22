from gymnasium.envs.classic_control import CartPoleEnv
from qiskit import QuantumCircuit
import qiskit as qk
import numpy as np
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import ParameterVector
from torch import Tensor
import torch


def one_qubit_rotation(qc, weights, gate_types_per_layer, enable_rx=True, enable_ry=True, enable_rz=True):
    """
    Adds rotation gates around the X, Y, and Z axis to the quantum circuit for a specific qubit,
    with the rotation angles specified by the values in `weights`.
    """
    if len(weights) != gate_types_per_layer * len(qc.qubits):
        print(f"len weights {len(weights)}")
        print(f"gate types per layer {gate_types_per_layer}")
        print(f"len qubits {len(qc.qubits)}")
        print(f"len qubits * gate types per layer {len(qc.qubits) * gate_types_per_layer}")
        raise ValueError(
            "The number of weights must be equal to the number of qubits times the number of gate types per layer")

    if not enable_rx and not enable_ry and not enable_rz:
        raise ValueError("At least one gate must be enabled")

    enabled_gates = sum([enable_rx, enable_ry, enable_rz])  # Count how many gates are enabled
    if len(weights) != enabled_gates * len(qc.qubits):
        print(len(weights))
        print(enabled_gates)
        print(len(qc.qubits))
        print(len(qc.qubits) * enabled_gates)
        raise ValueError("The number of weights does not match the number of enabled gates times the number of qubits")

    for qubit in qc.qubits:
        index = qc.qubits.index(qubit)
        weight_index = 0  # Initialize weight index for each qubit

        if enable_rx:
            qc.rx(weights[index + weight_index * len(qc.qubits)], index)  # Rotate around X-axis
            weight_index += 1  # Move to the next set of weights

        if enable_ry:
            qc.ry(weights[index + weight_index * len(qc.qubits)], index)  # Rotate around Y-axis
            weight_index += 1  # Move to the next set of weights

        if enable_rz:
            qc.rz(weights[index + weight_index * len(qc.qubits)], index)  # Rotate around Z-axis
            # No need to increase weight_index here since it's the last operation


def entangling_layer(qc: QuantumCircuit):
    """
    Adds a layer of CZ entangling gates (controlled-Z) on `qubits` (arranged in a circular topology) to the quantum circuit.
    """
    # Assume `qc` is your QuantumCircuit object that's defined outside this function
    if qc.num_qubits > 2:  # If more than 2 qubits, connect the first and last qubits to form a circle
        qc.cz(0, qc.num_qubits - 1)
    for i in range(qc.num_qubits - 1):
        qc.cz(i, i + 1)  # Apply CZ between consecutive qubits

def generate_circuit(qc, n_layers, enable_rx=True, enable_ry=True, enable_rz=True):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = qc.num_qubits
    # if gate is enabled, add a parameter for each qubit and for each layer

    gate_types_per_layer = 0
    if enable_rx:
        gate_types_per_layer += 1
    if enable_ry:
        gate_types_per_layer += 1
    if enable_rz:
        gate_types_per_layer += 1
    if gate_types_per_layer == 0:
        raise ValueError("At least one gate must be enabled")
    if n_layers < 1:
        raise ValueError("At least one layer is required")
    if n_qubits < 1:
        raise ValueError("At least one qubit is required")

    params = ParameterVector("theta", gate_types_per_layer * (n_layers + 1) * n_qubits)
    inputs = ParameterVector("inputs", n_qubits)

    for i in range(n_layers):
        for j in range(n_qubits):
            qc.rx(inputs[j], j)
        # Variational layer
        if i == 0:
            one_qubit_rotation(qc, params[0:gate_types_per_layer * n_qubits], gate_types_per_layer=gate_types_per_layer,
                               enable_rx=enable_rx, enable_ry=enable_ry, enable_rz=enable_rz)
        else:
            one_qubit_rotation(qc,
                               params[i * gate_types_per_layer * n_qubits:(i + 1) * gate_types_per_layer * n_qubits],
                               gate_types_per_layer=gate_types_per_layer, enable_rx=enable_rx, enable_ry=enable_ry,
                               enable_rz=enable_rz)
        entangling_layer(qc)

    return qc, list(inputs)





class encoding_layer(torch.nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()

        # Define weights for the layer
        weights = torch.Tensor(num_qubits)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, -1, 1)  # <--  Initialization strategy

    def forward(self, x):
        """Forward step, as explained above."""

        if not isinstance(x, Tensor):
            x = Tensor(x)

        x = self.weights * x
        x = torch.atan(x)

        return x


class exp_val_layer(torch.nn.Module):
    def __init__(self, action_space=2):
        super().__init__()

        # Define the weights for the layer
        weights = torch.Tensor(action_space)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, 35, 40)  # <-- Initialization strategy (heuristic choice)

        # Masks that map the vector of probabilities to <Z_0*Z_1> and <Z_2*Z_3>
        self.mask_ZZ_12 = torch.tensor([1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1.],
                                       requires_grad=False)
        self.mask_ZZ_34 = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1.],
                                       requires_grad=False)

    def forward(self, x):
        """Forward step, as described above."""

        expval_ZZ_12 = self.mask_ZZ_12 * x
        expval_ZZ_34 = self.mask_ZZ_34 * x

        # Single sample
        if len(x.shape) == 1:
            expval_ZZ_12 = torch.sum(expval_ZZ_12)
            expval_ZZ_34 = torch.sum(expval_ZZ_34)
            out = torch.cat((expval_ZZ_12.unsqueeze(0), expval_ZZ_34.unsqueeze(0)))

        # Batch of samples
        else:
            expval_ZZ_12 = torch.sum(expval_ZZ_12, dim=1, keepdim=True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34, dim=1, keepdim=True)
            out = torch.cat((expval_ZZ_12, expval_ZZ_34), 1)

        return self.weights * ((out + 1.) / 2.)


def get_quantum_neural_network(n_qubits=4, n_layers=8, enable_rx=False, enable_ry=True, enable_rz=True):
    qc = QuantumCircuit(n_qubits)
    circuit, inputs = generate_circuit(qc, n_layers, enable_rx=enable_rx, enable_ry=enable_ry,
                                               enable_rz=enable_rz)
    # The remaining ones are the trainable weights of the quantum neural network
    params = list(qc.parameters)[n_qubits:]

    samplerQNN = SamplerQNN(circuit=qc, input_params=inputs, weight_params=params)
    qnn = TorchConnector(samplerQNN)
    encoding = encoding_layer(n_qubits)

    exp_val = exp_val_layer()

    model = torch.nn.Sequential(encoding, qnn, exp_val)
    return model

def epsilon_greedy_policy(state, epsilon=0,model=None):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        with torch.no_grad():
            Q_values = model(Tensor(state)).numpy()
        return np.argmax(Q_values)


def play_one_step(env, state, epsilon,model):
    action = epsilon_greedy_policy(state, epsilon,model)
    next_state, reward, done,truncated, info = env.step(action)
    return next_state, reward, done,truncated, info
