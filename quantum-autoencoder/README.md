# Quantum Autoencoder

> Author: Peter Buschenreiter
> 
> Supporting Authors:
> - Paul-Cristian Mocanu
> - Seifeldin Sabry
> 
> KdG Coaches: 
> - Hans Vochten
> - Geert De Paepe
> 
> IBM supervisor: Eric Michiels

The goal of this project is to create a quantum circuit that can encode and compress images from the MNIST dataset.

Expanded from [this original tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/12_quantum_autoencoder.html).

### Test results

| Amount of digits | Images per digit | Latent space | Trash space | Iterations | Max fidelity | Min fidelity | 0 fidelity           | 1 fidelity          | 2 fidelity | 3 fidelity | 4 fidelity | 5 fidelity | 6 fidelity | 7 fidelity | 8 fidelity | 9 fidelity |
|------------------|------------------|--------------|-------------|------------|--------------|--------------|----------------------|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|
| 2                | 200              | 6            | 2           | 500        | -            | -            | 0.014805632615888497 | 0.04210399778807985 | -          | -          | -          | -          | -          | -          | -          | -          |
| 10               | 50               | 6            | 2           | 500        | 0.0751       | 0.0028       | 0.011841             | 0.039346            | 0.014605   | 0.013678   | 0.023822   | 0.015282   | 0.016249   | 0.019010   | 0.016061   | 0.020332   |