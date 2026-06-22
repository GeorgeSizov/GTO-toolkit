"""sandbox"""

from pathlib import Path
from GTO_toolkit.input import load_basis_input

input_basis = Path("test_input.txt")
N, K, Geom, Bgen, Bexp, E_nucl = load_basis_input(input_basis)
print("Bexp = ", Bexp)

