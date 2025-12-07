Quantum Thermodynamic Neuron on a 4‑Qubit Transmon Lattice

This repo contains a Python simulation of a **4‑qubit transmon‑style lattice**
evolving under a noisy GKSL master equation with mean‑field couplings,
square‑wave bath driving, and diagnostics inspired by a **thermodynamic neuron**
 developed by Lipka‑Bartosik *et al.*,Science Advances 4 Sep 2024 Vol 10, Issue 36 https://www.science.org/doi/10.1126/sciadv.adm8792
 
 Other citations:
 Bures angle = thermodynamic length far from equilibrium

S. Deffner, “Thermodynamic length for far‑from‑equilibrium quantum systems,” Phys. Rev. E 87, 022143 (2013).
Defines Bures angle as a thermodynamic length and connects it directly to dissipation in quantum systems. 
APS Links

Thermodynamic length for GKSL / Lindblad open systems

M. Scandi and M. Perarnau‑Llobet, “Thermodynamic length in open quantum systems,” Quantum 3, 197 (2019).
Builds a thermodynamic metric directly for Lindblad dynamics, which is basically what your GKSL code is simulating. 
Quantum Journal
+1

Bures vs Sjöqvist metrics over thermal state manifolds

C. Cafaro et al., “Bures and Sjöqvist metrics over thermal state manifolds for spin qubits and superconducting flux qubits,” arXiv:2303.01680 (2023).
Explicitly compares Bures and Sjöqvist metrics in thermal-state landscapes, including superconducting qubit models. 
arXiv

Comparing Sjöqvist and Bures more generally

P. M. Alsing et al., “Comparing metrics for mixed quantum states: Sjöqvist and Bures,” Phys. Rev. A 107, 052411 (2023).
Detailed math on how Sjöqvist and Bures metrics relate for mixed states—good to cite if you ever explicitly mention Sjöqvist.

Open quantum systems / GKSL master equation

Your whole engine lives inside a GKSL / Lindblad master equation, so it’s good to nod to that:

D. Manzano, “A short introduction to the Lindblad master equation,” AIP Adv. 10, 025106 (2020), arXiv:1906.04478.
Very readable intro to the GKSL equation and its structure. 
arXiv
+1

(If you want a “big textbook” reference, you can also mention Rivas & Huelga’s Open Quantum Systems, but that’s optional.)

Transmon qubits and lattices

You’re explicitly calling this a “transmon lattice” and using T₁/T₂ language etc., so I’d anchor that in the standard transmon literature:

Canonical transmon paper

J. Koch et al., “Charge‑insensitive qubit design derived from the Cooper pair box,” Phys. Rev. A 76, 042319 (2007).
This is the transmon paper explaining the EJ/EC regime and noise suppression. 
APS Links
+1

Standard review of superconducting qubits & lattices

M. Kjaergaard et al., “Superconducting qubits: Current state of play,” Annu. Rev. Condens. Matter Phys. 11, 369–395 (2020).
Reviews transmon lattices / architectures and is a nice pointer for “we are inspired by typical transmon arrays used in current processors.” 
Annual Reviews
+1

Floquet / periodically driven transmon arrays

O. Kyriienko et al., “Floquet Quantum Simulation with Superconducting Qubits,” Phys. Rev. Applied 9, 064029 (2018).
Shows how periodic modulation of transmons is used for Floquet engineering of spin models. 
APS Links

Explicit transmon lattice experiments

L. Xiang et al., “Long‑lived topological time‑crystalline order on a quantum processor,” Nat. Commun. 15, 5570 (2024).
They use an 18‑transmon 2D lattice to realize Floquet time‑crystalline phases—nice “real hardware” counterpart to your simulated transmon lattice

J. R. Johansson, P. D. Nation, and F. Nori,
“QuTiP: An open-source Python framework for the dynamics of open quantum systems,” Comput. Phys. Commun. 183, 1760–1772 (2012). 
ScienceDirect
+1

J. R. Johansson, P. D. Nation, and F. Nori,
“QuTiP 2: A Python framework for the dynamics of open quantum systems,” Comput. Phys. Commun. 184, 1234–1240 (2013). 

The script generates a high‑resolution animated dashboard (GIF) with:

- 4 Bloch spheres with trails and T₁/T₂/η info
- Per‑qubit phase histograms, density bars, and local Wigner functions
- A collective lattice Wigner snapshot
- Thermodynamic diagnostics (QFI, Bures length, entropy production, OTOC, Choi spectrum, etc.)
- A simple **Floquet Driven thermodynamic‑neuron logic self‑test** (NOT, NOR, 3‑MAJORITY) displayed in an HTML dashboard

## Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/quantum-thermodynamic-neuron.git
cd quantum-thermodynamic-neuron

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt

How to ctivate a Single Mode(1 Qubit) : python quantum_unified_revised_v23.py --mode single --bath_enable --q_tmax 10.0

How to activate a Lattice Mode ( 4 interacting Qubits) : python quantum_unified_revised_v23.py --mode lattice --bath_enable --q_tmax 12.0 --fps 18




