# Thermodynamic Perception on a 4-Qubit Transmon Lattice

**A Research-Grade Open Quantum System Simulation**

This repository contains a Python simulation of a **4-qubit Transmon-style lattice** evolving under a noisy GKSL master equation. The project explores the intersection of quantum thermodynamics and machine learning, implementing a **Thermodynamic Neuron** model inspired by recent breakthroughs in autonomous quantum thermal machines.



> **🔒 Access Note:** The full source code (`quantum_unified_revised_v136.py` / `v23`) is available upon request for collaboration purposes.  
> **Contact:** [csalnav2@gmail.com](mailto:csalnav2@gmail.com)

---

## ⚛️ Theoretical Framework & Mathematical Model

### 1. The Hamiltonian (Transmon Lattice)
The system models 4 Transmon qubits as Duffing oscillators truncated to the qubit subspace. They are driven by time-dependent fields (Floquet engineering) and coupled via exchange and inductive interactions.

The total lattice Hamiltonian $H(t)$ is:

$$
H(t) = \sum_{j=1}^4 \left[ \frac{\omega_{01,j}(t)}{2} \sigma_z^{(j)} + \frac{\Omega_d(t)}{2} \left( \sigma_x^{(j)} \cos(\phi_j(t)) + \sigma_y^{(j)} \sin(\phi_j(t)) \right) \right] + H_{\text{int}}
$$

The interaction term $H_{\text{int}}$ includes tunable capacitive ($XX+YY$) and inductive ($ZZ$) couplings. Note the explicit spacing in the capacitive term to distinguish the coupling constant:

$$
H_{\text{int}} = \sum_{\langle i,j \rangle} \left[ J_{\text{cap}} \, \left( \sigma_x^{(i)}\sigma_x^{(j)} + \sigma_y^{(i)}\sigma_y^{(j)} \right) + J_{\text{ind}} \, \sigma_z^{(i)}\sigma_z^{(j)} \right]
$$

### 2. Open System Dynamics (GKSL)
The system evolves according to the **Lindblad (GKSL) Master Equation**, which accounts for dissipation and decoherence due to coupling with a thermal bath:

$$
\frac{d\rho}{dt} = -i [H(t), \rho] + \sum_{j=1}^4 \left( \gamma_{\downarrow,j} \mathcal{D}[\sigma_-^{(j)}] \rho + \gamma_{\uparrow,j} \mathcal{D}[\sigma_+^{(j)}] \rho + \gamma_{\phi,j} \mathcal{D}[\sigma_z^{(j)}] \rho \right)
$$

* **Dissipator:** $\mathcal{D}[L]\rho = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho\}$
* **Detailed Balance:** The rates $\gamma_{\uparrow/\downarrow}$ are determined by the instantaneous bath temperature $T_b(t)$, ensuring the system relaxes toward a thermal state in the absence of driving.

### 3. The Thermodynamic Neuron & Feature Lifting
Inspired by *Lipka-Bartosik et al. (2024)*, this simulation tests if a quantum system can act as a perceptron. We map input data $\vec{u}$ into the bath parameters and measure a final observable $O$ (e.g., excited population).

$$
O(\vec{u}) \approx \sigma \left( \vec{w} \cdot \vec{u} + b \right)
$$

**Quantum Feature Lift (Logic Upgrade):**
A standard perceptron is limited to linearly separable logic (AND, OR). This lattice, however, acts as a high-dimensional reservoir.
* **The Lift:** The interaction $H_{\text{int}}$ entangles the qubits, mapping simple inputs into a complex, high-dimensional Hilbert space.
* **Solving XOR:** By upgrading the readout from simple local populations ($\langle \sigma_z \rangle$) to **non-linear quantum correlations** (such as Log-Negativity or Mutual Information), the system performs a "Feature Lift." This allows the "neuron" to solve non-linearly separable problems (like **XOR**) by exploiting the physical entanglement resources of the lattice rather than requiring a hidden neural layer.



### 4. Thermodynamic Length & Dissipation
To quantify the "cost" of information processing, we compute the **Bures Length** (Thermodynamic Length). Following *Deffner (2013)* and *Scandi & Perarnau-Llobet (2019)*, the distance the state travels in statistical manifold space is lower-bounded by the entropy production (dissipation).

$$
\mathcal{L} = \int_0^\tau \sqrt{g_{\mu\nu} \dot{\lambda}^\mu \dot{\lambda}^\nu} dt \quad \propto \quad \text{Dissipation Cost}
$$

The simulation tracks this length to identify "Hotspots"—moments of intense information erasure or irreversible dynamics.

---

## 📊 Dashboard Features
The script generates a high-resolution animated dashboard (GIF/MP4) containing:
* **Bloch Spheres:** 4 interactive spheres with state trails and $T_1/T_2$ visualizers.
* **Phase Space:** Local and Collective Lattice Wigner function snapshots. 
* **Diagnostics:** Real-time plots of Quantum Fisher Information (QFI), Bures Length, OTOC (Out-of-Time-Order Correlator), and Choi spectrum.
* **Logic Test:** An HTML report demonstrating the **Floquet Driven Thermodynamic-Neuron** solving logic gates (NOT, NOR, 3-MAJORITY).

---

## 💻 Installation

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/quantum-thermodynamic-neuron.git
cd quantum-thermodynamic-neuron

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

📚 References
Thermodynamic Computing & Neurons

Lipka-Bartosik, Perarnau-Llobet, Brunner. Thermodynamic computing via autonomous quantum thermal machines. Science Advances 10(36):eadm8792 (2024).

DOI: 10.1126/sciadv.adm8792

Thermodynamic Length & Metrics

S. Deffner. Thermodynamic length for far-from-equilibrium quantum systems. Phys. Rev. E 87, 022143 (2013).

DOI: 10.1103/PhysRevE.87.022143

M. Scandi, M. Perarnau-Llobet. Thermodynamic length in open quantum systems. Quantum 3, 197 (2019).

DOI: 10.22331/q-2019-10-24-197

C. Cafaro et al. Bures and Sjöqvist metrics over thermal state manifolds for spin qubits and superconducting flux qubits. arXiv:2303.01680 (2023).

arXiv:2303.01680

P. M. Alsing et al. Comparing metrics for mixed quantum states: Sjöqvist and Bures. Phys. Rev. A 107, 052411 (2023).

DOI: 10.1103/PhysRevA.107.052411

Transmon Qubits & Hardware

J. Koch et al. Charge-insensitive qubit design derived from the Cooper pair box. Phys. Rev. A 76, 042319 (2007).

DOI: 10.1103/PhysRevA.76.042319

M. Kjaergaard et al. Superconducting qubits: Current state of play. Annu. Rev. Condens. Matter Phys. 11, 369–395 (2020).

DOI: 10.1146/annurev-conmatphys-031119-050605

O. Kyriienko et al. Floquet Quantum Simulation with Superconducting Qubits. Phys. Rev. Applied 9, 064029 (2018).

DOI: 10.1103/PhysRevApplied.9.064029

L. Xiang et al. Long-lived topological time-crystalline order on a quantum processor. Nat. Commun. 15, 5570 (2024).

DOI: 10.1038/s41467-024-49923-z

Open Systems & Software

D. Manzano. A short introduction to the Lindblad master equation. AIP Adv. 10, 025106 (2020).

DOI: 10.1063/1.5115323

QuTiP: J. R. Johansson et al. QuTiP: An open-source Python framework for the dynamics of open quantum systems. Comput. Phys. Commun. 183, 1760 (2012).

DOI: 10.1016/j.cpc.2012.02.021

QuTiP 2: J. R. Johansson et al. QuTiP 2: A Python framework for the dynamics of open quantum systems. Comput. Phys. Commun. 184, 1234 (2013).

DOI: 10.1016/j.cpc.2012.11.019
