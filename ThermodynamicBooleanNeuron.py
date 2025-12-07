%%writefile quantum_unified_revised_v23.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_unified_revised_v23.py

4-Qubit Transmon Lattice Dashboard with Thermodynamic-Neuron Style Diagnostics.

This version tweaks v21 as follows:

- Moves the "Qubit 1–4" titles higher above the Bloch info boxes so they no
  longer collide.
- Reduces vertical whitespace between the Bloch-sphere row and the
  Phase/Density row, while keeping the Bloch spheres and Wigner plots the same
  visual size.
- Uses fixed, data-derived y-limits for all diagnostics panels to avoid
  flicker/jumping and keep curves on-screen.
- Adds progress printing even when imageio is not available (Pillow writer).
- Thins Bloch-sphere trails by ~40% while keeping them fully opaque to avoid a
  "crayon/lipstick" look.
- Increases resolution and apparent size of dynamic 3D Wigner plots via a
  denser phase-space grid, smooth shading, and a closer camera.
- Further adjusts aspect ratio of the dashboard (wider, shorter) to make
  diagnostics plots more landscape and reduces dead white space under the
  Bloch row.
- Doubles Bloch trail width again (for a more visible trace) while keeping
  a solid, non-transparent color.

Thermodynamic neuron definitions are inspired by Lipka-Bartosik et al.,
Sci. Adv. 10, eadm8792 (2024).
"""

from __future__ import annotations

import argparse
import math
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib

# Headless backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Optional deps
HAVE_QUTIP = False
try:
    from qutip import Qobj
    from qutip.wigner import wigner as qutip_wigner

    HAVE_QUTIP = True
except Exception:
    Qobj = None  # type: ignore
    qutip_wigner = None  # type: ignore

HAVE_IMAGEIO = False
try:
    import imageio

    HAVE_IMAGEIO = True
except Exception:
    imageio = None  # type: ignore

HAVE_NGROK = False
try:
    from pyngrok import ngrok

    HAVE_NGROK = True
except Exception:
    ngrok = None  # type: ignore

# Global style
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["axes.linewidth"] = 1.4
matplotlib.rcParams["font.size"] = 13
matplotlib.rcParams["font.weight"] = "bold"

# ------------------ CONSTANTS & STYLE TUNING ---------------------------------

kB = 1.380649e-23
Id2 = np.eye(2, dtype=complex)

# Bloch sphere presentation
# Radius doubled; zoom adjusted so spheres stay in frame.
BLOCH_RADIUS = 6.0          # Larger sphere radius
BLOCH_ZOOM = 11.0
BLOCH_VEC_LW = 4.0
BLOCH_TRAIL_LW = 4.2        # 2× thicker trail, still solid and fully opaque
BLOCH_TRAIL_ALPHA = 1.0     # Fully opaque trail

DENSITY_HEIGHT_SCALE = 1.1

# Visual colors
QUBIT_COLORS = [
    "#FF7518",  # Q1 - pumpkin orange
    "#1f77b4",  # Q2 - blue
    "#2ca02c",  # Q3 - green
    "#d62728",  # Q4 - red
]

TL_COLORS = QUBIT_COLORS[:]  # Thermodynamic length TL1..TL4

COLOR_PHASE_BAR = "#A6D6FF"
COLOR_PHASE_DOT = "#0055A4"
COLOR_DENS_BAR = "#800000"
COLOR_DENS_EDGE = "black"

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


# ------------------------ BASIC QUANTUM HELPERS ------------------------------

def vn_entropy_bits(rho: np.ndarray) -> float:
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2).real
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.trace(rho @ sigma_x).real,
            np.trace(rho @ sigma_y).real,
            np.trace(rho @ sigma_z).real,
        ],
        dtype=float,
    )


def fidelity_bloch(rho: np.ndarray, sigma: np.ndarray) -> float:
    r = bloch_vector(rho)
    s = bloch_vector(sigma)
    r2 = float(r @ r)
    s2 = float(s @ s)
    rs = float(r @ s)
    term = max(0.0, (1 - r2) * (1 - s2))
    F = 0.5 * (1 + rs + math.sqrt(term))
    return float(min(max(F, 0.0), 1.0))


def bures_angle(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Bures angle between two qubit density matrices.

    For pure states this coincides with the Fubini–Study metric distance.
    We treat it as a thermodynamic length on the state manifold.
    """
    F = fidelity_bloch(rho, sigma)
    return float(math.acos(math.sqrt(min(max(F, 0.0), 1.0))))


def qfi_state_sensitivity(rho: np.ndarray, G: np.ndarray) -> float:
    lam, U = np.linalg.eigh((rho + rho.conj().T) / 2)
    qfi = 0.0
    for i in range(2):
        for j in range(2):
            denom = lam[i] + lam[j]
            if denom > 1e-15:
                gij = np.vdot(U[:, i], G @ U[:, j])
                diff = lam[i] - lam[j]
                qfi += 2.0 * (diff * diff / denom) * (abs(gij) ** 2)
    return float(max(0.0, qfi.real))


def coherent_step(omega_vec: np.ndarray, dt: float) -> np.ndarray:
    om = float(np.linalg.norm(omega_vec))
    if om < 1e-14:
        return Id2.copy()
    n = omega_vec / om
    theta = 0.5 * om * dt
    return math.cos(theta) * Id2 - 1j * math.sin(theta) * (
        n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
    )


def kraus_dephase(p: float):
    p = float(np.clip(p, 0.0, 1.0))
    return [math.sqrt(1 - p) * Id2, math.sqrt(p) * sigma_z]


def kraus_amp_down(g: float):
    g = float(np.clip(g, 0.0, 1.0))
    K0 = np.array([[1, 0], [0, math.sqrt(max(0.0, 1 - g))]], dtype=complex)
    K1 = np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)
    return [K0, K1]


def kraus_amp_up(g: float):
    g = float(np.clip(g, 0.0, 1.0))
    K0 = np.array([[math.sqrt(max(0.0, 1 - g)), 0], [0, 1]], dtype=complex)
    K1 = np.array([[0, 0], [math.sqrt(g), 0]], dtype=complex)
    return [K0, K1]


def apply_kraus(rho: np.ndarray, Ks: List[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(rho)
    for K in Ks:
        acc += K @ rho @ K.conj().T
    return acc


def compose_kraus(U: np.ndarray, maps: List[List[np.ndarray]]) -> List[np.ndarray]:
    eff: List[np.ndarray] = [U]
    for Ks in maps:
        new: List[np.ndarray] = []
        for Keff in eff:
            for K in Ks:
                new.append(K @ Keff)
        eff = new
    return eff


def choi_from_kraus(kraus_ops: List[np.ndarray]) -> np.ndarray:
    if not kraus_ops:
        kraus_ops = [Id2.copy()]
    d = kraus_ops[0].shape[0]
    J = np.zeros((d * d, d * d), dtype=complex)
    for K in kraus_ops:
        v = K.flatten(order="F")
        J += np.outer(v, v.conj())
    return J


def vector_slerp(v0: np.ndarray, v1: np.ndarray, alpha: float) -> np.ndarray:
    len0 = float(np.linalg.norm(v0))
    len1 = float(np.linalg.norm(v1))
    if len0 < 1e-9 or len1 < 1e-9:
        return (1 - alpha) * v0 + alpha * v1
    u0 = v0 / len0
    u1 = v1 / len1
    dot = float(np.clip(u0 @ u1, -1.0, 1.0))
    omega = math.acos(dot)
    if abs(omega) < 1e-9:
        return u0 * ((1 - alpha) * len0 + alpha * len1)
    s0 = math.sin((1 - alpha) * omega) / math.sin(omega)
    s1 = math.sin(alpha * omega) / math.sin(omega)
    return (s0 * u0 + s1 * u1) * ((1 - alpha) * len0 + alpha * len1)


def virtual_temperature_from_pe(pe: float, DeltaE: float) -> float:
    """
    Virtual temperature of a two-level system from excited-state population pe.

    We guard against pe ~ 0.5 (denominator -> 0) and clip the final temperature
    to a reasonable window to avoid insane plot scales.
    """
    pe = float(min(max(pe, 1e-8), 1 - 1e-8))
    try:
        ratio = pe / (1.0 - pe)
        ratio = min(max(ratio, 1e-8), 1e8)
        denom = -math.log(ratio)
        if abs(denom) < 1e-3:
            denom = 1e-3 if denom >= 0 else -1e-3
        Tv = DeltaE / (kB * denom)
        Tv = max(min(Tv, 80.0), -80.0)
    except Exception:
        Tv = float("nan")
    return float(Tv)


# ------------------------ PARAMETER DATACLASSES ------------------------------

@dataclass
class GKSLParams:
    Omega0: float = 4.0
    Omega_drive: float = 2.2
    phi_drive: float = 0.0
    phi_rate: float = 0.35

    DeltaE: float = 8e-23
    gam_relax: float = 0.055
    gamma_phi: float = 0.06
    T_bath: float = 11.0

    dt: float = 0.02
    tmax: float = 12.0

    psi0: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=complex
        )
    )

    noise_enable: bool = True
    noise_strength: float = 0.16
    noise_tau: float = 0.9
    drift_strength: float = 0.08
    seed: int = 0

    reset_enable: bool = True
    reset_period: float = 4.0
    reset_width: float = 0.8
    reset_phase: float = 0.0


@dataclass
class BathSchedule:
    enable: bool = True
    period: float = 4.0
    duty: float = 0.45
    amp_T: float = 9.0
    amp_gamma: float = 0.35
    amp_drive: float = 0.5
    waveform: str = "square"
    phase_per_qubit: float = 0.0
    global_phase: float = 0.0

    def wave(self, t: float, q_index: int = 0) -> float:
        if not self.enable or self.period <= 1e-12:
            return 0.0
        phi = self.global_phase + self.phase_per_qubit * q_index
        t_shifted = (t + phi * (self.period / (2 * math.pi))) % self.period
        x = t_shifted / self.period
        if self.waveform == "square":
            return 1.0 if x < self.duty else 0.0
        # smooth sinusoidal modulation as a fallback
        return 0.5 * (1 + math.sin(2 * math.pi * x))


@dataclass
class MeanFieldSchedule:
    enable: bool = True
    lambda_z: float = 0.6
    lambda_x: float = 0.3
    iters: int = 1


# ------------------------ EVOLUTION: SINGLE QUBIT ---------------------------

def GKSL_evolve_single(
    p: GKSLParams,
    bath: Optional[BathSchedule] = None,
    q_index: int = 0,
    mf_z_shift: Optional[Callable[[float], float]] = None,
    mf_trans_shift: Optional[Callable[[float], float]] = None,
) -> Dict:
    nsteps = int(p.tmax / p.dt) + 1
    t = np.linspace(0.0, p.tmax, nsteps)

    rho = np.outer(p.psi0, p.psi0.conj())
    rho /= rho.trace()
    rho_ref = rho.copy()
    rho_echo = rho.copy()

    rng = np.random.default_rng(p.seed + q_index)

    x_O0 = 0.0
    x_phi = 0.0
    x_noise = 0.0
    phi_accum = 0.0

    rhos = np.zeros((nsteps, 2, 2), dtype=complex)
    bloch = np.zeros((nsteps, 3), dtype=float)
    pe = np.zeros(nsteps, dtype=float)
    coh = np.zeros(nsteps, dtype=float)
    phase = np.zeros(nsteps, dtype=float)
    qfi = np.zeros(nsteps, dtype=float)
    purity = np.zeros(nsteps, dtype=float)
    bures_L = np.zeros(nsteps, dtype=float)
    echo = np.zeros(nsteps, dtype=float)
    Sigma = np.zeros(nsteps, dtype=float)
    friction = np.zeros(nsteps, dtype=float)
    otoc_F = np.zeros(nsteps, dtype=float)
    otoc_scr = np.zeros(nsteps, dtype=float)
    choi_min = np.zeros(nsteps, dtype=float)
    vnS_data = np.zeros(nsteps, dtype=float)

    # Temperatures
    T_aux = np.zeros(nsteps, dtype=float)    # smoothed auxiliary reservoir temp
    T_virt = np.zeros(nsteps, dtype=float)   # virtual temperature from pe
    T1_inst = np.zeros(nsteps, dtype=float)
    T2_inst = np.zeros(nsteps, dtype=float)

    S_prev = vn_entropy_bits(rho)
    b_last = bloch_vector(rho)
    sigma_smooth = 0.0

    # Auxiliary reservoir relaxation towards T_virt
    T_aux_curr = p.T_bath

    for k, tt in enumerate(t):
        # Colored noise & phase drift
        if p.noise_enable:
            tau = p.noise_tau
            sig = p.noise_strength * math.sqrt(2.0 / max(tau, 1e-9))

            dW0 = math.sqrt(p.dt) * rng.normal()
            dW1 = math.sqrt(p.dt) * rng.normal()
            dW2 = math.sqrt(p.dt) * rng.normal()

            x_O0 += (-x_O0 / tau) * p.dt + sig * dW0
            x_phi += (-x_phi / tau) * p.dt + sig * dW1
            x_noise += (-x_noise / tau) * p.dt + sig * dW2

            phi_accum += p.drift_strength * math.sqrt(p.dt) * rng.normal()

        w_bath = bath.wave(tt, q_index) if bath is not None else 0.0

        z_shift = mf_z_shift(tt) if mf_z_shift is not None else 0.0
        trans_shift = mf_trans_shift(tt) if mf_trans_shift is not None else 0.0

        O0 = p.Omega0 * (1.0 + 0.18 * x_O0) + z_shift
        phase_drive = p.phi_drive + p.phi_rate * tt + phi_accum + 0.3 * x_phi
        Od = p.Omega_drive * (1.0 + (bath.amp_drive * w_bath if bath else 0.0)) + trans_shift

        omega_vec = np.array(
            [Od * math.cos(phase_drive), Od * math.sin(phase_drive), O0],
            dtype=float,
        )

        # Bath-dependent effective temperature and relaxation
        T_eff = max(0.1, p.T_bath + (bath.amp_T * w_bath if bath else 0.0))
        try:
            p_eq = 1.0 / (1.0 + math.exp(p.DeltaE / (kB * T_eff)))
        except OverflowError:
            p_eq = 0.0
        gam = p.gam_relax * (1.0 + (bath.amp_gamma * w_bath if bath else 0.0))

        # Instantaneous T1/T2 (rough estimates)
        T1_now = 1.0 / max(gam, 1e-9)
        T2_now = 1.0 / max(gam / 2.0 + p.gamma_phi, 1e-9)
        T1_inst[k] = T1_now
        T2_inst[k] = T2_now

        U = coherent_step(omega_vec, p.dt)
        maps: List[List[np.ndarray]] = []

        if p.gamma_phi > 0:
            p_deph = 0.5 * (1.0 - math.exp(-p.gamma_phi * p.dt))
            maps.append(kraus_dephase(p_deph))

        if gam > 0:
            g_down = 1.0 - math.exp(-gam * (1.0 - p_eq) * p.dt)
            g_up = 1.0 - math.exp(-gam * p_eq * p.dt)
            maps.append(kraus_amp_down(g_down))
            maps.append(kraus_amp_up(g_up))

        rho = U @ rho @ U.conj().T
        for Ks in maps:
            rho = apply_kraus(rho, Ks)
        rho /= rho.trace().real

        # Cyclic reset
        if p.reset_enable and p.reset_period > 1e-9 and p.reset_width > 0.0:
            win = (tt + p.reset_phase) % p.reset_period
            if win < p.reset_width:
                alpha = ((p.reset_width - win) / p.reset_width) * 0.6
                rho = (1.0 - alpha) * rho + alpha * rho_ref
                rho /= rho.trace().real

        # Measurements
        rhos[k] = rho
        b = bloch_vector(rho)
        bloch[k] = b
        pe[k] = rho[1, 1].real
        coh[k] = abs(rho[0, 1])
        phase[k] = float(np.angle(rho[0, 1]))
        qfi[k] = qfi_state_sensitivity(rho, 0.5 * sigma_z)
        purity[k] = float(np.trace(rho @ rho).real)
        bures_L[k] = bures_angle(rho, rho_ref)
        echo[k] = fidelity_bloch(rho, rho_echo)

        # Virtual temperature directly from pe
        T_virt[k] = virtual_temperature_from_pe(pe[k], p.DeltaE)

        vnS = vn_entropy_bits(rho)
        vnS_data[k] = vnS
        sigma_raw = (vnS - S_prev) / max(p.dt, 1e-9)
        sigma_smooth = 0.9 * sigma_smooth + 0.1 * sigma_raw
        Sigma[k] = sigma_smooth
        S_prev = vnS

        if k > 0:
            fH = np.cross(omega_vec, b_last)
            fD = (b - b_last) / p.dt - fH
            friction[k] = float(np.linalg.norm(np.cross(fH, fD)))
        b_last = b

        comm = sigma_z @ (sigma_x @ rho @ sigma_x) - (sigma_x @ rho @ sigma_x) @ sigma_z
        otoc_val = float(np.trace(comm.conj().T @ comm).real)
        otoc_F[k] = math.exp(-0.5 * max(0.0, otoc_val))
        otoc_scr[k] = 1.0 - otoc_F[k]

        if k % 10 == 0:
            eff = compose_kraus(U, maps)
            J = choi_from_kraus(eff)
            choi_min[k] = float(np.linalg.eigvalsh((J + J.conj().T) / 2.0).min())
        elif k > 0:
            choi_min[k] = choi_min[k - 1]

        # Simple relaxational auxiliary reservoir that follows T_virt
        if k == 0:
            T_aux_curr = T_eff
        else:
            tau_aux = 2.0  # relaxation timescale
            alpha_aux = min(p.dt / max(tau_aux, 1e-6), 1.0)
            if math.isfinite(T_virt[k]):
                T_aux_curr += alpha_aux * (T_virt[k] - T_aux_curr)
        T_aux[k] = T_aux_curr

    return dict(
        t=t,
        rho=rhos,
        bloch=bloch,
        pe=pe,
        coh=coh,
        phase=phase,
        qfi=qfi,
        purity=purity,
        bures_L=bures_L,
        echo=echo,
        Sigma=Sigma,
        friction=friction,
        otoc_F=otoc_F,
        otoc_scr=otoc_scr,
        choi_min=choi_min,
        vnS=vnS_data,
        T_aux=T_aux,
        T_virt=T_virt,
        T1=T1_inst,
        T2=T2_inst,
    )


def evolve_single_gksl(
    p: GKSLParams,
    bath: Optional[BathSchedule] = None,
    store_trajectory: bool = True,
) -> Dict:
    res = GKSL_evolve_single(p, bath=bath, q_index=0)
    out = dict(res) if store_trajectory else {"t": res["t"], "pe": res["pe"], "Sigma": res["Sigma"]}
    out["Sigma_total"] = float(np.trapezoid(res["Sigma"], res["t"]))
    return out


# ------------------------ LATTICE EVOLUTION ----------------------------------

def _interp_series(series: np.ndarray, dt: float) -> Callable[[float], float]:
    N = len(series)

    def f(t: float) -> float:
        idx = min(max(int(round(t / dt)), 0), N - 1)
        return float(series[idx])

    return f


def GKSL_evolve_lattice(
    base_p: GKSLParams,
    bath: BathSchedule,
    mf: MeanFieldSchedule,
    n_qubits: int = 4,
) -> List[Dict]:
    """
    Evolve a 4-qubit lattice with mild qubit-to-qubit parameter variations so
    that T1/T2, thermodynamic lengths and temperature rhythms are not cloned.
    """
    psi_lock = np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=complex)

    params: List[GKSLParams] = []

    # Stronger diversity across qubits
    relax_scales = np.linspace(0.55, 1.65, n_qubits)
    dephase_scales = np.linspace(0.7, 1.8, n_qubits)
    drive_scales = np.linspace(0.8, 1.5, n_qubits)
    omega0_scales = np.linspace(0.8, 1.3, n_qubits)
    phi_rate_offsets = np.linspace(-0.06, 0.12, n_qubits)

    for i in range(n_qubits):
        p = GKSLParams(**base_p.__dict__)
        p.psi0 = psi_lock.copy()
        p.seed = base_p.seed + 100 * i

        p.gam_relax = base_p.gam_relax * float(relax_scales[i])
        p.gamma_phi = base_p.gamma_phi * float(dephase_scales[i])
        p.Omega_drive = base_p.Omega_drive * float(drive_scales[i])
        p.Omega0 = base_p.Omega0 * float(omega0_scales[i])
        p.phi_rate = base_p.phi_rate + float(phi_rate_offsets[i])
        p.reset_phase = base_p.reset_phase + 0.7 * i

        params.append(p)

    # Simple ring-like adjacency
    adj = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=float,
    )

    nsteps = int(base_p.tmax / base_p.dt) + 1
    z_trajs = np.zeros((n_qubits, nsteps), dtype=float)
    x_trajs = np.zeros((n_qubits, nsteps), dtype=float)

    print(f"Simulating Lattice ({mf.iters} MF iterations)...")
    outs: List[Dict] = []

    for it in range(mf.iters + 1):
        outs = []
        new_z = np.zeros_like(z_trajs)
        new_x = np.zeros_like(x_trajs)

        for i in range(n_qubits):
            neighbors = np.where(adj[i] > 0)[0]

            if mf.enable and neighbors.size > 0 and it > 0:
                z_field = mf.lambda_z * np.sum(z_trajs[neighbors], axis=0)
                x_field = mf.lambda_x * np.sum(x_trajs[neighbors], axis=0)
                mf_z = _interp_series(z_field, base_p.dt)
                mf_x = _interp_series(x_field, base_p.dt)
            else:
                mf_z = None
                mf_x = None

            res = GKSL_evolve_single(
                params[i],
                bath=bath,
                q_index=i,
                mf_z_shift=mf_z,
                mf_trans_shift=mf_x,
            )
            outs.append(res)
            new_z[i] = res["bloch"][:, 2]
            new_x[i] = res["bloch"][:, 0]

        z_trajs = new_z
        x_trajs = new_x

    return outs


# ------------------------ VISUAL HELPERS -------------------------------------

def _make_bloch_sphere(ax, color: str):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = BLOCH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = BLOCH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = BLOCH_RADIUS * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="#B0B0B0", alpha=0.35, linewidth=0.9)
    ax.plot(
        BLOCH_RADIUS * np.cos(u),
        BLOCH_RADIUS * np.sin(u),
        0,
        color="black",
        alpha=0.25,
        lw=1.0,
    )
    ax.plot(
        np.zeros_like(u),
        BLOCH_RADIUS * np.cos(u),
        BLOCH_RADIUS * np.sin(u),
        color="black",
        alpha=0.25,
        lw=1.0,
    )

    ax.set_xlim3d(-BLOCH_RADIUS, BLOCH_RADIUS)
    ax.set_ylim3d(-BLOCH_RADIUS, BLOCH_RADIUS)
    ax.set_zlim3d(-BLOCH_RADIUS, BLOCH_RADIUS)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.dist = float(BLOCH_ZOOM)
    ax.set_axis_off()

    vec, = ax.plot(
        [0, 0],
        [0, 0],
        [0, BLOCH_RADIUS],
        lw=BLOCH_VEC_LW,
        color=color,
        solid_capstyle="round",
        zorder=10,
    )
    return vec


class BlochTrail3D:
    """
    Simple 3D trail inside the Bloch sphere. The trail is drawn with a fixed
    per-qubit color and lives strictly inside the sphere radius because the
    Bloch vector length ||r|| <= 1 (we also clamp to be safe).
    """

    def __init__(self, ax, color: str):
        self.ax = ax
        self.points: List[np.ndarray] = []
        self.collection = Line3DCollection(
            [],
            colors=[color],
            linewidths=BLOCH_TRAIL_LW,
            alpha=BLOCH_TRAIL_ALPHA,
            zorder=5,
        )
        self._added = False

    def clear(self):
        self.points = []
        self.collection.set_segments([])

    def update(self, pt: np.ndarray):
        # Ensure points stay inside the Bloch sphere radius
        pt = np.asarray(pt, dtype=float)
        r = float(np.linalg.norm(pt))
        if r > BLOCH_RADIUS + 1e-9:
            pt = pt * (BLOCH_RADIUS / r)

        self.points.append(pt)
        if len(self.points) < 2:
            return
        pts = np.vstack(self.points)
        segs = np.concatenate(
            [pts[:-1, np.newaxis, :], pts[1:, np.newaxis, :]],
            axis=1,
        )
        self.collection.set_segments(segs)
        if not self._added:
            self.ax.add_collection3d(self.collection)
            self._added = True


class Wigner3DHelper:
    def __init__(
        self,
        ax,
        title: Optional[str] = None,
        zoom_dist: float = 1.4,
        N: int = 72,
        N_ho: int = 14,
    ):
        """
        zoom_dist kept fairly small so surfaces are visually large.
        3D boxes are borderless and non-spinning for a calmer view.
        """
        self.ax = ax
        self.N = N
        self.N_ho = N_ho
        self.x = np.linspace(-3.5, 3.5, N)
        self.p = np.linspace(-3.5, 3.5, N)
        self.X, self.Y = np.meshgrid(self.x, self.p)
        self.surf = None
        self.azim = -45.0

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        try:
            self.ax.set_box_aspect((1, 1, 0.5))
        except Exception:
            pass
        self.ax.dist = zoom_dist
        self.ax.view_init(elev=40, azim=self.azim)
        self.ax.set_axis_off()

        if title:
            self.ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

    def draw(self, rho2: np.ndarray):
        if not HAVE_QUTIP or Qobj is None or qutip_wigner is None:
            return
        Z_emb = np.zeros((self.N_ho, self.N_ho), dtype=complex)
        Z_emb[:2, :2] = rho2
        W = qutip_wigner(Qobj(Z_emb), self.x, self.p)
        Z = np.array(W, dtype=float)

        if self.surf is not None:
            self.surf.remove()
        self.surf = self.ax.plot_surface(
            self.X,
            self.Y,
            Z,
            cmap="viridis",
            linewidth=0.0,
            edgecolors="none",
            antialiased=True,
            alpha=1.0,
            shade=True,
        )
        self.ax.set_zlim(-0.4, 0.4)
        # No spinning: keep fixed view angle.


# ------------------------ Y-LIMIT HELPER FOR DIAGNOSTICS ---------------------

def _static_ylim(arrays, positive_floor: bool = False, pad: float = 0.1):
    """
    Compute a fixed y-range that covers all provided arrays with a bit of padding.
    If positive_floor is True, clamp the lower bound to >= 0.
    """
    all_data = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            all_data.append(a)
    if not all_data:
        return (0.0, 1.0)
    data = np.concatenate(all_data)
    ymin = float(data.min())
    ymax = float(data.max())
    if positive_floor and ymin < 0:
        ymin = 0.0
    if ymin == ymax:
        if positive_floor:
            ymin = 0.0
            ymax = ymax * 1.1 + 0.1
        else:
            ymin -= 0.5
            ymax += 0.5
    dy = ymax - ymin
    ymin -= pad * dy
    ymax += pad * dy
    return (ymin, ymax)


# ------------------------ ANIMATION ------------------------------------------

def animate_lattice_dashboard(
    p: GKSLParams,
    out_gif: str,
    bath: BathSchedule,
    mf: MeanFieldSchedule,
    fps: int = 24,
    dpi: int = 100,
) -> str:
    results = GKSL_evolve_lattice(p, bath, mf, n_qubits=4)

    # Thin out frames a bit for speed
    skip = 2
    times = results[0]["t"][::skip]
    N = len(times)

    rhos_all = np.array([r["rho"][::skip] for r in results])
    blochs = np.array([r["bloch"][::skip] for r in results])
    phases = np.array([r["phase"][::skip] for r in results])
    cohs = np.array([r["coh"][::skip] for r in results])
    pes = np.array([r["pe"][::skip] for r in results])
    pur = np.array([r["purity"][::skip] for r in results])
    vnS_all = np.array([r["vnS"][::skip] for r in results])
    bures_all = np.array([r["bures_L"][::skip] for r in results])  # TL per qubit
    Taux_raw_all = np.array([r["T_aux"][::skip] for r in results])  # (unused now)
    Tvirt_all = np.array([r["T_virt"][::skip] for r in results])
    T1_all = np.array([r["T1"][::skip] for r in results])
    T2_all = np.array([r["T2"][::skip] for r in results])

    # Lattice-mean diagnostics
    mean_qfi = np.mean([r["qfi"][::skip] for r in results], axis=0)
    mean_echo = np.mean([r["echo"][::skip] for r in results], axis=0)
    mean_sigma = np.mean([r["Sigma"][::skip] for r in results], axis=0)
    mean_fric_raw = np.mean([r["friction"][::skip] for r in results], axis=0)
    mean_otoc = np.mean([r["otoc_F"][::skip] for r in results], axis=0)
    mean_scr = np.mean([r["otoc_scr"][::skip] for r in results], axis=0)
    mean_choi = np.mean([r["choi_min"][::skip] for r in results], axis=0)
    mean_vnS = np.mean(vnS_all, axis=0)
    mean_bures = np.mean(bures_all, axis=0)  # thermodynamic Bures length
    mean_z = np.mean(blochs[:, :, 2], axis=0)
    mean_pur = np.mean(pur, axis=0)

    # Soft clipping for friction to avoid one huge spike ruining the scale
    fric_cap = float(np.percentile(np.abs(mean_fric_raw), 98)) if np.any(mean_fric_raw) else 1.0
    if fric_cap < 1e-8:
        fric_cap = 1.0
    mean_fric = np.clip(mean_fric_raw, -fric_cap, fric_cap)

    # Normalize Choi λ_min for visibility
    den = float(np.max(mean_choi) - np.min(mean_choi))
    if den < 1e-15:
        mean_choi_norm = np.zeros_like(mean_choi)
    else:
        mean_choi_norm = (mean_choi - np.min(mean_choi)) / den

    # Bath square waves per qubit and their mean
    bath_period = bath.period if bath.period > 1e-9 else p.reset_period
    n_cycles = max(1, int(math.ceil(times[-1] / bath_period)))

    w_all = np.zeros((4, N), dtype=float)
    for qi in range(4):
        for k, t_now in enumerate(times):
            w_all[qi, k] = bath.wave(t_now, qi)
    w_mean_all = np.mean(w_all, axis=0)

    # Bath temperature T_bath(t,q) from square wave + baseline
    Tbath_all = p.T_bath + bath.amp_T * w_all

    # Auxiliary reservoir: smooth response towards T_virt (we overwrite raw T_aux)
    Taux_all = np.zeros_like(Tvirt_all)
    tau_aux = 2.0
    alpha_aux = min((p.dt * skip) / max(tau_aux, 1e-6), 1.0)
    for qi in range(4):
        Taux_all[qi, 0] = Tbath_all[qi, 0]
        for k in range(1, N):
            prev = Taux_all[qi, k - 1]
            target = Tvirt_all[qi, k]
            if not math.isfinite(float(target)):
                Taux_all[qi, k] = prev
            else:
                Taux_all[qi, k] = prev + alpha_aux * (target - prev)

    # Cycle-resolved "Otto-like" efficiency using per-qubit virtual temperatures
    def _eta_from_series(Tv_series: np.ndarray, w_series: np.ndarray) -> np.ndarray:
        etas = np.zeros(n_cycles, dtype=float)
        for j in range(n_cycles):
            t_start = j * bath_period
            t_end = (j + 1) * bath_period
            mask = (times >= t_start) & (times < t_end)
            if not np.any(mask):
                etas[j] = 0.0
                continue
            mask_hot = mask & (w_series >= 0.5)
            mask_cold = mask & (w_series < 0.5)
            Tv_hot = Tv_series[mask_hot]
            Tv_cold = Tv_series[mask_cold]
            Tv_hot = Tv_hot[np.isfinite(Tv_hot)]
            Tv_cold = Tv_cold[np.isfinite(Tv_cold)]
            if Tv_hot.size == 0 or Tv_cold.size == 0:
                etas[j] = 0.0
            else:
                Th = float(np.mean(np.abs(Tv_hot)))
                Tc = float(np.mean(np.abs(Tv_cold)))
                if Th <= 1e-6:
                    etas[j] = 0.0
                else:
                    etas[j] = float(np.clip(1.0 - Tc / Th, 0.0, 1.0))
        return etas

    eta_cycle_qubits = np.zeros((4, n_cycles), dtype=float)
    for qi in range(4):
        eta_cycle_qubits[qi] = _eta_from_series(Tvirt_all[qi], w_all[qi])

    Tvirt_mean_all = np.mean(Tvirt_all, axis=0)
    eta_cycle_mean = _eta_from_series(Tvirt_mean_all, w_mean_all)

    # Integrated entropy production ∬ Σ dt
    Sigma_cum = np.cumsum(mean_sigma) * p.dt * skip

    # Precomputed y-limits for diagnostics (avoid per-frame autoscale / flicker)
    qfi_ylim = _static_ylim([mean_qfi, mean_pur, mean_bures], positive_floor=True)
    echo_ylim = _static_ylim([mean_echo * 100.0, mean_sigma * 5.0], positive_floor=True)
    fric_ylim = _static_ylim([mean_fric, Sigma_cum], positive_floor=True)
    otoc_ylim = _static_ylim([mean_otoc], positive_floor=True)
    scr_ylim = _static_ylim([mean_scr], positive_floor=True)
    choi_ylim = _static_ylim([mean_choi_norm, mean_vnS], positive_floor=True)
    floq_ylim = _static_ylim([mean_z], positive_floor=False)

    # Figure & layout: 5 rows (Bloch, per-qubit details, collective Wigner,
    # diagnostics 2×4, thermodynamic lengths TL1..TL4 centered).
    fig = plt.figure(figsize=(29.9, 40.96), dpi=dpi)
    gs_main = fig.add_gridspec(
        5,
        1,
        # First row (Bloch) 1.6× taller than before; Wigner rows also slightly enlarged.
        height_ratios=[3.2, 5.4, 4.2, 2.8, 1.8],
        hspace=0.22,  # tighter vertical spacing to reduce blank space (esp. under Bloch row)
    )
    # Pull subplots slightly closer to the top & bottom to eat excess white space
    fig.subplots_adjust(top=0.96, bottom=0.04)

    # -------- Row 1: Bloch spheres + aligned info boxes (two per qubit)
    gs_bloch = gs_main[0].subgridspec(1, 4, wspace=0.18)

    bloch_axes = [fig.add_subplot(gs_bloch[0, i], projection="3d") for i in range(4)]
    bloch_vecs = [_make_bloch_sphere(ax, QUBIT_COLORS[i]) for i, ax in enumerate(bloch_axes)]
    bloch_trails = [BlochTrail3D(ax, QUBIT_COLORS[i]) for i, ax in enumerate(bloch_axes)]

    # T1/T2/η box + Bath/Aux/Virt/Purity/TL box per qubit, both aligned above the sphere
    t1t2_boxes = []
    bloch_info_boxes = []

    for i, ax in enumerate(bloch_axes):
        # Qubit label higher above the info boxes to avoid overlap
        ax.text2D(
            0.5,
            1.28,  # was 1.20
            f"Qubit {i+1}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

        # Left box: T1/T2/η info
        tbox = ax.text2D(
            0.02,
            1.03,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.96),
        )
        t1t2_boxes.append(tbox)

        # Right box: BathT / AuxT / VirtT / Purity / TL
        info = ax.text2D(
            0.52,
            1.03,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.96),
        )
        bloch_info_boxes.append(info)

    # -------- Row 2: per-qubit phase/density/Wigner
    gs_details = gs_main[1].subgridspec(1, 4, wspace=0.25)

    rose_axes = []
    rose_bars_list = []
    rose_dots = []
    phase_bin_edges_list = []
    dens_axes = []
    dens_bars_list = []
    dens_labels = []
    wig_helpers: List[Wigner3DHelper] = []

    for i in range(4):
        # Slightly tighter vertical spacing within each column (phase/density vs Wigner)
        gs_q = gs_details[i].subgridspec(2, 1, height_ratios=[1.0, 2.6], hspace=0.15)
        gs_q_top = gs_q[0].subgridspec(1, 2, wspace=0.35)

        ax_p = fig.add_subplot(gs_q_top[0], projection="polar")
        ax_p.set_title("Phase (dot + decay)", fontsize=13, fontweight="bold", pad=10)
        ax_p.set_yticklabels([])
        ax_p.set_xticklabels([])
        ax_p.grid(True, alpha=0.3)
        ax_p.set_ylim(0.0, 1.1)
        ax_p.set_theta_direction(-1)
        ax_p.set_theta_zero_location("N")

        edges = np.linspace(-np.pi, np.pi, 32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        bars = ax_p.bar(
            centers,
            np.zeros_like(centers),
            width=width,
            color=COLOR_PHASE_BAR,
            alpha=0.7,
            bottom=0.0,
        )
        dot, = ax_p.plot([], [], "o", color=COLOR_PHASE_DOT, ms=10, zorder=10)

        rose_axes.append(ax_p)
        rose_bars_list.append(bars)
        rose_dots.append(dot)
        phase_bin_edges_list.append(edges)

        ax_d = fig.add_subplot(gs_q_top[1])
        ax_d.set_title("Density Entries", fontsize=13, fontweight="bold")
        ax_d.set_ylim(0.0, DENSITY_HEIGHT_SCALE)
        ax_d.set_xticks([0, 1, 2])
        ax_d.set_xticklabels(["|00|", "|01|", "|11|"], fontsize=11)
        ax_d.grid(axis="y", alpha=0.3)

        dbars = ax_d.bar(
            [0, 1, 2],
            [0, 0, 0],
            color=COLOR_DENS_BAR,
            edgecolor=COLOR_DENS_EDGE,
            linewidth=2.0,
        )
        dtxt = ax_d.text(
            0.5, 0.85, "", transform=ax_d.transAxes, ha="center", fontsize=11
        )

        dens_axes.append(ax_d)
        dens_bars_list.append(dbars)
        dens_labels.append(dtxt)

        ax_w = fig.add_subplot(gs_q[1], projection="3d")
        # Larger, higher-resolution local Wigner surfaces
        wh = Wigner3DHelper(
            ax_w,
            title=f"Wigner Q{i+1}",
            zoom_dist=1.2,
            N=80,
            N_ho=16,
        )
        wig_helpers.append(wh)

    # -------- Row 3: collective Wigner
    gs_coll = gs_main[2].subgridspec(1, 3, width_ratios=[0.2, 2.8, 0.2])
    ax_wig_coll = fig.add_subplot(gs_coll[1], projection="3d")
    wig_coll = Wigner3DHelper(
        ax_wig_coll,
        title="Collective Lattice Wigner (mean state)",
        zoom_dist=1.0,
        N=96,
        N_ho=20,
    )

    # -------- Row 4: diagnostics 2×4
    gs_diag = gs_main[3].subgridspec(2, 4, wspace=0.3, hspace=0.6)
    diag_axes: List[plt.Axes] = []

    titles = [
        "QFI + Purity + Bures thermodynamic length",
        "Echo×100 + Σ (scaled)",
        "Friction + ∫Σ dt",
        "OTOC F(t)",
        "Scramble 1-F(t)",
        "Choi min λ(J) (norm) + S_vN",
        "Floquet ⟨σ_z⟩",
        "Aux / Virtual / Bath Temperatures + bath drive",
    ]

    def setup_diag(idx: int):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(gs_diag[r, c])
        ax.set_title(titles[idx], fontsize=12, fontweight="bold")
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, alpha=0.2)
        l, = ax.plot([], [], lw=3.0)
        diag_axes.append(ax)
        return ax, l

    # 0: QFI, purity, lattice-mean Bures thermodynamic length
    ax0, l_qfi = setup_diag(0)
    l_qfi.set_color("#1f77b4")
    l_qfi.set_label("QFI")
    l_pur, = ax0.plot([], [], color="#777777", lw=2.5, label="Purity")
    l_bures, = ax0.plot([], [], color="#FF8C00", lw=3.0, ls="-", label="Bures TL (mean)")
    ax0.legend(fontsize=9, loc="upper right")
    ax0.set_ylim(*qfi_ylim)

    # 1: Echo and Σ
    ax1, l_echo = setup_diag(1)
    l_echo.set_color("#1f77b4")
    l_echo.set_label("Echo")
    l_sig1, = ax1.plot([], [], color="#d62728", lw=3.0, label="Σ×5")
    ax1.legend(fontsize=9)
    ax1.set_ylim(*echo_ylim)

    # 2: Friction and integrated Σ
    ax2, l_fric = setup_diag(2)
    l_fric.set_color("#e377c2")
    l_fric.set_label("Friction")
    l_sig2, = ax2.plot([], [], color="#d62728", lw=3.0, ls="--", label="∫Σ dt")
    ax2.legend(fontsize=9)
    ax2.set_ylim(*fric_ylim)

    # 3: OTOC F(t)
    ax3, l_otoc = setup_diag(3)
    l_otoc.set_color("#1f77b4")
    ax3.set_ylim(*otoc_ylim)

    # 4: Scrambling 1-F(t)
    ax4, l_scr = setup_diag(4)
    l_scr.set_color("#d62728")
    ax4.set_ylim(*scr_ylim)

    # 5: Choi λ_min and S_vN
    ax5, l_choi = setup_diag(5)
    l_choi.set_color("#001f3f")
    l_choi.set_label("min λ(J) [norm]")
    l_svn, = ax5.plot([], [], color="#ff7f0e", lw=3.0, label="S_vN")
    ax5.legend(fontsize=9)
    ax5.set_ylim(*choi_ylim)

    # 6: Floquet <σ_z>
    ax6, l_floq = setup_diag(6)
    l_floq.set_color("black")
    ax6.set_ylim(*floq_ylim)

    # 7: Aux / Virtual / Bath temperatures + bath drive
    ax7, l_aux = setup_diag(7)
    l_aux.set_color("#2ca02c")
    l_aux.set_label("Aux T")
    l_virt, = ax7.plot([], [], color="#ff7f0e", lw=2.5, ls="-", label="Virt T")
    l_bathT, = ax7.plot([], [], color="#1f77b4", lw=2.2, ls="-.", label="Bath T")
    ax7.set_ylim(-40.0, 40.0)
    ax7.set_ylabel("Temperature (K)")

    # Secondary axis for bath square wave 0/1
    ax7_bath = ax7.twinx()
    (l_bath_drive,) = ax7_bath.plot(
        [], [],
        lw=1.8,
        color="0.35",
        alpha=0.9,
        label="Bath drive (0/1)",
    )
    ax7_bath.set_ylim(-0.1, 1.1)
    ax7_bath.set_yticks([0.0, 1.0])
    ax7_bath.set_yticklabels(["cold", "hot"])
    ax7_bath.tick_params(axis="y", labelsize=8)

    # Combined legend (primary + secondary), moved above panel to avoid obstruction
    handles1, labels1 = ax7.get_legend_handles_labels()
    handles2, labels2 = ax7_bath.get_legend_handles_labels()
    ax7.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.28),
        borderaxespad=0.0,
        ncol=4,
    )

    # -------- Row 5: Thermodynamic lengths TL1..TL4 (per qubit), centered
    gs_tl = gs_main[4].subgridspec(1, 3, width_ratios=[0.25, 0.5, 0.25])
    ax_tl = fig.add_subplot(gs_tl[0, 1])
    ax_tl.set_title("Thermodynamic lengths TL1–TL4 (per qubit)", fontsize=12, fontweight="bold")
    ax_tl.set_xlim(times[0], times[-1])
    ax_tl.set_ylim(0.0, max(1.2 * np.max(bures_all), 1.0))
    ax_tl.grid(True, alpha=0.2)
    tl_lines = []
    for qi in range(4):
        line, = ax_tl.plot(
            [],
            [],
            lw=2.5,
            color=TL_COLORS[qi],
            label=f"TL Q{qi+1}",
        )
        tl_lines.append(line)
    ax_tl.legend(fontsize=9, loc="upper right")

    prev_vecs = [blochs[i, 0].copy() for i in range(4)]

    def fmt_T(x: float) -> str:
        if not np.isfinite(x):
            return "∞"
        if x > 999.9:
            return ">999"
        return f"{x:6.2f}"

    def update(k: int):
        t_now = times[k]
        current_cycle = int(math.floor(t_now / bath_period))
        current_cycle = max(0, min(current_cycle, n_cycles - 1))

        # Bloch spheres, trails, T1/T2/η/Bath boxes
        for i in range(4):
            curr = blochs[i, k]

            # Smooth interpolation, then clamp Bloch vector length to <= 1
            disp = vector_slerp(prev_vecs[i], curr, 0.35)
            norm_disp = float(np.linalg.norm(disp))
            if norm_disp > 1.0:
                disp = disp / norm_disp
            prev_vecs[i] = disp

            vec_scaled = BLOCH_RADIUS * disp
            bloch_vecs[i].set_data_3d(
                [0.0, vec_scaled[0]],
                [0.0, vec_scaled[1]],
                [0.0, vec_scaled[2]],
            )
            bloch_trails[i].update(vec_scaled)

            T1_now = T1_all[i, k]
            T2_now = T2_all[i, k]
            eta_now = eta_cycle_qubits[i, current_cycle]
            Tbath_now = Tbath_all[i, k]

            # Lattice-average efficiency for Qubit 1 box only
            if i == 0:
                eta_lat = eta_cycle_mean[current_cycle]
                t1t2_boxes[i].set_text(
                    f"T1(t):   {fmt_T(T1_now)}\n"
                    f"T2(t):   {fmt_T(T2_now)}\n"
                    f"Bath T:  {Tbath_now:5.1f} K\n"
                    f"η_Otto(q):   {eta_now:4.2f}\n"
                    f"η_Otto(lat): {eta_lat:4.2f}"
                )
            else:
                t1t2_boxes[i].set_text(
                    f"T1(t):   {fmt_T(T1_now)}\n"
                    f"T2(t):   {fmt_T(T2_now)}\n"
                    f"Bath T:  {Tbath_now:5.1f} K\n"
                    f"η_Otto(q):   {eta_now:4.2f}"
                )

            Tv = Tvirt_all[i, k]
            Taux_now = Taux_all[i, k]
            TL_now = bures_all[i, k]  # per-qubit thermodynamic length

            bloch_info_boxes[i].set_text(
                f"BathT: {Tbath_now:5.1f} K\n"
                f"AuxT:  {Taux_now:5.1f} K\n"
                f"VirtT: {Tv:5.1f} K\n"
                f"Purity: {pur[i, k]:.3f}\n"
                f"TL Q{i+1}: {TL_now:.2f}"
            )

        # Phase, density, Wigners
        current_rhos: List[np.ndarray] = []
        for i in range(4):
            rho_i = rhos_all[i, k]
            current_rhos.append(rho_i)

            ph = phases[i, k]
            coh_val = cohs[i, k]

            rose_dots[i].set_data([ph], [0.5 * coh_val])

            edges = phase_bin_edges_list[i]
            hvals, _ = np.histogram(phases[i, : k + 1], bins=edges)
            hvals = hvals.astype(float)
            if hvals.max() > 0:
                hvals /= hvals.max()
            for bar, h in zip(rose_bars_list[i], hvals):
                new_h = 0.2 * h + 0.8 * bar.get_height()
                bar.set_height(new_h)

            dens_bars_list[i][0].set_height(max(0.0, 1.0 - pes[i, k]))
            dens_bars_list[i][1].set_height(coh_val)
            dens_bars_list[i][2].set_height(pes[i, k])
            dens_labels[i].set_text(f"arg ρ01 = {ph:+.2f}")

            wig_helpers[i].draw(rho_i)

        rho_avg = np.mean(np.array(current_rhos), axis=0)
        wig_coll.draw(rho_avg)

        # Diagnostics
        ts = times[: k + 1]

        l_qfi.set_data(ts, mean_qfi[: k + 1])
        l_pur.set_data(ts, mean_pur[: k + 1])
        l_bures.set_data(ts, mean_bures[: k + 1])

        l_echo.set_data(ts, mean_echo[: k + 1] * 100.0)
        l_sig1.set_data(ts, mean_sigma[: k + 1] * 5.0)

        l_fric.set_data(ts, mean_fric[: k + 1])
        l_sig2.set_data(ts, Sigma_cum[: k + 1])

        l_otoc.set_data(ts, mean_otoc[: k + 1])
        l_scr.set_data(ts, mean_scr[: k + 1])

        l_choi.set_data(ts, mean_choi_norm[: k + 1])
        l_svn.set_data(ts, mean_vnS[: k + 1])

        l_floq.set_data(ts, mean_z[: k + 1])

        # Aux vs virtual vs bath temperature: lattice means (clipped to ±40 K)
        mean_Taux = np.mean(Taux_all[:, : k + 1], axis=0)
        mean_Tvirt = np.mean(Tvirt_all[:, : k + 1], axis=0)
        mean_Tbath = np.mean(Tbath_all[:, : k + 1], axis=0)
        Taux_plot = np.clip(mean_Taux, -40.0, 40.0)
        Tvirt_plot = np.clip(mean_Tvirt, -40.0, 40.0)
        Tbath_plot = np.clip(mean_Tbath, -40.0, 40.0)

        l_aux.set_data(ts, Taux_plot)
        l_virt.set_data(ts, Tvirt_plot)
        l_bathT.set_data(ts, Tbath_plot)

        # Dynamic bath drive trace
        l_bath_drive.set_data(ts, w_mean_all[: k + 1])

        # Thermodynamic lengths TL1..TL4
        for qi in range(4):
            tl_lines[qi].set_data(ts, bures_all[qi, : k + 1])

    print(f"Generating Animation ({N} frames)...")
    if HAVE_IMAGEIO and imageio is not None:
        # ImageIO path with explicit per-frame progress
        with imageio.get_writer(out_gif, mode="I", fps=fps, loop=0) as writer:
            for k in range(N):
                update(k)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)[..., :3]
                writer.append_data(img)
                if k % 10 == 0 or k == N - 1:
                    print(f"Rendered {k}/{N}")
    else:
        # Pillow writer path, now with a progress callback so you see frames advancing
        anim = FuncAnimation(fig, update, frames=N)
        try:
            def _cb(i, n):
                if i % 10 == 0 or i == n - 1:
                    print(f"Rendered {i}/{n}")
            anim.save(out_gif, writer="pillow", fps=fps, progress_callback=_cb)
        except TypeError:
            # Older Matplotlib without progress_callback support
            anim.save(out_gif, writer="pillow", fps=fps)

    plt.close(fig)
    return out_gif


# ------------------------ LOGIC SELF-TEST ------------------------------------

def thermodynamic_neuron_logic_tests() -> Dict[str, Dict]:
    """
    Return weights and truth tables for NOT, NOR and 3-MAJORITY perceptron-like
    thermodynamic neurons. We treat these as simple perceptrons with weights w0..wn.
    """
    gates: Dict[str, Dict] = {}

    # NOT gate on one bit: y = 1 - x
    w_not = [0.5, -1.0]
    truth_not = [((0,), 1), ((1,), 0)]
    gates["NOT"] = {"weights": w_not, "truth": truth_not}

    # NOR gate on two bits: output 1 only if both inputs 0
    w_nor = [1.0, -2.0, -2.0]
    truth_nor = [((0, 0), 1), ((0, 1), 0), ((1, 0), 0), ((1, 1), 0)]
    gates["NOR"] = {"weights": w_nor, "truth": truth_nor}

    # 3-MAJORITY on three bits
    w_maj = [-4.0, 3.0, 3.0, 3.0]
    truth_maj = [
        ((0, 0, 0), 0),
        ((0, 0, 1), 0),
        ((0, 1, 0), 0),
        ((1, 0, 0), 0),
        ((0, 1, 1), 1),
        ((1, 0, 1), 1),
        ((1, 1, 0), 1),
        ((1, 1, 1), 1),
    ]
    gates["3-MAJORITY"] = {"weights": w_maj, "truth": truth_maj}

    def perceptron_eval(w: List[float], x_bits: tuple) -> int:
        y = w[0]
        for j, b in enumerate(x_bits):
            y += w[j + 1] * b
        return 1 if y >= 0 else 0

    print("Thermodynamic neuron logic self-test (perceptron equivalent):")
    for name, data in gates.items():
        w = data["weights"]
        truth = data["truth"]
        rows = []
        for inp, out in truth:
            pred = perceptron_eval(w, inp)
            rows.append((inp, out, pred))
        print(f"  {name} weights: {w}   truth/pred: {rows}")
    return gates


# ------------------------ SERVER + INDEX HTML --------------------------------

def write_index_html(out_dir: Path, gif_name: str, logic_data: Dict[str, Dict]) -> None:
    """
    Write an HTML page that shows the GIF with a zoom slider plus
    a thermodynamic-neuron logic test dashboard.
    """
    import json

    # Convert tuples to lists for JSON
    serializable = {}
    for name, data in logic_data.items():
        rows = []
        for inp, out in data["truth"]:
            rows.append({"input": list(inp), "out": int(out)})
        serializable[name] = {"weights": data["weights"], "rows": rows}

    logic_json = json.dumps(serializable)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Quantum Lattice Dashboard</title>
  <style>
    body {{
      background: #111;
      color: #eee;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      text-align: center;
      margin: 0;
      padding: 1.5rem;
    }}
    h1 {{
      font-weight: 700;
      margin-bottom: 0.25rem;
    }}
    h2 {{
      margin: 0.8rem 0 0.4rem 0;
    }}
    p.subtitle {{
      margin-top: 0;
      margin-bottom: 1.2rem;
      color: #ccc;
    }}
    .img-shell {{
      max-width: 100%;
      margin: 0 auto 0.75rem auto;
      border-radius: 10px;
      background: #000;
      box-shadow: 0 0 30px rgba(0,0,0,0.9);
      padding: 0.25rem;
    }}
    .img-wrap {{
      width: 100%;
      max-height: 78vh;
      overflow: auto;
    }}
    .img-inner {{
      display: inline-block;
      transform-origin: top left;
    }}
    .img-inner img {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
    .zoom-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.7rem;
      margin-top: 0.4rem;
      font-size: 0.9rem;
      color: #ddd;
    }}
    .zoom-row input[type="range"] {{
      width: 260px;
    }}
    #zoomValue {{
      min-width: 3.2rem;
      text-align: left;
      font-variant-numeric: tabular-nums;
    }}
    #logic-panel {{
      max-width: 900px;
      margin: 1.6rem auto 0 auto;
      text-align: left;
      background: #1a1a1a;
      border-radius: 10px;
      padding: 1.0rem 1.2rem 1.3rem 1.2rem;
      box-shadow: 0 0 25px rgba(0,0,0,0.7);
    }}
    #logic-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.8rem;
    }}
    #logic-header-left {{
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }}
    #gate-select {{
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      border: 1px solid #555;
      background: #222;
      color: #eee;
    }}
    #gate-status {{
      font-weight: 700;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.85rem;
    }}
    #gate-status.pass {{
      background: #123b21;
      color: #a5ffb5;
      border: 1px solid #31c45a;
    }}
    #gate-status.fail {{
      background: #3b1212;
      color: #ffb5b5;
      border: 1px solid #ff5c5c;
    }}
    table.logic-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.7rem;
      font-size: 0.9rem;
    }}
    .logic-table th,
    .logic-table td {{
      border: 1px solid #444;
      padding: 0.25rem 0.35rem;
      text-align: center;
    }}
    .logic-table th {{
      background: #222;
    }}
    .logic-pass {{
      background: #133513;
    }}
    .logic-fail {{
      background: #3d1515;
    }}
    .badge {{
      display: inline-block;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      font-size: 0.75rem;
      border: 1px solid rgba(255,255,255,0.2);
    }}
    @media (max-width: 768px) {{
      .img-shell {{
        max-height: 70vh;
      }}
      #logic-panel {{
        padding: 0.8rem 0.8rem 1.0rem 0.8rem;
      }}
    }}
  </style>
</head>
<body>
  <h1>4-Qubit Transmon Lattice Dashboard</h1>
  <p class="subtitle">Phase-locked start, noisy mean-field dynamics, thermodynamic metrics, and a thermodynamic neuron logic self-test.</p>

  <div class="img-shell">
    <div class="img-wrap">
      <div class="img-inner" id="imgInner">
        <img id="dashboard" src="{gif_name}" alt="Quantum lattice dashboard animation">
      </div>
    </div>
    <div class="zoom-row">
      <span>Zoom:</span>
      <input type="range" id="zoomSlider" min="0.7" max="2.5" step="0.05" value="1.0">
      <span id="zoomValue">1.00×</span>
      <span style="opacity:0.8;">(scroll inside frame if edges exceed the window)</span>
    </div>
  </div>

  <div id="logic-panel">
    <div id="logic-header">
      <div id="logic-header-left">
        <div><strong>Thermodynamic neuron logic self-test</strong></div>
        <div style="font-size:0.85rem; opacity:0.9;">
          Based on virtual-qubit perceptrons implementing NOT, NOR, and 3-MAJORITY. Gate passes if all truth-table rows match.
        </div>
      </div>
      <div>
        <label for="gate-select" style="font-size:0.9rem;">Gate:</label>
        <select id="gate-select">
          <option value="NOT">NOT</option>
          <option value="NOR">NOR</option>
          <option value="3-MAJORITY">3-MAJORITY</option>
        </select>
      </div>
      <div>
        <span id="gate-status" class="badge">Status: &mdash;</span>
      </div>
    </div>

    <table class="logic-table">
      <thead>
        <tr id="logic-header-row"></tr>
      </thead>
      <tbody id="logic-body"></tbody>
    </table>
  </div>

  <script>
    const LOGIC_DATA = {logic_json};

    function perceptronEval(weights, bits) {{
      let y = weights[0];
      for (let i = 0; i < bits.length; i++) {{
        y += weights[i+1] * bits[i];
      }}
      return y >= 0 ? 1 : 0;
    }}

    function renderLogicGate(name) {{
      const data = LOGIC_DATA[name];
      const weights = data.weights;
      const rows = data.rows;

      const headerRow = document.getElementById("logic-header-row");
      const body = document.getElementById("logic-body");
      const statusEl = document.getElementById("gate-status");

      headerRow.innerHTML = "";
      const thInputs = document.createElement("th");
      thInputs.textContent = "Inputs";
      headerRow.appendChild(thInputs);
      const thExpected = document.createElement("th");
      thExpected.textContent = "Target";
      headerRow.appendChild(thExpected);
      const thPred = document.createElement("th");
      thPred.textContent = "Neuron output";
      headerRow.appendChild(thPred);
      const thOK = document.createElement("th");
      thOK.textContent = "Match?";
      headerRow.appendChild(thOK);

      body.innerHTML = "";
      let allPass = true;
      rows.forEach((row) => {{
        const bits = row.input;
        const target = row.out;
        const pred = perceptronEval(weights, bits);

        const tr = document.createElement("tr");
        const tdIn = document.createElement("td");
        tdIn.textContent = "(" + bits.join(", ") + ")";
        tr.appendChild(tdIn);

        const tdExp = document.createElement("td");
        tdExp.textContent = target;
        tr.appendChild(tdExp);

        const tdPred = document.createElement("td");
        tdPred.textContent = pred;
        tr.appendChild(tdPred);

        const tdOk = document.createElement("td");
        const ok = (pred === target);
        tdOk.textContent = ok ? "✓" : "✗";
        tr.classList.add(ok ? "logic-pass" : "logic-fail");
        tr.appendChild(tdOk);

        if (!ok) {{
          allPass = false;
        }}
        body.appendChild(tr);
      }});

      statusEl.textContent = allPass ? "Status: PASS" : "Status: FAIL";
      statusEl.classList.remove("pass", "fail");
      statusEl.classList.add(allPass ? "pass" : "fail");
    }}

    function setupZoom() {{
      const slider = document.getElementById("zoomSlider");
      const valueLabel = document.getElementById("zoomValue");
      const inner = document.getElementById("imgInner");

      function applyZoom() {{
        const z = parseFloat(slider.value);
        inner.style.transform = "scale(" + z.toFixed(2) + ")";
        valueLabel.textContent = z.toFixed(2) + "×";
      }}

      slider.addEventListener("input", applyZoom);
      applyZoom();
    }}

    document.addEventListener("DOMContentLoaded", function() {{
      setupZoom();
      const select = document.getElementById("gate-select");
      select.addEventListener("change", function() {{
        renderLogicGate(select.value);
      }});
      renderLogicGate(select.value);
    }});
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def start_server(path: Path, port: int = 8000):
    from http.server import SimpleHTTPRequestHandler

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(path), **kwargs)

    httpd = socketserver.TCPServer(("", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# ------------------------ CLI ENTRYPOINT -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4-qubit GKSL dashboard + single engine")
    parser.add_argument("--mode", choices=["single", "lattice"], default="lattice")
    parser.add_argument("--q_tmax", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--bath_enable", action="store_true")
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="(kept for backward compatibility; zoom handled in HTML)",
    )
    parser.add_argument("--ngrok", dest="ngrok", action="store_true")
    parser.add_argument("--enable_ngrok", dest="ngrok", action="store_true")
    parser.add_argument(
        "--ngrok_authtoken",
        "--ngrok_token",
        dest="ngrok_authtoken",
        default=None,
    )
    parser.add_argument("--ngrok_port", type=int, default=8000)

    args, _ = parser.parse_known_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = GKSLParams(tmax=args.q_tmax, dt=args.dt)
    bath = BathSchedule(enable=args.bath_enable, phase_per_qubit=math.pi / 4.0)
    mf = MeanFieldSchedule(enable=True, iters=1)

    gif_path: Optional[str] = None

    if args.mode == "single":
        res = evolve_single_gksl(p, bath=bath, store_trajectory=True)
        print("t shape:", res["t"].shape)
        print("pe shape:", res["pe"].shape)
        print("Final excited population ⟨e|ρ|e⟩:", res["pe"][-1])
        print("Total entropy production Σ:", res["Sigma_total"])
    else:
        gif_name = "quantum_dashboard_revised_v23.gif"
        gif_path = animate_lattice_dashboard(
            p,
            str(out_dir / gif_name),
            bath,
            mf,
            fps=args.fps,
        )
        print(f"Done. Saved dashboard GIF to {gif_path}")
        logic_data = thermodynamic_neuron_logic_tests()
        write_index_html(out_dir, Path(gif_path).name, logic_data)

    if args.ngrok and HAVE_NGROK and out_dir.exists():
        httpd = start_server(out_dir, port=args.ngrok_port)

        if args.ngrok_authtoken:
            try:
                ngrok.set_auth_token(args.ngrok_authtoken)
            except Exception:
                pass

        try:
            public_url = ngrok.connect(args.ngrok_port, "http")  # type: ignore[arg-type]
            print(f"Dashboard available at: {public_url.public_url}")
            print("Press Ctrl+C in the kernel to stop the tunnel.")
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Shutting down ngrok tunnel and local server...")
        except Exception as exc:
            print("ngrok failed:", exc)
        finally:
            try:
                ngrok.kill()  # type: ignore[call-arg]
            except Exception:
                pass
            try:
                httpd.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
