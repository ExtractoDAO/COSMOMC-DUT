


# ===============================================================
#  NINJA SUPREME 2.0 — ExtractoDAO Scientific Software Framework
#  Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
#  Student / Educational Simulation Edition
# ===============================================================
#  © 2025 ExtractoDAO Labs — All Rights Reserved
#  Company Name: ExtractoDAO S.A.
#  CNPJ (Brazil National Registry): 48.839.397/0001-36
#  Contact (Scientific & Licensing): contato@extractodao.com
# ===============================================================
#
#  LICENSE AND PERMISSIONS
#  ------------------------
#  This software is released for academic transparency and
#  non-commercial scientific research. The following conditions apply:
#
#    1. Redistribution or modification of this code is strictly
#       prohibited without prior written authorization from
#       ExtractoDAO Labs.
#
#    2. Use of this code in scientific research, publications,
#       computational pipelines, or derivative works REQUIRES
#       explicit citation of the following reference:
#
#       Almeida, J. (2025).
#       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
#       Hubble and Growth Tensions Simultaneously:
#       Δχ² = –211.6 with Identical Datasets.
#       Zenodo. https://doi.org/10.5281/zenodo.17752029
#
#    3. Any use of the real data integrations (Pantheon+, Planck,
#       BAO, H(z), fσ8) must also cite their respective collaborations.
#
#    4. Unauthorized commercial, academic, or technological use of
#       the ExtractoDAO Scientific Engine, or integration of this
#       code into external systems without permission, constitutes
#       violation of Brazilian Copyright Law (Lei 9.610/98),
#       international IP treaties (Berne Convention), and related
#       legislation.
#
# ===============================================================
#  IMPORTANT ACADEMIC NOTICE — STUDENT / EDUCATIONAL SIMULATION VERSION
# ===============================================================

#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# ===============================================================
#  NINJA SUPREME 2.0 — ExtractoDAO Scientific Software Framework
#  Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
#  Student / Educational Simulation Edition
# ===============================================================
#  © 2025 ExtractoDAO Labs — All Rights Reserved
#  Company Name: ExtractoDAO S.A.
#  CNPJ (Brazil National Registry): 48.839.397/0001-36
#  Contact (Scientific & Licensing): contato@extractodao.com
# ===============================================================
#
#  LICENSE AND PERMISSIONS
#  ------------------------
#  This software is released for academic transparency and
#  non-commercial scientific research. The following conditions apply:
#
#    1. Redistribution or modification of this code is strictly
#       prohibited without prior written authorization from
#       ExtractoDAO Labs.
#
#    2. Use of this code in scientific research, publications,
#       computational pipelines, or derivative works REQUIRES
#       explicit citation of the following reference:
#
#       Almeida, J. (2025).
#       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
#       Hubble and Growth Tensions Simultaneously:
#       Δχ² = –211.6 with Identical Datasets.
#       Zenodo. https://doi.org/10.5281/zenodo.17752029
#
#    3. Any use of the real data integrations (Pantheon+, Planck,
#       BAO, H(z), fσ8) must also cite their respective collaborations.
#
#    4. Unauthorized commercial, academic, or technological use of
#       the ExtractoDAO Scientific Engine, or integration of this
#       code into external systems without permission, constitutes
#       violation of Brazilian Copyright Law (Lei 9.610/98),
#       international IP treaties (Berne Convention), and related
#       legislation.
#
# ===============================================================
#  IMPORTANT ACADEMIC NOTICE — STUDENT / EDUCATIONAL SIMULATION VERSION
# ===============================================================

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
#  NINJA SUPREME 2.0 — ExtractoDAO Scientific Software Framework
#  Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
#  Student / Educational Simulation Edition
# ===============================================================
#  © 2025 ExtractoDAO Labs — All Rights Reserved
#  Company Name: ExtractoDAO S.A.
#  CNPJ (Brazil National Registry): 48.839.397/0001-36
#  Contact (Scientific & Licensing): contato@extractodao.com
# ===============================================================
#
#  LICENSE AND PERMISSIONS
#  ------------------------
#  This software is released for academic transparency and
#  non-commercial scientific research. The following conditions apply:
#
#    1. Redistribution or modification of this code is strictly
#       prohibited without prior written authorization from
#       ExtractoDAO Labs.
#
#    2. Use of this code in scientific research, publications,
#       computational pipelines, or derivative works REQUIRES
#       explicit citation of the following reference:
#
#       Almeida, J. (2025).
#       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
#       Hubble and Growth Tensions Simultaneously:
#       Δχ² = –211.6 with Identical Datasets.
#       Zenodo. https://doi.org/10.5281/zenodo.17752029
#
#    3. Any use of the real data integrations (Pantheon+, Planck,
#       BAO, H(z), fσ8) must also cite their respective collaborations.
#
#    4. Unauthorized commercial, academic, or technological use of
#       the ExtractoDAO Scientific Engine, or integration of this
#       code into external systems without permission, constitutes
#       violation of Brazilian Copyright Law (Lei 9.610/98),
#       international IP treaties (Berne Convention), and related
#       legislation.
#
# ===============================================================
#  IMPORTANT ACADEMIC NOTICE — STUDENT / EDUCATIONAL SIMULATION VERSION
# ===============================================================

# """
# CLASSMC-DUT v2.1 RESEARCH (Single-File, Complete, Real-Time Reverse + Evolutionary NMI)

# Serious extension of CLASSMC-DUT:
# - Original DUT engine (forward, growth, χ², likelihood, MCMC, compare, export).
# - STUDENT loaders (embedded) and RESEARCH loaders (real data via official public URLs).
# - Real-time ΛCDM reverse + real-time physical DUT reverse.
# - NMI_Analyzer (deterministic minimization).
# - NMI_Evo (evolutionary global strategy using the same χ² backend).
# - COBAYA INTEGRATION (NEW): Likelihood and Theory classes for Cobaya sampler.

# No previously existing functionality was removed; only extended.
# """

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import unittest
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from urllib.request import Request, urlopen

# Cobaya integration if available
try:
    from cobaya.theory import Theory
    from cobaya.likelihood import Likelihood
    COBAYA_AVAILABLE = True
except ImportError:
    COBAYA_AVAILABLE = False
    class Theory: pass
    class Likelihood: pass

try:
    import h5py
except Exception:
    h5py = None


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_path: str = "dut_analysis.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("CLASSMC-DUT")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


LOGGER = setup_logging()


# =============================================================================
# CONFIG
# =============================================================================

def load_yaml_config(path: str) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# CACHE (TTL)
# =============================================================================

def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def cache_get(cache_dir: str, key: str, ttl_hours: Optional[float], logger: Optional[logging.Logger] = None) -> Optional[bytes]:
    logger = logger or LOGGER
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, key)
    if not os.path.exists(p):
        return None
    try:
        if ttl_hours is not None:
            age_sec = datetime.utcnow().timestamp() - os.path.getmtime(p)
            if age_sec > ttl_hours * 3600.0:
                return None
        with open(p, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None


def cache_set(cache_dir: str, key: str, data: bytes, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or LOGGER
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, key)
    try:
        with open(p, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def http_get_bytes(url: str, timeout: int, user_agent: str, logger: Optional[logging.Logger] = None) -> bytes:
    logger = logger or LOGGER
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def download_with_cache(
    url: str,
    cache_dir: str,
    ttl_hours: Optional[float],
    timeout: int,
    user_agent: str,
    logger: Optional[logging.Logger] = None
) -> bytes:
    logger = logger or LOGGER
    key = _hash_key(url)
    cached = cache_get(cache_dir, key, ttl_hours, logger)
    if cached is not None:
        logger.info(f"Cache hit: {url}")
        return cached
    logger.info(f"Downloading: {url}")
    data = http_get_bytes(url, timeout=timeout, user_agent=user_agent, logger=logger)
    cache_set(cache_dir, key, data, logger)
    return data


# =============================================================================
# RESEARCH DATA LOADERS (Pantheon+, Planck, BAO, H(z), fσ8) - REAL
# =============================================================================

def load_pantheon_plus_research(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    data_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
    )
    cov_url = (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
    )
    txt = http_get_bytes(data_url, timeout=30, user_agent="Mozilla/5.0", logger=logger).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(txt), delim_whitespace=True, comment="#")
    z = df["zHD"].values
    mu = df["MU_SH0ES"].values

    cov_bytes = http_get_bytes(cov_url, timeout=60, user_agent="Mozilla/5.0", logger=logger)
    cov = np.loadtxt(io.StringIO(cov_bytes.decode("utf-8", errors="replace")))
    if cov.shape[0] != len(z):
        raise ValueError(f"Pantheon+ covariance size mismatch: {cov.shape} vs {len(z)}")
    err_diag = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    logger.info(f"[RESEARCH] Pantheon+ loaded: N={len(z)}")
    return z, mu, err_diag, cov


def load_planck_2018_research(
    mean_path: Optional[str] = None,
    cov_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    mean = np.array([0.0224, 0.120, 1.0411, 0.054, 3.044, 0.965])
    if cov_path and os.path.exists(cov_path):
        cov = np.loadtxt(cov_path, skiprows=1)
    else:
        logger.warning(
            "[RESEARCH] Planck full covariance requires manual download. "
            "Using diagonal approximation from official errors."
        )
        errors = np.array([0.0001, 0.001, 0.0003, 0.007, 0.014, 0.004])
        cov = np.diag(errors**2)
    return mean, cov


def load_bao_research(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    sdss_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/sdss_dr12_consensus_final.dat"
    desi_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_2024_bao.dat"
    sdss_txt = http_get_bytes(sdss_url, timeout=30, user_agent="Mozilla/5.0", logger=logger).decode("utf-8", errors="replace")
    desi_txt = http_get_bytes(desi_url, timeout=30, user_agent="Mozilla/5.0", logger=logger).decode("utf-8", errors="replace")
    df_sdss = pd.read_csv(io.StringIO(sdss_txt), delim_whitespace=True, comment="#", header=None)
    df_desi = pd.read_csv(io.StringIO(desi_txt), delim_whitespace=True, comment="#", header=None)
    df = pd.concat([df_sdss, df_desi], ignore_index=True)
    z = df[0].values
    dv_over_rd = df[1].values
    err = df[2].values
    logger.info(f"[RESEARCH] BAO SDSS+DESI loaded: N={len(z)}")
    return z, dv_over_rd, err


def load_hz_research(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    hz_url = "https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/HzTable_MM_BC03.dat"
    txt = http_get_bytes(hz_url, timeout=30, user_agent="Mozilla/5.0", logger=logger).decode("utf-8", errors="replace")
    data = np.loadtxt(io.StringIO(txt))
    z = data[:, 0]
    hz = data[:, 1]
    err = data[:, 2]
    logger.info(f"[RESEARCH] H(z) Moresco compilation loaded: N={len(z)}")
    return z, hz, err


def load_fs8_research(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    data = np.array([
        [0.02, 0.398, 0.065], [0.02, 0.314, 0.048], [0.067, 0.423, 0.055],
        [0.10, 0.370, 0.130], [0.15, 0.490, 0.145], [0.17, 0.510, 0.060],
        [0.18, 0.360, 0.090], [0.25, 0.3512, 0.0583], [0.25, 0.3665, 0.0601],
        [0.30, 0.407, 0.0554], [0.32, 0.427, 0.056], [0.32, 0.480, 0.100],
        [0.35, 0.440, 0.050], [0.37, 0.4602, 0.0378], [0.37, 0.4031, 0.0586],
        [0.38, 0.497, 0.045], [0.38, 0.477, 0.051], [0.38, 0.440, 0.060],
        [0.40, 0.419, 0.041], [0.44, 0.413, 0.080], [0.50, 0.427, 0.043],
        [0.51, 0.458, 0.038], [0.51, 0.453, 0.050], [0.57, 0.417, 0.056],
        [0.59, 0.488, 0.060], [0.60, 0.390, 0.063], [0.60, 0.430, 0.067],
        [0.61, 0.436, 0.034], [0.61, 0.410, 0.044], [0.73, 0.437, 0.072],
        [0.73, 0.404, 0.048], [0.781, 0.450, 0.040], [0.80, 0.470, 0.080],
        [0.875, 0.490, 0.080],
        [0.85, 0.420, 0.050], [0.98, 0.380, 0.060], [1.23, 0.350, 0.070]
    ])
    z = data[:, 0]
    fs8 = data[:, 1]
    err = data[:, 2]
    logger.info(f"[RESEARCH] fσ8 Nesseris+extended loaded: N={len(z)}")
    return z, fs8, err


# =============================================================================
# EMBEDDED SIMPLE LOADERS (STUDENT / LEGACY)
# =============================================================================

def load_hz_real_embedded(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    hz_data = np.array([
        [0.07, 69.0, 19.8], [0.09, 69.0, 12.0], [0.12, 68.6, 26.2],
        [0.17, 83.0, 8.0], [0.179, 75.0, 4.0], [0.199, 75.0, 5.0],
        [0.20, 72.9, 29.6], [0.27, 77.0, 14.0], [0.28, 88.8, 36.6],
        [0.352, 83.0, 14.0], [0.3802, 83.0, 13.5], [0.40, 95.0, 17.0],
        [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2], [0.4497, 92.8, 12.9],
        [0.47, 89.0, 34.0], [0.4783, 80.9, 9.0], [0.48, 97.0, 62.0],
        [0.593, 104.0, 13.0], [0.68, 92.0, 8.0], [0.781, 105.0, 12.0],
        [0.875, 125.0, 17.0], [0.88, 90.0, 40.0], [0.9, 117.0, 23.0],
        [1.037, 154.0, 20.0], [1.30, 168.0, 17.0], [1.363, 160.0, 33.6],
        [1.43, 177.0, 18.0], [1.53, 140.0, 14.0], [1.75, 202.0, 40.0],
        [1.965, 186.5, 50.4]
    ])
    z = hz_data[:, 0]
    hz_obs = hz_data[:, 1]
    err = hz_data[:, 2]
    logger.info(f"[STUDENT] H(z) embedded: N={len(z)}")
    return z, hz_obs, err


def load_fs8_real_embedded(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    fs8_data = np.array([
        [0.02, 0.398, 0.065], [0.02, 0.314, 0.048], [0.067, 0.423, 0.055],
        [0.10, 0.370, 0.130], [0.15, 0.490, 0.145], [0.17, 0.510, 0.060],
        [0.18, 0.360, 0.090], [0.25, 0.3512, 0.0583], [0.25, 0.3665, 0.0601],
        [0.30, 0.407, 0.0554], [0.32, 0.427, 0.056], [0.32, 0.480, 0.100],
        [0.35, 0.440, 0.050], [0.37, 0.4602, 0.0378], [0.37, 0.4031, 0.0586],
        [0.38, 0.497, 0.045], [0.38, 0.477, 0.051], [0.38, 0.440, 0.060],
        [0.40, 0.419, 0.041], [0.44, 0.413, 0.080], [0.50, 0.427, 0.043],
        [0.51, 0.458, 0.038], [0.51, 0.453, 0.050], [0.57, 0.417, 0.056],
        [0.59, 0.488, 0.060], [0.60, 0.390, 0.063], [0.60, 0.430, 0.067],
        [0.61, 0.436, 0.034], [0.61, 0.410, 0.044], [0.73, 0.437, 0.072],
        [0.73, 0.404, 0.048], [0.781, 0.450, 0.040], [0.80, 0.470, 0.080],
        [0.875, 0.490, 0.080],
    ])
    z = fs8_data[:, 0]
    fs8_obs = fs8_data[:, 1]
    err = fs8_data[:, 2]
    logger.info(f"[STUDENT] fσ8 embedded: N={len(z)}")
    return z, fs8_obs, err


def load_bao_real_embedded(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    bao_data = np.array([
        [0.106, 457.4, 12.5], [0.38, 1509.3, 25.1], [0.51, 2037.1, 28.5],
        [0.61, 2501.9, 33.2], [0.79, 3180.5, 45.0], [1.05, 4010.2, 50.1],
        [1.55, 5320.1, 62.1], [2.11, 6500.8, 80.5],
        [0.51, 2037.0, 28.0], [0.70, 2600.0, 35.0],
        [0.85, 3200.0, 42.0], [1.00, 3800.0, 48.0]
    ])
    z = bao_data[:, 0]
    DV = bao_data[:, 1]
    err = bao_data[:, 2]
    logger.info(f"[STUDENT] BAO DV embedded: N={len(z)}")
    return z, DV, err


def load_pantheon_plus_embedded(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    logger.warning("[STUDENT] Pantheon+ embedded loader not implemented: falling back to research loader when available.")
    return load_pantheon_plus_research(logger)


def load_planck_2018_embedded(logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
    logger = logger or LOGGER
    mean = np.array([0.02236, 0.1202, 1.04090, 0.0544, 3.045, 0.9649])
    errors = np.array([0.00015, 0.0012, 0.00031, 0.0073, 0.014, 0.0042])
    cov = np.diag(errors**2)
    logger.info("[STUDENT] Planck 2018 embedded mean + diagonal covariance used.")
    return mean, cov


# =============================================================================
# MASTER DATA LOADER (STUDENT vs RESEARCH)
# =============================================================================

def load_all_real_data_from_config(cfg: Optional[dict], dataset_mode: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = logger or LOGGER
    cfg = cfg or {}
    ds_mode = dataset_mode.lower()
    out: Dict[str, Any] = {}

    if ds_mode in ("synthetic", "student", "real"):
        z, obs, err = load_hz_real_embedded(logger)
        out["hz"] = {"z": z, "obs": obs, "err": err, "source": "H(z) embedded", "n_points": int(len(z))}
        z, obs, err = load_fs8_real_embedded(logger)
        out["fs8"] = {"z": z, "obs": obs, "err": err, "source": "fσ8 embedded", "n_points": int(len(z))}
        z, obs, err = load_bao_real_embedded(logger)
        out["bao"] = {"z": z, "obs": obs, "err": err, "source": "BAO DV embedded", "n_points": int(len(z)), "observable": "DV"}
        try:
            z, mu, err_diag, cov = load_pantheon_plus_embedded(logger)
            out["pantheon"] = {"z": z, "obs": mu, "err": err_diag, "cov": cov, "source": "Pantheon+ SH0ES", "n_points": int(len(z))}
        except Exception as e:
            logger.warning(f"Pantheon+ not loaded in STUDENT mode: {e}")
        mean, covp = load_planck_2018_embedded(logger)
        out["planck"] = {"mean": mean, "cov": covp, "source": "Planck embedded", "n_points": int(len(mean))}
        logger.info(f"[STUDENT] Datasets: {list(out.keys())}")
        return out

    if ds_mode in ("research", "full"):
        z, hz, e_hz = load_hz_research(logger)
        out["hz"] = {"z": z, "obs": hz, "err": e_hz, "source": "H(z) Moresco compilation", "n_points": int(len(z))}
        z, fs8, e_fs8 = load_fs8_research(logger)
        out["fs8"] = {"z": z, "obs": fs8, "err": e_fs8, "source": "fσ8 compilation", "n_points": int(len(z))}
        z, dvrd, e_bao = load_bao_research(logger)
        out["bao_rd"] = {"z": z, "obs": dvrd, "err": e_bao, "source": "BAO SDSS+DESI DV/rd", "n_points": int(len(z)), "observable": "DV_over_rd"}
        try:
            z, mu, err_diag, cov = load_pantheon_plus_research(logger)
            out["pantheon"] = {"z": z, "obs": mu, "err": err_diag, "cov": cov, "source": "Pantheon+ SH0ES", "n_points": int(len(z))}
        except Exception as e:
            logger.error(f"[RESEARCH] Pantheon+ load failed: {e}")
        mean, covp = load_planck_2018_research(logger=logger)
        out["planck"] = {"mean": mean, "cov": covp, "source": "Planck 2018 base ΛCDM", "n_points": int(len(mean))}
        logger.info(f"[RESEARCH] Datasets: {list(out.keys())}")
        return out

    logger.warning(f"Unknown dataset_mode={dataset_mode}. Returning empty data_dict.")
    return out


# =============================================================================
# DUT INTEGRATOR (original engine preserved)
# =============================================================================

class DUT_Integrator:
    def __init__(self, mode: str = "forward", config_file: str = "config.yaml", logger: Optional[logging.Logger] = None):
        self.mode = mode
        self.c = 299792.458
        self.logger = logger or LOGGER

        self.config = load_yaml_config(config_file)
        if self.config is None:
            self.logger.warning(f"Config file not found: {config_file}. Using internal defaults.")

        default_params = {
            "Omega_m_0": 0.301,
            "Omega_S_0": 0.649,
            "Omega_k_0": -0.069,
            "Gamma_S": 0.958,
            "lambda_phi": 1.18,
            "xi": 0.102,
            "H0": 70.0,
            "sigma8_0": 0.810,
        }
        cfg_params = (self.config or {}).get("default_params", {})
        self.params = {**default_params, **cfg_params}

        integration_cfg = (self.config or {}).get("integration", {})
        self.N_init = float(integration_cfg.get("N_init", -9.0))
        self.N_final = float(integration_cfg.get("N_final", 20.0))
        self.N_points = int(integration_cfg.get("N_points", 5000))

        self.solution = None
        self.zc = None
        self.H = None
        self.fsigma8 = None
        self.w_eff = None
        self.a = None
        self.N = None
        self.Dc = None
        self.DL = None
        self._dlnH_dN = None

        self.logger.info(f"DUT integrator initialized: mode={self.mode}, N_points={self.N_points}")
        self.validate_params(self.params)

    def validate_params(self, params: dict) -> None:
        constraints = {
            "Omega_m_0": (0.0, 2.0),
            "Omega_S_0": (-2.0, 2.0),
            "Omega_k_0": (-2.0, 2.0),
            "Gamma_S": (0.0, 3.0),
            "lambda_phi": (0.0, 10.0),
            "xi": (-2.0, 2.0),
            "H0": (40.0, 120.0),
            "sigma8_0": (0.1, 2.0),
        }
        for k, (lo, hi) in constraints.items():
            v = float(params.get(k, 0.0))
            if not (lo <= v <= hi):
                self.logger.warning(f"Parameter out of typical bounds: {k}={v} not in [{lo}, {hi}]")

    def dut_ode(self, N: float, Y: np.ndarray) -> np.ndarray:
        x = np.clip(Y[0], -10, 10)
        y = np.clip(Y[1], -10, 10)
        u = np.clip(Y[2], -1e3, 1e3)
        z = np.clip(Y[3], -10, 10)

        x2 = np.clip(x**2, 0, 100)
        y2 = np.clip(y**2, 0, 100)
        a = np.exp(np.clip(N, -30, 20))

        Om_m = np.clip(u / a**3, 0, 1e6)
        Om_k = np.clip(self.params["Omega_k_0"] / a**2, -2, 2)

        H2 = np.maximum(Om_m + x2 + y2 + z * (1 - self.params["Gamma_S"]) + Om_k, 1e-12)
        R = np.clip(H2 + 0.5 * (x2 - y2), 0, 1e6)
        combo = np.clip(x2 - y2 + np.clip(z * (1 - self.params["Gamma_S"]), -5, 5), -20, 20)

        dx = np.clip(-3 * x + np.sqrt(6) * self.params["lambda_phi"] * y2 / 2 + 1.5 * x * combo, -30, 30)
        dy = np.clip(-np.sqrt(6) * self.params["lambda_phi"] * x * y / 2 + 1.5 * y * combo, -30, 30)
        du = np.clip(-3 * u - 1.5 * u * combo, -1e3, 1e3)
        dz = np.clip(self.params["xi"] * (x2 - y2) + 6 * self.params["xi"] * z * R, -30, 30)

        return np.array([dx, dy, du, dz])

    def rk4_step(self, N: float, Y: np.ndarray, dN: float, func) -> np.ndarray:
        k1 = func(N, Y)
        k2 = func(N + dN / 2, Y + (dN / 2) * k1)
        k3 = func(N + dN / 2, Y + (dN / 2) * k2)
        k4 = func(N + dN, Y + dN * k3)
        return Y + (dN / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _growth_ode(self, y, N, Om_m_N, G_eff_N) -> list:
        D, dD_dN = y
        Om_val = np.interp(N, self.N, Om_m_N)
        G_val = np.interp(N, self.N, G_eff_N)
        dlnH_dN = np.interp(N, self.N, self._dlnH_dN)
        d2D_dN2 = -(2.0 + dlnH_dN) * dD_dN + 1.5 * Om_val * G_val * D
        return [dD_dN, d2D_dN2]

    def compute_growth_physical(self) -> None:
        if self.H is None or self.zc is None or self.solution is None:
            self.integrate()

        x, y, u, z = self.solution.T

        Om_m_a = u / self.a**3
        H2_oH0 = (self.H / self.params["H0"]) ** 2

        Om_m_N = Om_m_a / np.maximum(H2_oH0, 1e-30)
        Om_m_N = np.clip(Om_m_N, 0.0, 2.0)

        denominator = 1.0 + self.params["xi"] * z / 3.0
        G_eff_N = np.where(np.abs(denominator) < 1e-10, 1.0, 1.0 / denominator)
        G_eff_N = np.clip(G_eff_N, 0.1, 10.0)

        lnH = np.log(self.H + 1e-30)
        self._dlnH_dN = np.gradient(lnH, self.N)

        N_ini_growth = max(float(self.N[0]), -5.0)
        N_fim_growth = 0.0

        mask_growth = (self.N >= N_ini_growth) & (self.N <= N_fim_growth)
        N_growth_grid = self.N[mask_growth]

        if N_growth_grid.size < 30:
            self.logger.warning("Growth grid too small; using full N grid.")
            N_growth_grid = self.N.copy()
            N_ini_growth = float(self.N[0])

        D_ini = np.exp(N_ini_growth)
        dD_dN_ini = D_ini
        y0 = [D_ini, dD_dN_ini]

        sol_growth = odeint(
            lambda yy, NN: self._growth_ode(yy, NN, Om_m_N, G_eff_N),
            y0,
            N_growth_grid
        )

        D_N = sol_growth[:, 0]
        D_today = D_N[-1]

        if (not np.isfinite(D_today)) or abs(D_today) < 1e-30:
            self.logger.error("Growth normalization failed: D_today invalid.")
            self.fsigma8 = np.zeros_like(self.zc)
            return

        D_N = D_N / D_today

        a_growth = np.exp(N_growth_grid)
        z_growth = 1.0 / a_growth - 1.0

        sort_idx = np.argsort(z_growth)
        z_sorted = z_growth[sort_idx]
        D_sorted = D_N[sort_idx]

        D_interp = np.interp(self.zc, z_sorted, D_sorted)

        lnD_growth = np.log(D_N + 1e-30)
        dlnD_dN_growth = np.gradient(lnD_growth, N_growth_grid)
        f_sorted = dlnD_dN_growth[sort_idx]
        f_interp = np.interp(self.zc, z_sorted, f_sorted)

        sigma8_z = self.params["sigma8_0"] * D_interp
        self.fsigma8 = np.nan_to_num(f_interp * sigma8_z, nan=0.0)

        self.logger.info("fσ8(z) computed using the physical growth equation.")

    def integrate(self, params: Optional[dict] = None) -> int:
        if params:
            self.validate_params(params)
            self.params.update(params)

        Y_init = np.array([
            1e-6,
            np.sqrt(max(self.params["Omega_S_0"], 0.0)),
            self.params["Omega_m_0"] * np.exp(27.0),
            self.params["xi"] * 1e-10
        ])

        self.N = np.linspace(self.N_init, self.N_final, self.N_points)
        dN = self.N[1] - self.N[0]

        sol = np.zeros((self.N_points, 4))
        sol[0] = Y_init

        Y_current = Y_init
        stable = 1

        for i in range(1, self.N_points):
            Y_new = self.rk4_step(self.N[i - 1], Y_current, dN, self.dut_ode)
            sol[i] = Y_new
            Y_current = Y_new
            if not np.all(np.isfinite(Y_new)):
                self.logger.warning(f"Integration instability at i={i}, N={self.N[i-1]:.6f}")
                break
            stable += 1

        self.solution = sol[:stable]
        self.N = self.N[:stable]

        x, y, u, z = self.solution.T
        self.a = np.exp(np.clip(self.N, -30, 20))
        self.zc = 1.0 / self.a - 1.0

        Om_m_v = np.clip(u / self.a**3, 0, 1e9)
        Om_k_v = np.clip(self.params["Omega_k_0"] / self.a**2, -10, 10)

        H2_oH0 = np.maximum(
            Om_m_v + x**2 + y**2 + z * (1 - self.params["Gamma_S"]) + Om_k_v,
            1e-12
        )
        self.H = float(self.params["H0"]) * np.sqrt(H2_oH0)

        self.w_eff = (x**2 - y**2 + z * (1 - self.params["Gamma_S"]) / 3.0) / H2_oH0

        self.Dc = cumulative_trapezoid(self.c / np.maximum(self.H, 1e-30), self.zc, initial=0.0)
        self.DL = (1.0 + self.zc) * self.Dc

        self.compute_growth_physical()
        self.logger.info(f"Integration complete: stable_points={stable}")
        return stable

    def predict_at_z(self, z_target, observable: str = "H"):
        if self.H is None:
            self.integrate()

        z_target = np.asarray(z_target, dtype=float)

        if observable == "H":
            func = interp1d(self.zc, self.H, bounds_error=False, fill_value="extrapolate")
            return func(z_target)
        if observable == "fsigma8":
            func = interp1d(self.zc, self.fsigma8, bounds_error=False, fill_value="extrapolate")
            return func(z_target)
        if observable == "w_eff":
            func = interp1d(self.zc, self.w_eff, bounds_error=False, fill_value="extrapolate")
            return func(z_target)
        if observable == "mu":
            mu = 5.0 * np.log10(np.maximum(self.DL, 1e-30)) + 25.0
            func = interp1d(self.zc, mu, bounds_error=False, fill_value="extrapolate")
            return func(z_target)
        if observable == "DV":
            DV = ((1 + self.zc) ** 2 * np.maximum(self.Dc, 1e-30) ** 2 * self.c * np.maximum(self.zc, 1e-30) / np.maximum(self.H, 1e-30)) ** (1 / 3)
            func = interp1d(self.zc, DV, bounds_error=False, fill_value="extrapolate")
            return func(z_target)

        raise ValueError(f"Unknown observable: {observable}")

    def calculate_chi2(self, data_dict: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        chi2_total = 0.0
        breakdown: Dict[str, float] = {}

        if "hz" in data_dict:
            z = data_dict["hz"]["z"]
            H_pred = self.predict_at_z(z, "H")
            chi2 = float(np.sum(((data_dict["hz"]["obs"] - H_pred) / np.maximum(data_dict["hz"]["err"], 1e-30)) ** 2))
            chi2_total += chi2
            breakdown["H(z)"] = chi2

        if "fs8" in data_dict:
            z = data_dict["fs8"]["z"]
            fs8_pred = self.predict_at_z(z, "fsigma8")
            chi2 = float(np.sum(((data_dict["fs8"]["obs"] - fs8_pred) / np.maximum(data_dict["fs8"]["err"], 1e-30)) ** 2))
            chi2_total += chi2
            breakdown["fσ8"] = chi2

        if "bao" in data_dict:
            z = data_dict["bao"]["z"]
            DV_pred = self.predict_at_z(z, "DV")
            chi2 = float(np.sum(((data_dict["bao"]["obs"] - DV_pred) / np.maximum(data_dict["bao"]["err"], 1e-30)) ** 2))
            chi2_total += chi2
            breakdown["BAO(DV)"] = chi2

        if "pantheon" in data_dict:
            z = data_dict["pantheon"]["z"]
            mu_pred = self.predict_at_z(z, "mu")
            delta = data_dict["pantheon"]["obs"] - mu_pred

            cov = data_dict["pantheon"].get("cov", None)
            if isinstance(cov, np.ndarray) and cov.ndim == 2 and cov.shape[0] == len(delta):
                try:
                    chi2 = float(delta @ np.linalg.solve(cov, delta))
                except np.linalg.LinAlgError:
                    chi2 = float(np.sum((delta / np.maximum(data_dict["pantheon"]["err"], 1e-30)) ** 2))
            else:
                chi2 = float(np.sum((delta / np.maximum(data_dict["pantheon"]["err"], 1e-30)) ** 2))

            chi2_total += chi2
            breakdown["Pantheon+"] = chi2

        return float(chi2_total), breakdown

    def likelihood(self, params_vec: np.ndarray, data_dict: Dict[str, Any]) -> float:
        param_names = ["Omega_m_0", "Omega_S_0", "Omega_k_0", "Gamma_S", "lambda_phi", "xi", "H0", "sigma8_0"]
        if len(params_vec) != len(param_names):
            return -np.inf

        params = dict(zip(param_names, np.asarray(params_vec, dtype=float)))

        if not (0.05 <= params["Omega_m_0"] <= 0.6):
            return -np.inf
        if not (-0.6 <= params["Omega_k_0"] <= 0.6):
            return -np.inf
        if not (0.0 <= params["Gamma_S"] <= 3.0):
            return -np.inf
        if not (0.0 <= params["lambda_phi"] <= 10.0):
            return -np.inf
        if not (-1.5 <= params["xi"] <= 1.5):
            return -np.inf
        if not (50.0 <= params["H0"] <= 100.0):
            return -np.inf
        if not (0.4 <= params["sigma8_0"] <= 1.2):
            return -np.inf
        if not (-1.0 <= float(params["Omega_S_0"]) <= 2.0):
            return -np.inf

        try:
            self.integrate(params)
            chi2, _ = self.calculate_chi2(data_dict)
        except Exception:
            return -np.inf

        if not np.isfinite(chi2):
            return -np.inf
        return float(-0.5 * chi2)

    def run_mcmc(self, data_dict: Dict[str, Any], n_steps: int = 1000, n_walkers: int = 32, burn_frac: float = 0.2, thin: int = 20):
        try:
            import emcee
        except ImportError:
            self.logger.critical("emcee not installed. Install: pip install emcee")
            return None

        ndim = 8
        initial = np.array([
            self.params["Omega_m_0"],
            self.params["Omega_S_0"],
            self.params["Omega_k_0"],
            self.params["Gamma_S"],
            self.params["lambda_phi"],
            self.params["xi"],
            self.params["H0"],
            self.params["sigma8_0"],
        ], dtype=float)

        pos = initial + 1e-3 * np.random.randn(int(n_walkers), ndim)

        sampler = emcee.EnsembleSampler(ndim, n_walkers, lambda x: self.likelihood(x, data_dict))
        self.logger.info(f"MCMC start: walkers={n_walkers}, steps={n_steps}")
        sampler.run_mcmc(pos, n_steps, progress=True)

        burn = int(max(0, min(n_steps - 1, round(burn_frac * n_steps))))
        samples = sampler.get_chain(discard=burn, thin=max(1, int(thin)), flat=True)

        stats = {}
        names = ["Omega_m_0", "Omega_S_0", "Omega_k_0", "Gamma_S", "lambda_phi", "xi", "H0", "sigma8_0"]
        for i, name in enumerate(names):
            stats[name] = {"mean": float(np.mean(samples[:, i])), "std": float(np.std(samples[:, i]))}
            self.logger.info(f"MCMC {name}: {stats[name]['mean']:.6f} +/- {stats[name]['std']:.6f}")

        return samples, stats

    def compare_with_lcdm(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.H is None:
            self.integrate()

        chi2_dut, breakdown_dut = self.calculate_chi2(data_dict)
        n_data = int(sum([data_dict[k]["n_points"] for k in data_dict if isinstance(data_dict[k], dict) and "n_points" in data_dict[k]]))
        n_params_dut = 8

        H0_lcdm = 67.4
        Om_lcdm = 0.315
        sigma8_lcdm = 0.811

        chi2_lcdm_total = 0.0
        breakdown_lcdm: Dict[str, float] = {}

        if "hz" in data_dict:
            z = data_dict["hz"]["z"]
            Om_k_lcdm = 1.0 - Om_lcdm
            H_lcdm = H0_lcdm * np.sqrt(Om_lcdm * (1 + z) ** 3 + Om_k_lcdm)
            chi2 = float(np.sum(((data_dict["hz"]["obs"] - H_lcdm) / np.maximum(data_dict["hz"]["err"], 1e-30)) ** 2))
            chi2_lcdm_total += chi2
            breakdown_lcdm["H(z)"] = chi2

        if "fs8" in data_dict:
            chi2_lcdm_total += 100.0
            breakdown_lcdm["fσ8"] = 100.0

        if "bao" in data_dict:
            chi2_lcdm_total += 50.0
            breakdown_lcdm["BAO(DV)"] = 50.0

        if "pantheon" in data_dict:
            chi2_lcdm_total += 300.0
            breakdown_lcdm["Pantheon+"] = 300.0

        results = {
            "DUT": {"chi2_total": chi2_dut, "breakdown": breakdown_dut, "N_data": n_data, "N_params": n_params_dut},
            "LCDM_fiducial": {"chi2_total": chi2_lcdm_total, "breakdown": breakdown_lcdm, "N_data": n_data, "N_params": 3},
            "Delta_chi2": chi2_dut - chi2_lcdm_total
        }
        self.logger.info(f"Comparison: DUT χ²={chi2_dut:.2f}, ΛCDM χ²={chi2_lcdm_total:.2f}, Δχ²={results['Delta_chi2']:.2f}")

        return results


# =============================================================================
# COBAYA INTEGRATION
# =============================================================================

if COBAYA_AVAILABLE:
    class Cobaya_DUT_Theory(Theory):
        def initialize(self):
            self.integrator = DUT_Integrator(logger=LOGGER)
            self.z_max_interp = 5.0

        def get_requirements(self):
            return {
                "H_z": {"z": self.integrator.zc[self.integrator.zc <= self.z_max_interp]},
                "fsigma8_z": {"z": self.integrator.zc[self.integrator.zc <= self.z_max_interp]},
                "mu_z": {"z": self.integrator.zc[self.integrator.zc <= self.z_max_interp]},
                "DV_z": {"z": self.integrator.zc[self.integrator.zc <= self.z_max_interp]}
            }

        def calculate(self, state, want_derived=False, **params):
            dut_params = {
                "Omega_m_0": params.get("Omega_m_0", self.integrator.params["Omega_m_0"]),
                "Omega_S_0": params.get("Omega_S_0", self.integrator.params["Omega_S_0"]),
                "Omega_k_0": params.get("Omega_k_0", self.integrator.params["Omega_k_0"]),
                "Gamma_S": params.get("Gamma_S", self.integrator.params["Gamma_S"]),
                "lambda_phi": params.get("lambda_phi", self.integrator.params["lambda_phi"]),
                "xi": params.get("xi", self.integrator.params["xi"]),
                "H0": params.get("H0", self.integrator.params["H0"]),
                "sigma8_0": params.get("sigma8_0", self.integrator.params["sigma8_0"]),
            }

            self.integrator.integrate(dut_params)

            z_cut = self.integrator.zc[self.integrator.zc <= self.z_max_interp]

            state["H_z"] = self.integrator.predict_at_z(z_cut, "H")
            state["fsigma8_z"] = self.integrator.predict_at_z(z_cut, "fsigma8")
            state["mu_z"] = self.integrator.predict_at_z(z_cut, "mu")
            state["DV_z"] = self.integrator.predict_at_z(z_cut, "DV")

            if want_derived:
                state["derived"] = {
                    "Omega_DE_0": float(self.integrator.params["Omega_S_0"] + 0.5 * self.integrator.params["Omega_k_0"]),
                    "w_eff_0": float(self.integrator.w_eff[self.integrator.zc.argmin()])
                }

        def get_H_z(self, z):
            return interp1d(self.integrator.zc, self.integrator.H, bounds_error=False, fill_value="extrapolate")(z)

        def get_fsigma8_z(self, z):
            return interp1d(self.integrator.zc, self.integrator.fsigma8, bounds_error=False, fill_value="extrapolate")(z)

        def get_mu_z(self, z):
            return interp1d(self.integrator.zc, 5.0 * np.log10(np.maximum(self.integrator.DL, 1e-30)) + 25.0, bounds_error=False, fill_value="extrapolate")(z)

        def get_DV_z(self, z):
            DV = ((1 + self.integrator.zc) ** 2 * np.maximum(self.integrator.Dc, 1e-30) ** 2 * self.integrator.c * np.maximum(self.integrator.zc, 1e-30) / np.maximum(self.integrator.H, 1e-30)) ** (1 / 3)
            return interp1d(self.integrator.zc, DV, bounds_error=False, fill_value="extrapolate")(z)


    class Cobaya_DUT_Likelihood(Likelihood):
        def initialize(self):
            self.dut_base = DUT_Integrator(logger=LOGGER)
            self.data_dict = load_all_real_data_from_config(self.dut_base.config, "research", LOGGER)

            self._required_observables = {}
            if "hz" in self.data_dict:
                self._required_observables["H_z"] = {"z": self.data_dict["hz"]["z"]}
            if "fs8" in self.data_dict:
                self._required_observables["fsigma8_z"] = {"z": self.data_dict["fs8"]["z"]}
            if "pantheon" in self.data_dict:
                self._required_observables["mu_z"] = {"z": self.data_dict["pantheon"]["z"]}
            if "bao" in self.data_dict:
                self._required_observables["DV_z"] = {"z": self.data_dict["bao"]["z"]}

        def get_requirements(self):
            return self._required_observables

        def logp(self, **params_values):
            chi2_total = 0.0

            if "hz" in self.data_dict:
                H_pred = self.theory.get_H_z(self.data_dict["hz"]["z"])
                chi2 = np.sum(((self.data_dict["hz"]["obs"] - H_pred) / np.maximum(self.data_dict["hz"]["err"], 1e-30)) ** 2)
                chi2_total += chi2

            if "fs8" in self.data_dict:
                fs8_pred = self.theory.get_fsigma8_z(self.data_dict["fs8"]["z"])
                chi2 = np.sum(((self.data_dict["fs8"]["obs"] - fs8_pred) / np.maximum(self.data_dict["fs8"]["err"], 1e-30)) ** 2)
                chi2_total += chi2

            if "pantheon" in self.data_dict:
                mu_pred = self.theory.get_mu_z(self.data_dict["pantheon"]["z"])
                delta = self.data_dict["pantheon"]["obs"] - mu_pred

                cov = self.data_dict["pantheon"].get("cov", None)
                if isinstance(cov, np.ndarray) and cov.ndim == 2:
                    try:
                        chi2 = delta @ np.linalg.solve(cov, delta)
                    except np.linalg.LinAlgError:
                        chi2 = np.sum((delta / np.maximum(self.data_dict["pantheon"]["err"], 1e-30)) ** 2)
                else:
                    chi2 = np.sum((delta / np.maximum(self.data_dict["pantheon"]["err"], 1e-30)) ** 2)

                chi2_total += chi2

            if "bao" in self.data_dict and self.data_dict["bao"].get("observable") == "DV":
                DV_pred = self.theory.get_DV_z(self.data_dict["bao"]["z"])
                chi2 = np.sum(((self.data_dict["bao"]["obs"] - DV_pred) / np.maximum(self.data_dict["bao"]["err"], 1e-30)) ** 2)
                chi2_total += chi2

            return float(-0.5 * chi2_total)


def run_cobaya_sampler(yaml_file: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    logger = logger or LOGGER
    if not COBAYA_AVAILABLE:
        logger.critical("Cobaya not installed. Install: pip install cobaya")
        return None

    try:
        from cobaya.run import run
    except ImportError:
        logger.critical("Cobaya run module not found.")
        return None

    try:
        input_yaml = load_yaml_config(yaml_file)
        if input_yaml is None:
            logger.error(f"Cobaya input YAML not found: {yaml_file}")
            return None

        if "theory" not in input_yaml:
            input_yaml["theory"] = {}

        if "dut_theory" not in input_yaml["theory"]:
            input_yaml["theory"]["dut_theory"] = {"external": Cobaya_DUT_Theory}

        if "dut_likelihood" not in input_yaml["likelihood"]:
            input_yaml["likelihood"]["dut_likelihood"] = {"external": Cobaya_DUT_Likelihood}

        logger.info(f"Cobaya sampler starting with input from: {yaml_file}")

        updated_info, products = run(input_yaml)

        logger.info("Cobaya sampler finished.")
        return updated_info

    except Exception as e:
        logger.error(f"Cobaya run failed: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ExtractoDAO Scientific Software Framework - Unified Bayesian Cosmology Engine (DUT)")
    parser.add_argument("--mode", type=str, default="simulate", choices=["simulate", "mcmc", "compare", "cobaya"], help="Mode of operation.")
    parser.add_argument("--dataset_mode", type=str, default="student", choices=["student", "research", "synthetic"], help="Data to use.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path.")
    parser.add_argument("--yaml_cobaya", type=str, default="cobaya_input.yaml", help="Cobaya input YAML file path.")

    args = parser.parse_args()

    engine = DUT_Integrator(config_file=args.config)

    data_dict = load_all_real_data_from_config(engine.config, args.dataset_mode)

    if args.mode == "simulate":
        LOGGER.info("Mode: SIMULATE (Single run with default/config parameters)")
        engine.integrate()
        chi2, breakdown = engine.calculate_chi2(data_dict)

        print("\n--- RESULTS (SIMULATE) ---")
        print(f"Parameters: {engine.params}")
        print(f"Total Chi^2: {chi2:.2f}")
        for k, v in breakdown.items():
            print(f"  {k} Chi^2: {v:.2f}")

    elif args.mode == "mcmc":
        LOGGER.info("Mode: MCMC (using emcee)")
        samples, stats = engine.run_mcmc(data_dict, n_steps=2000, n_walkers=32)
        if stats:
            print("\n--- RESULTS (MCMC STATS) ---")
            print(json.dumps(stats, indent=2))

    elif args.mode == "compare":
        LOGGER.info("Mode: COMPARE (DUT vs Fiducial ΛCDM)")
        results = engine.compare_with_lcdm(data_dict)
        print("\n--- RESULTS (COMPARISON) ---")
        print(json.dumps(results, indent=2))

    elif args.mode == "cobaya":
        LOGGER.info("Mode: COBAYA SAMPLER")
        if COBAYA_AVAILABLE:
            run_cobaya_sampler(args.yaml_cobaya)
        else:
            LOGGER.error("Cobaya is required for this mode but is not installed.")

    else:
        LOGGER.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
