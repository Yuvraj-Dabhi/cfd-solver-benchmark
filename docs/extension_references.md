# Extension Directions — Key References

Cross-referenced with implemented modules. Each entry links the citation
to the specific code it informs.

---

## 1. Neural Operators → `neural_operator_surrogate.py`

| Citation | Relevance |
|----------|-----------|
| Wang et al. (2025), *"Machine-learning-based simulation of turbulent flows over periodic hills using HUFNO"*, Phys. Fluids — arXiv:2504.13126 | Direct basis for `HUFNO` class; benchmarked on periodic hill (our DNS reference case) |
| Li et al. (2021), *"Fourier Neural Operator for Parametric PDEs"*, ICLR 2021 | Foundational FNO architecture → `FourierLayer`, `FNO2d` |
| Duruisseaux et al. (2026), *"Fourier Neural Operators Explained: A Practical Perspective"*, arXiv:2512.01421 | Implementation guidance for spectral convolution and resolution invariance |

---

## 2. Graph Neural Networks → `mesh_graphnet.py`, `physics_guided_gnn.py`

| Citation | Relevance |
|----------|-----------|
| Pelissier et al. (2024), *"GNNs for Mesh Generation and Adaptation in Structural and Fluid Mechanics"*, Mathematics 12, 2933 | Mesh-to-graph conversion and GNN architecture patterns |
| NVIDIA Modulus X-MeshGraphNet, arXiv:2411.17164v2 (2024) | MeshGraphNet encode–process–decode architecture → `MeshGraphNet` class |

---

## 3. PINN-Data Assimilation → `dns_data_assimilation.py`, `pinn_corrector.py`

| Citation | Relevance |
|----------|-----------|
| Habibi et al. (2024), *"Turbulence model augmented PINNs for mean flow reconstruction"*, arXiv:2306.01065v2 (JFM) | PINN loss design with momentum-integral constraints; informs `DNSDataExtractor` and RANS-DNS fusion |

---

## 4. Generalized FIML → `gep_explicit_closure.py`, `fiml_pipeline.py`

| Citation | Relevance |
|----------|-----------|
| Srivastava et al. (2024), *"On generalizably improving RANS predictions of flow separation"*, AIAA SciTech 2024, p. 2520 | Feature space design as dominant factor for cross-case generalization → `evaluate_transfer()` |
| Frontiers in Physics (2024), *"Data-driven RANS closures for improving mean field calculation of separated flows"*, doi:10.3389/fphy.2024.1347657 | GEP for explicit closure models → `GEPClosureDiscovery`, `ClosureFormula.to_su2_cpp()` |

---

## 5. LLM-CFD → `llm_benchmark_assistant.py`

| Citation | Relevance |
|----------|-----------|
| Pandey et al. (2025), *"OpenFOAMGPT: A RAG-Augmented LLM Agent"*, Phys. Fluids 37, 035120 | RAG architecture for CFD config generation → `ConfigGenerator`, `DiagnosticEngine` |
| Chen et al. (2025), *"MetaOpenFOAM 2.0"*, arXiv:2502.00498 | LLM-driven OpenFOAM automation → `CrossSolverAligner` |

---

## 6. Adjoint Optimization → `fiml_correction.py`, `solver_coupling.py`

| Citation | Relevance |
|----------|-----------|
| Blühdorn et al. (2025), *"Hybrid parallel discrete adjoints in SU2"*, Computers & Fluids 289, 106528 | SU2 discrete adjoint integration for FIML β-field optimization |
| Adjoint + Diffusion Models (2025), arXiv:2507.23443 | Future direction: generative model for β-field initialization |

---

## 7. UQ / Probabilistic → `deep_ensemble.py`, `gep_explicit_closure.py`

| Citation | Relevance |
|----------|-----------|
| Levine & McKeon (2023), *"A probabilistic data-driven closure model for RANS"*, arXiv | Probabilistic closure framework; connects to deep ensemble UQ |
| Bin et al. (2024), *"Constrained recalibration of two-equation RANS models"*, TAML 14, 100503 | Physically constrained ML for SST → `constrained_recalibration()` in GEP module |
