# ML/AI Pipeline (74 Modules)

Physics-informed machine learning augmentation for RANS turbulence model improvement, spanning closures, surrogates, neural operators, DRL flow control, and deployment.

---

## Core Physics-Informed Modules

| Module | Purpose |
|--------|---------|
| `tbnn_closure.py` | Tensor-Basis Neural Network (Pope 1975 bases), Lumley triangle realizability |
| `tbnn_dns_pipeline.py` | TBNN transfer learning: periodic hill DNS → wall hump |
| `fiml_su2_adjoint.py` | Field Inversion via SU2 discrete adjoint + L-BFGS-B optimization |
| `fiml_nn_embedding.py` | Neural network weight embedding for runtime β(x) evaluation |
| `fiml_pipeline.py` | Multi-case FIML pipeline with leave-one-out cross-validation |
| `fiml_correction.py` | FIML β-correction on wall hump (22.1% Cf RMSE reduction) |
| `deep_ensemble.py` | Deep ensemble (N=5) for epistemic uncertainty quantification |
| `bayesian_dnn_closure.py` | Variational BNN via MC-Dropout serving model-form stochastic predictions |
| `stochastic_evaluator.py` | Evaluator quantifying empirical data coverage within 95% Credible Intervals |
| `run_drl_wall_hump.py` | DRL Active Flow Control evaluating closed-loop PPO drag reduction vs simple open-loop blowing |
| `distribution_surrogate.py` | Multi-output MLP: (AoA, Re, Mach, H, dCp/dx, Reθ) → 80-pt Cp + 80-pt Cf |
| `pinn_boundary_layer.py` | PINN Cf correction with von Karman momentum-integral physics loss |
| `pinn_data_assimilation.py` | PINN-DA with 2D RANS + SA closure (Habibi 2024) |
| `neural_operator_surrogate.py` | FNO/HUFNO for resolution-independent CFD prediction |
| `gep_explicit_closure.py` | GEP symbolic regression: discovers interpretable RANS closures |
| `symbolic_closure_benchmark.py` | Unit-constrained GEP, cross-flow eval vs SA/SST/TBNN/FIML, interpretability |
| `design_control_uq_workflows.py` | Multi-fidelity design opt + DRL flow-control benchmark + UQ wrapping |
| `spatial_blending.py` | Multi-agent spatial blending (separation/reattachment/attached) |
| `curated_benchmark_evaluator.py` | Metrics contract API (RMSE/MAE/realizability/Cf separation) |
| `benchmark_case_registry.py` | McConkey curated dataset case alignment + export |
| `benchmark_targets.py` | Per-case target definitions + 9-model baseline error table |
| `separation_correction.py` | Non-intrusive ML separation metric correction |
| `ml_cf_correction.py` | ML Cf correction with deep ensemble |
| `diffusion_flow_surrogate.py` | FoilDiff-style DDIM diffusion model for Cp/Cf with native UQ |
| `transolver_surrogate.py` | Transolver slice-attention + AB-UPT multi-branch transformer |
| `multifidelity_framework.py` | cINN + residual correction + co-kriging multi-fidelity |
| `cross_case_generalization.py` | LOO cross-case study (Srivastava 2024 protocol) |
| `loo_extrapolation_study.py` | Systematic LOO extrapolation (6 flows × 4 architectures + UQ) |
| `calibrated_stochastic_closures.py` | Coverage-calibrated BNN/Ensemble/Diffusion + space-dependent aggregation |
| `operator_temporal_case_studies.py` | DeepONet vs FNO comparison + ConvLSTM temporal surrogate + design screening |
| `ml_validation_reporter.py` | R², RMSE, MAPE, overfitting detection, k-fold CV, model comparison |
| `surrogate_model.py` | GP/MLP surrogate for CL/CD (R²=0.9973) |
| `feature_extraction.py` | Galilean-invariant features (Pope invariants, Q-criterion, APG indicator) |
| `deeponet_surrogate.py` | Branch-trunk DeepONet operator learning (Lu 2021) for SWBLI/transonic |
| `bayesian_pce_uq.py` | PCE surrogate + MCMC Bayesian inversion + eigenspace perturbation |
| `ddpg_rans_calibration.py` | DDPG actor-critic for autonomous SST coefficient optimization |
| `drl_flow_control.py` | DRL flow control agent (PPO/SAC) |
| `constrained_rans_recalibration.py` | Physics-penalty SST recalibration (Bin 2024) |
| `pod_transformer_closure.py` | Transformer ROM closure with easy-attention (Eiximeno 2025) |
| `foundation_model_alignment.py` | CFD foundation model alignment study |
| `physicsnemo_domino_integration.py` | NVIDIA DoMINO operator + MoE (mock) |
| `hypersonic_extrapolation.py` | OOD extrapolation benchmark (M=5, heated jet) |
| `physics_informed_benchmark.py` | Invariance/realizability Pareto benchmark |

## Graph Neural Network Modules

| Module | Purpose |
|--------|---------|
| `mesh_graphnet.py` | MeshGraphNet encoder-processor-decoder (Pfaff et al. 2021) |
| `mesh_graph_utils.py` | SU2 mesh → PyG graph converter |
| `physics_guided_gnn.py` | PG-GNN with RANS residual physics losses |
| `gnn_fiml_pipeline.py` | GNN-based FIML β-correction pipeline |
| `gnn_mesh_adaptation.py` | Adaptnet-style GNN for anisotropic Hessian metric prediction |

## Deployment & Automation

| Module | Purpose |
|--------|---------|
| `native_inference.py` | C/NumPy native model inference (no PyTorch dependency) |
| `solver_coupling.py` | Online RANS-ML coupling for closed-loop correction |
| `llm_benchmark_assistant.py` | LLM-powered CFD analysis, diagnostics, and reporting |
| `streaming_surrogate.py` | Online learning surrogate with streaming data |

## DRL / Multi-Agent Calibration

| Module | Purpose |
|--------|---------|
| `ddpg_rans_calibration.py` | DDPG autonomous RANS coefficient calibration + SciMARL multi-agent wall models |

## Complex Geometry Benchmarks

| Module | Purpose |
|--------|---------|
| `crm_hl_benchmark.py` | CRM-HL/HLPW-5 benchmark with warm-start α-sweep, drag decomposition, WMLES config |

## GPU/CUDA Acceleration

| Module | Purpose |
|--------|---------|
| `gpu_cuda_accelerator.py` | CUDA config generator, NvBLAS, hybrid GPU/CPU workflow, memory estimator |

---

## Key ML Results

> [!IMPORTANT]
> **Maturity note:** Of the 74 ML modules, only 3 have been exercised end-to-end
> with quantified train/test results (Tier 1 MLP surrogate, FIML β-correction,
> multi-agent spatial blending). The remaining modules define tested API contracts
> and pass unit tests, but have **not** been trained on real CFD data with full
> before/after validation.

| Model | Metric | Value | Evidence Level |
|-------|--------|-------|----------------|
| CL/CD Surrogate (MLP) | R² | 0.9973 | End-to-end experiment |
| CL/CD Surrogate (MLP) | MAPE | 3.48% | End-to-end experiment |
| PINN BL Correction | Cf RMSE improvement | 30-80% (vs uncorrected RANS) | Synthetic unit test |
| FIML β-Correction | Cf RMSE reduction | **22.1%** (post-hoc, wall hump) | End-to-end experiment |
| Distribution Surrogate | Outputs | 160 surface points per query | API contract only |
| Multi-Agent Blended | R² improvement | **2.9× vs global TBNN** | Synthetic features |
| GEP Symbolic Closure | Output | Interpretable algebraic g^(n)(λ) + SU2 C++ export | API contract only |
| Diffusion Surrogate (DDIM) | UQ Method | Full Cp/Cf distribution sampling via denoising | API contract only |
| Bayesian DNN | UQ Method | Epistemic structural limits for RANS model closures | API contract only |
| Transolver Transformer | Architecture | Slice-attention + divergence-free projection | API contract only |
| Multi-Fidelity (cINN) | Method | P(DNS \| RANS) via invertible net + co-kriging | API contract only |
| Cross-Case LOO | Generalization | 5-case LOO rankings with transferability analysis | Synthetic features |
| Curated Benchmark Suite | McConkey | 5-case registry, 8 tasks, metrics contract API, JSON/CSV/MD export | API contract only |
| Constrained Recalibration | Bin (2024) | Physics-penalty SST coefficient optimization | API contract only |
| POD-Transformer Closure | Eiximeno (2025) | Easy-attention Transformer ROM closure | API contract only |
| Foundation Model Alignment | Transolver | Pre-trained vs custom domain model comparison | API contract only |
| PhysicsNeMo DoMINO | NVIDIA NIM | DoMINO operator fine-tuning + MoE routing | API contract only |
| Hypersonic OOD | Extrapolation | ML correction failure modes on M=5 SWBLI | API contract only |
| Physics Constraint Benchmark | Pareto | Galilean invariance × realizability × accuracy | API contract only |
| DeepONet | Architecture | Branch-trunk operator learning (Lu 2021) | API contract only |
| DDPG Calibration | Method | Autonomous SST coefficient optimization via actor-critic RL | API contract only |
| SciMARL | Architecture | Multi-agent per-cell eddy viscosity adaptation | API contract only |
| Bayesian PCE | UQ Method | Polynomial chaos + MCMC inversion + eigenspace perturbation | API contract only |
| GNN Mesh Adapt | Architecture | Hessian metric prediction for anisotropic h-adaptation | API contract only |

---

## Key Literature (2024–2026)

| Category | Key References |
|----------|---------------|
| **Generative / Diffusion** | CoNFiLD (*Nature Comms*, 2024), FoilDiff (arXiv 2510.04325), DDPM turbulence (Stanford CTR 2024) |
| **Transformer Surrogates** | Transolver (ICML 2024 / NVIDIA PhysicsNeMo), AB-UPT (TMLR 2025, 160M-cell DrivAerML) |
| **DRL Flow Control** | Font et al. (*Nature Comms* 16, 2025) — turbulent separation bubble; Montalà (arXiv 2025) — 3D wings |
| **Multi-Fidelity** | Geneva & Zabaras conditional invertible NN; DrivAerML RANS/HRLES hierarchy |
| **RANS-ML Generalization** | Srivastava (AIAA SciTech 2024), Bin et al. (*TAML* 14, 2024) |
| **Curated Repository** | [`awesome-machine-learning-fluid-mechanics`](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics) |
| **ROM Closures** | Eiximeno et al. (*JFM* Oct 2025) — Transformer closure for POD energy recovery |
| **Constrained Recal** | Bin et al. (*TAML* 14, 2024) — Physics-penalty SST coefficient optimization |
| **Foundation Models** | DoMINO (NVIDIA PhysicsNeMo 25.11), Aurora (Microsoft 2024), Poseidon (ETH 2024) |
