"""
ML Augmentation Module for CFD Turbulence Model Correction
============================================================

Physics-informed machine learning modules for augmenting RANS turbulence
closures with data-driven corrections.

Core Modules (Active):
  - tbnn_closure: Tensor-Basis Neural Network (Pope 1975 / Ling 2016)
  - fiml_pipeline: Field Inversion and Machine Learning pipeline
  - fiml_correction: FIML eddy-viscosity β-correction (wall hump focused)
  - model: Turbulence correction architectures (MLP, TBNN, PINN)
  - feature_extraction: Galilean-invariant tensor features
  - dataset: Multi-source CFD dataset builder
  - mcconkey_dataset_loader: McConkey et al. (2021) curated turbulence data
  - evaluate: Prediction evaluation and realizability checks
  - drl_flow_control: DRL environments for 2D/3D active flow control + MARL (Montalà)

Graph Neural Network Modules:
  - mesh_graph_utils: SU2 mesh → PyG graph converter
  - mesh_graphnet: MeshGraphNet encoder-processor-decoder (Pfaff 2021)
  - physics_guided_gnn: PG-GNN with RANS residual losses
  - gnn_fiml_pipeline: GNN-based FIML β-correction pipeline

Data Assimilation Modules:
  - pinn_data_assimilation: PINN-DA with 2D RANS + SA closure (Habibi 2024)

Generative Surrogate Modules:
  - diffusion_flow_surrogate: FoilDiff-style DDIM diffusion for Cp/Cf fields with native UQ

ROM Closure Modules:
  - pod_transformer_closure: Transformer encoder closure for POD energy recovery (Eiximeno 2025)

Transformer Surrogate Modules:
  - transolver_surrogate: Transolver slice-attention + AB-UPT multi-branch (ICML 2024 / TMLR 2025)

Multi-Fidelity Modules:
  - multifidelity_framework: Hierarchical learning with cINN, residual correction, and co-kriging

Generalization Study Modules:
  - cross_case_generalization: LOO cross-case generalization study (Srivastava 2024 protocol)
  - loo_extrapolation_study: Systematic LOO extrapolation experiment (6 flows × 4 architectures + UQ)

LLM Automation Modules:
  - llm_benchmark_assistant: LLM-powered CFD analysis and reporting

Constrained Recalibration Modules:
  - constrained_rans_recalibration: Physics-penalty SST recalibration (Bin 2024)

Foundation Model Alignment Modules:
  - foundation_model_alignment: CFD foundation model alignment study
  - physicsnemo_domino_integration: NVIDIA DoMINO Neural Operator mock integration

Operator Learning Modules:
  - neural_operator_surrogate: FNO/HUFNO resolution-independent surrogate (Li 2021)
  - deeponet_surrogate: Branch-trunk Deep Operator Network (Lu 2021) for SWBLI/transonic

GNN Mesh Adaptation Modules:
  - gnn_mesh_adaptation: Adaptnet-style GNN for anisotropic Hessian metric prediction

DRL/MARL Calibration Modules:
  - ddpg_rans_calibration: DDPG actor-critic for autonomous RANS coefficient optimization
                           + SciMARL multi-agent dynamic wall models (Bae 2022)

Advanced Uncertainty Quantification Modules:
  - sobol_uq_bayesian: Sobol sensitivity + Bayesian Model Averaging
  - bayesian_pce_uq: PCE surrogate-accelerated Bayesian inversion with eigenspace perturbation
  - calibrated_stochastic_closures: Coverage-calibrated BNN/Ensemble/Diffusion + space-dependent aggregation + extended RSS

ML-Turbulence Benchmark Suite Modules:
  - benchmark_case_registry: McConkey curated dataset case alignment and export
  - benchmark_targets: Per-case target definitions and baseline error tables
  - curated_benchmark_evaluator: RMSE/MAE/physics-constraint evaluator + BenchmarkMetricsContract API

Complex Geometry Benchmark Modules:
  - crm_hl_benchmark: CRM-HL/HLPW-5 high-lift benchmark with warm-start + WMLES

GPU/CUDA Acceleration Modules:
  - gpu_cuda_accelerator: CUDA config, NvBLAS, hybrid GPU/CPU workflow, memory management

Operator-Learning & Temporal Surrogate Modules:
  - operator_temporal_case_studies: DeepONet vs FNO comparison, ConvLSTM temporal surrogate, design screening

Symbolic Closure Benchmark Modules:
  - symbolic_closure_benchmark: Unit-constrained GEP, cross-flow evaluation, interpretability analysis

Design & Control Workflow Modules:
  - design_control_uq_workflows: Multi-fidelity design opt + DRL flow control + UQ wrapping

Deprecated (see legacy/README.md):
  - legacy/naca_surrogate: Black-box MLP coefficient surrogate (superseded)
  - legacy/tier1_mlp_experiment: Tier-1 analytical aero model (superseded)
  - legacy/train_mlp: sklearn MLP training utilities (superseded)
  - legacy/run_naca_mlp: MLP entry point (superseded)
"""
