# Literature Review: CFD Solver Benchmark for Flow Separation Prediction

**Author:** Yuvraj  
**Affiliation:** MSc Space Engineering, University of Bremen  
**Date:** February 2026  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Foundational Works and Classical Benchmarks](#2-foundational-works-and-classical-benchmarks)
3. [NASA Benchmark Programs and Validation Databases](#3-nasa-benchmark-programs-and-validation-databases)
4. [Turbulence Modeling Approaches for Separated Flows](#4-turbulence-modeling-approaches-for-separated-flows)
5. [Machine Learning and Data-Driven Approaches](#5-machine-learning-and-data-driven-approaches)
6. [Solver Comparison Studies (OpenFOAM, SU2, and Others)](#6-solver-comparison-studies)
7. [European and German Research Initiatives (DLR, ZARM)](#7-european-and-german-research-initiatives)
8. [Research Gaps and Open Challenges](#8-research-gaps-and-open-challenges)
9. [References](#9-references)

---

## 1. Introduction

Flow separation is one of the most challenging phenomena in computational fluid dynamics (CFD). It occurs when the boundary layer detaches from a surface due to adverse pressure gradients, geometric discontinuities, or shock-boundary layer interactions. Accurate prediction of separation onset, extent, and reattachment is critical for aerospace applications—including airfoil stall, engine inlet performance, base flows behind spacecraft, and nozzle flow behavior.

Despite decades of research, no single turbulence model reliably predicts separated flows across all regimes. This review surveys the state-of-the-art in CFD benchmarking for flow separation prediction, covering foundational experiments, modern benchmark initiatives, turbulence modeling strategies, machine learning augmentation, solver comparisons, and identified research gaps.

---

## 2. Foundational Works and Classical Benchmarks

### 2.1 Backward-Facing Step (Driver & Seegmiller, 1985)

| **Detail** | **Information** |
|---|---|
| **Authors** | D.M. Driver, H.L. Seegmiller |
| **Publication** | "Features of a Reattaching Turbulent Shear Layer in Divergent Channel Flow," AIAA Journal, Vol. 23, No. 2, 1985 |
| **Source** | NASA Ames Research Center |
| **Summary** | Classic benchmark for turbulent flow separation and reattachment. Provides detailed measurements of mean velocities, Reynolds stresses, and triple products in a diverging channel with a rearward-facing step. Reattachment length x/H ≈ 6.25 at Re = 36,000. Widely used to validate k-ε, SST k-ω, Spalart-Allmaras, and algebraic-stress models. |

### 2.2 Bachalo-Johnson Transonic Bump (1986)

| **Detail** | **Information** |
|---|---|
| **Authors** | W.D. Bachalo, D.A. Johnson |
| **Publication** | "Transonic, Turbulent Boundary-Layer Separation Generated on an Axisymmetric Flow Model," AIAA Journal, Vol. 24, No. 3, pp. 437–443, 1986 (also AIAA Paper 79-1479) |
| **Source** | NASA Ames Research Center |
| **Summary** | Axisymmetric circular-arc bump in transonic flow (M = 0.875, Re_c ≈ 2.7 × 10⁶). Shock-induced separation with subsequent reattachment. Dataset includes mean velocity, turbulence intensity, Reynolds shear stress profiles, and surface Cp. Revisited by Sandia National Laboratories ("Sandia ATB") with modern diagnostics (PIV, fast PSP, oil-film interferometry) at lower Re ≈ 10⁶ for LES/DNS tractability. |

### 2.3 Periodic Hill Flow (Mellen, Fröhlich, Rodi et al., 2000–2009)

| **Detail** | **Information** |
|---|---|
| **Authors** | C.P. Mellen, J. Fröhlich, W. Rodi (2000); L.E. Temmerman, M.A. Leschziner (2005); M. Breuer, N. Peller, Ch. Rapp, M. Manhart (2009) |
| **Publications** | (1) "LES of Flow over Periodic Hills," ERCOFTAC Bulletin, 2000. (2) Fröhlich et al., "Highly Resolved LES of Separated Flow in a Channel with Streamwise Periodic Constrictions," J. Fluid Mech., 2005. (3) Breuer et al., "Flow over periodic hills—Numerical and experimental study over a wide range of Reynolds numbers," Computers & Fluids, 2009 |
| **Source** | University of Erlangen-Nuremberg / TU Dresden / Karlsruhe Institute of Technology |
| **Summary** | Canonical benchmark for separated flows featuring separation from a curved surface, recirculation, shear layers, and reattachment. Standard Re = 10,595 (also 5,600 and 37,000). Provides highly resolved DNS/LES reference data crucial for validating hybrid RANS-LES approaches and advanced turbulence closures. |

### 2.4 NACA 0012 Airfoil at Stall Conditions

| **Detail** | **Information** |
|---|---|
| **Authors** | Various (Gregory & O'Reilly, 1970; Ladson, 1988; numerous subsequent studies) |
| **Source** | NASA Langley Research Center / Multiple institutions |
| **Summary** | The NACA 0012 symmetric airfoil is the most widely used airfoil for CFD validation. At high angles of attack, trailing-edge and leading-edge stall provide separation benchmark data. Experimental databases include Cp distributions, lift/drag polars, and boundary-layer profiles across a range of Reynolds numbers. |

### 2.5 NASA GA(W)-1 Airfoil Separation

| **Detail** | **Information** |
|---|---|
| **Authors** | Various NASA researchers |
| **Source** | NASA Technical Reports |
| **Summary** | Experimental studies of separated flow over the GA(W)-1 airfoil characterizing flow profiles, skin-friction distributions, and boundary-layer integral properties at various angles of attack past separation onset. |

---

## 3. NASA Benchmark Programs and Validation Databases

### 3.1 NASA Turbulence Modeling Resource (TMR / TMBWG)

| **Detail** | **Information** |
|---|---|
| **Maintainer** | NASA Langley Research Center / Turbulence Model Benchmarking Working Group (TMBWG) |
| **URL** | https://tmbwg.github.io/turbmodels (migrated from https://turbmodels.larc.nasa.gov/) |
| **Summary** | Comprehensive online database of verification and validation test cases for turbulence models. Includes flat plate, bump-in-channel, airfoil, backward-facing step, and 2D/3D separated flow cases. Provides grids, boundary conditions, and reference solutions for systematic model comparison. Essential resource for any CFD benchmarking study. The site has migrated to GitHub (TMBWG) and is actively updated with new model variants including SST-Vm, SST-2003m, SA-neg, SA-QCR2000, and updated validation cases. The Schulein SWBLI case is most commonly validated against SST-2003m (Menter et al. 2003 modified), not the original 1994 SST. |

### 3.2 NASA CFDVAL2004 Workshop – Wall-Mounted Hump (Greenblatt et al.)

| **Detail** | **Information** |
|---|---|
| **Authors** | D. Greenblatt, K.B. Paschal, C.-S. Yao, J. Harris, N.W. Schaeffler, A.E. Washburn |
| **Publication** | "Experimental Investigation of Separation Control Part 1: Baseline and Steady Suction," AIAA Journal, Vol. 44, No. 12, 2006 |
| **Source** | NASA Langley Research Center |
| **Summary** | Wall-mounted Glauert-Goldschmied type hump model at Re ≈ 936,000, M = 0.1. Comprehensive dataset: static/dynamic surface pressures, 2D/3D PIV in separated/reattachment regions, oil-film interferometry for skin friction. Baseline (no control) and active control (steady suction, oscillatory blowing) configurations. Became "Case 3" of the CFDVAL2004 Workshop. One of the most extensively used 2D separation benchmarks for turbulence model validation. |

### 3.3 BeVERLI Hill (Benchmark Validation Experiments for RANS and LES)

| **Detail** | **Information** |
|---|---|
| **Authors** | C.J. Lowe, R.L. Simpson (Virginia Tech); NASA Langley collaborators |
| **Publication** | Multiple AIAA papers (2019–2024), part of AIAA CFD Turbulence Model Validation Challenge |
| **Source** | Virginia Tech / NASA Langley |
| **Summary** | 3D smooth-body geometry designed specifically for studying turbulent separation at subsonic Re = 250k–650k. Experimental data includes oil flow visualization, surface pressures, skin friction (oil-film interferometry), and velocity fields (PIV, LDV). Used in blind CFD challenges to rigorously benchmark RANS and LES turbulence models on 3D separation. |

### 3.4 NASA Juncture Flow (JF) Experiment

| **Detail** | **Information** |
|---|---|
| **Authors** | C.L. Rumsey, J.-R. Carlson, et al. (NASA Langley) |
| **Publication** | Multiple NASA Technical Memoranda and AIAA papers (2017–2023) |
| **Source** | NASA Langley Research Center |
| **Summary** | Dedicated CFD validation experiment for wing-fuselage junction flows experiencing separation. Provides high-quality flowfield data (LDV, PIV, surface pressures) focused on the corner/junction vortex and separation bubble. Crucial for validating RANS models in practical aircraft configurations. |

### 3.5 NASA Axisymmetric Afterbody

| **Detail** | **Information** |
|---|---|
| **Authors** | NASA researchers |
| **Source** | NASA Technical Reports |
| **Summary** | Designed for detailed measurements of smooth, adverse-pressure-gradient-induced separation. Allows study of flow states from fully attached to incipient and small-scale separation. Includes steady pressure and 2D/Stereo-PIV measurements. |

### 3.6 High-Lift Common Research Model (CRM-HL)

| **Detail** | **Information** |
|---|---|
| **Authors** | NASA / International collaborative teams |
| **Source** | Open-source geometry (NASA) |
| **Summary** | Developed to improve CFD modeling of turbulent separated flows relevant to low-speed, high-lift aircraft performance. Open-source geometry definitions foster international collaboration. Focus on leading-edge slat, trailing-edge flap, and wing-body juncture flows. |

### 3.7 CFD Vision 2030 Study

| **Detail** | **Information** |
|---|---|
| **Authors** | J. Slotnick, A. Khodadoust, J. Alonso, D. Darmofal, W. Gropp, E. Lurie, D. Mavriplis |
| **Publication** | "CFD Vision 2030 Study: A Path to Revolutionary Computational Aerosciences," NASA/CR-2014-218178, 2014 |
| **Source** | NASA |
| **Summary** | Landmark roadmap document identifying the critical need for improved CFD capabilities for separated flows. Highlights persistent failures of RANS in predicting viscous flow separation over streamlined bodies. Recommends advancing hybrid RANS/LES, wall-modeled LES, and physics-based modeling as priority research directions. Directly motivates the current generation of separation benchmarks. |

### 3.8 Three-Dimensional Tapered Hump ("Speed Bump")

| **Detail** | **Information** |
|---|---|
| **Authors** | Boeing / University of Washington / NASA collaborators |
| **Source** | AIAA / APS publications |
| **Summary** | Particularly challenging 3D geometry for RANS models. Strongly separated flow. Studies show RANS captures Cp magnitude but fails to reproduce Reynolds-number insensitivity and inflection points near reattachment. Highlights fundamental limitations of RANS closures for 3D separation. |

### 3.9 Flow Separation in Rocket Nozzles

| **Detail** | **Information** |
|---|---|
| **Authors** | Various NASA researchers (historical and recent) |
| **Source** | NASA Technical Reports |
| **Summary** | Early NASA research on flow separation in liquid-propellant rocket nozzles, summarizing theoretical and empirical prediction methods. Comparison of numerical results with hot-firing test data. Relevant to overexpanded nozzle flows, side loads, and aerospace propulsion systems. |

---

## 4. Turbulence Modeling Approaches for Separated Flows

### 4.1 RANS Models

| **Model** | **Key References** | **Summary** |
|---|---|---|
| **Spalart-Allmaras (SA)** | Spalart & Allmaras, AIAA Paper 92-0439, 1992 | One-equation model. Reasonable for mildly separated flows; tends to under-predict separation extent in strongly adverse pressure gradients. Widely used in aerospace industry. |
| **k-ω SST** | Menter, "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications," AIAA Journal, Vol. 32, No. 8, 1994 | Blends k-ω near wall with k-ε in freestream. Shear-stress limiter improves separation prediction. De facto standard for aerospace RANS. Still over-predicts reattachment in many cases. |
| **k-ε Realizable** | Shih et al., "A New k-ε Eddy Viscosity Model for High Reynolds Number Turbulent Flows," Computers & Fluids, 1995 | Improved realizability constraints. Better for flows with strong streamline curvature and separation compared to standard k-ε. |
| **Reynolds Stress Models (RSM)** | Launder, Reece, Rodi (1975); Speziale, Sarkar, Gatski (SSG, 1991) | Full Reynolds stress transport. Theoretically superior for anisotropic separated flows. Higher computational cost and stability challenges. |

### 4.2 Hybrid RANS-LES Methods

| **Method** | **Key References** | **Summary** |
|---|---|---|
| **Detached Eddy Simulation (DES)** | Spalart et al., "Comments on the Feasibility of LES for Wings, and on a Hybrid RANS/LES Approach," 1st AFOSR Int. Conf. on DNS/LES, 1997 | Switches from RANS near walls to LES in separated regions. Improved unsteady separation prediction at moderate cost. |
| **Delayed DES (DDES)** | Spalart et al., AIAA Paper 2006-3298, 2006 | Fixes premature grid-induced separation in DES by delaying the switch to LES mode. More robust for complex geometries. |
| **Improved DDES (IDDES)** | Shur et al., Int. J. Heat and Fluid Flow, 2008 | Combines DDES with wall-modeled LES capability. Enables resolved turbulent content in attached boundary layers when grid allows. |
| **Stress-Blended Eddy Simulation (SBES)** | Menter, "Stress-Blended Eddy Simulation (SBES)," ANSYS internal / 2018 | Proprietary hybrid method with explicit shielding and rapid transition from RANS to LES in separated regions. |

### 4.3 Wall-Modeled LES (WMLES)

| **Detail** | **Key References** | **Summary** |
|---|---|---|
| **Equilibrium wall models** | Kawai & Larsson, "Wall-modeling in large eddy simulation," Phys. Fluids, 2012 | Use simplified wall-layer equations to bypass resolution requirements near the wall. Enable LES of high-Re separated flows at practical cost. |
| **Non-equilibrium wall models** | Park & Moin, "An improved dynamic non-equilibrium wall-model," Phys. Fluids, 2014 | Account for pressure gradients and convection in the wall model. Critical for accurately capturing separation onset. |

### 4.4 Scale-Resolving Simulations (DNS/LES)

| **Detail** | **Key References** | **Summary** |
|---|---|---|
| **DNS** | Breuer et al. (2009), Krank et al. (2018) | Full resolution of all turbulent scales. Provides ground-truth reference data but limited to moderate Re. Periodic hill at Re = 5,600 is a common DNS target. |
| **LES** | Fröhlich et al. (2005), multiple subsequent studies | Resolves large energy-containing eddies; models subgrid-scale stresses. Standard subgrid models: Smagorinsky, Dynamic Smagorinsky, WALE. More accurate for separated flows than RANS but 100–1000× more expensive. |

---

## 5. Machine Learning and Data-Driven Approaches

### 5.1 ML-Augmented RANS Closures

| **Detail** | **Information** |
|---|---|
| **Authors** | J. Ling, A. Kurzawski, J. Templeton |
| **Publication** | "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance," J. Fluid Mech., Vol. 807, pp. 155–166, 2016 |
| **Summary** | Pioneering work using deep neural networks to predict Reynolds stress anisotropy from mean flow features while respecting Galilean invariance. Trained on DNS data of separated flows (e.g., duct, wavy wall). Demonstrated significant improvement over linear eddy-viscosity models. |

### 5.2 Data-Driven Turbulence Model Correction

| **Detail** | **Information** |
|---|---|
| **Authors** | E.J. Parish, K. Duraisamy |
| **Publication** | "A paradigm for data-driven predictive modeling using field inversion and machine learning," J. Comput. Phys., Vol. 305, pp. 758–774, 2016 |
| **Summary** | Field Inversion and Machine Learning (FIML) framework. Inverts high-fidelity data to identify spatially-varying correction fields for RANS equations, then trains ML models to generalize corrections to new flows. Applied to separated flows including airfoil stall and bump flows. |

### 5.3 Physics-Informed Neural Networks (PINNs) for Flow Separation

| **Detail** | **Information** |
|---|---|
| **Authors** | M. Raissi, P. Perdikaris, G.E. Karniadakis |
| **Publication** | "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs," J. Comput. Phys., Vol. 378, 2019 |
| **Summary** | Embeds governing equations (Navier-Stokes) as constraints in neural network training. Applied to reconstruct flow fields from sparse data. Promising for separation modeling but challenges remain for high-Re turbulent flows. |

### 5.4 Graph Neural Networks for Aerodynamic Predictions

| **Detail** | **Information** |
|---|---|
| **Authors** | T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. Battaglia (DeepMind, 2021); various subsequent works |
| **Summary** | Graph Convolutional Networks (GCNs) for efficient prediction of aerodynamic loads, capturing shock waves and flow separation. Offer 100–1000× speedups over traditional CFD. Active area of research with applications to airfoil flow prediction and optimization. |

### 5.5 ML-Enhanced Wall Models for LES

| **Detail** | **Information** |
|---|---|
| **Authors** | A. Lozano-Durán, H. Bae, M.P. Encinar |
| **Publication** | Various, Physics of Fluids, 2023–2024 |
| **Summary** | ML models trained to map local mean flow features to wall-shear-stress discrepancies in WMLES. Targets supersonic and transonic separated flows. Aims to improve skin-friction prediction in separation and reattachment zones. |

### 5.6 Hybrid Expert Models and Blending Approaches

| **Detail** | **Information** |
|---|---|
| **Authors** | Various (arXiv preprints, 2023–2025) |
| **Summary** | ML-assisted blending of multiple specialized turbulence closures ("expert models"). Each expert is optimized for a specific flow regime (attached, mildly separated, massively separated). Blending function trained to select/combine experts based on local flow features. Aims to improve generalizability across diverse geometries. |

### 5.7 FIML-Guided RANS Augmentation (Srivastava et al., 2024)

| **Detail** | **Information** |
|---|---|
| **Authors** | B. Srivastava, R. Hartfield, D. Kofke, et al. |
| **Publication** | "Augmenting RANS Turbulence Models Guided by Field Inversion and Machine Learning," NASA TM-20240012512, 2024 |
| **Source** | NASA Technical Reports Server (NTRS) |
| **Summary** | Extends the Parish & Duraisamy (2016) FIML paradigm by identifying a specific Galilean-invariant feature set (q1–q5) that achieves generalizable reattachment-location correction when used to train an MLP for eddy-viscosity β-correction. Key features: turbulence-to-mean-strain ratio (q1), wall-distance Reynolds number (q2), strain-rotation ratio (q3), pressure-gradient alignment (q4), and turbulent viscosity ratio (q5). Demonstrated on several separated-flow configurations including bump and channel flows. The q1–q5 feature set and β-correction framework directly inspire the ML augmentation pipeline used in this project (`fiml_correction.py`). Our work is framed as a student implementation of this FIML methodology using the NASA wall hump as the validation case. |

### 5.8 Curated Turbulence Dataset (McConkey et al., 2021)

| **Detail** | **Information** |
|---|---|
| **Authors** | R. McConkey, E. Yee, F. Lien |
| **Publication** | "A curated dataset for data-driven turbulence modelling," *Scientific Data*, Vol. 8, No. 1, 2021. DOI: 10.1038/s41597-021-01034-2 |
| **Data DOI** | 10.34740/kaggle/dsv/2637500 (Kaggle); metadata DOI: 10.6084/m9.figshare.15124857 |
| **Summary** | Open-access curated dataset comprising 895,640 spatial datapoints from 29 CFD cases across 4 RANS models (k-ε, k-ε-ϕt-f, k-ω, k-ω SST), each paired with DNS/LES reference labels. Geometries include periodic hills, square duct, parametric bumps, converging-diverging channel, and curved backward-facing step. The parametric bump cases are geometrically closest to the NASA wall hump and provide the physically most relevant training data for the ML correction model. This dataset replaces the need for generating thousands of synthetic NACA cases and provides more training data than could be generated in months of computation. Used as the primary external ML training data source in this project. |

### 5.9 DNN-Based Flow Prediction and Solver Coupling (Haghahenas, Hedayatpour, Groll, 2023)

| **Detail** | **Information** |
|---|---|
| **Authors** | A. Haghahenas, S. Hedayatpour, R. Groll |
| **Publication** | "Prediction of particle-laden pipe flows using deep neural network models," *Physics of Fluids*, Vol. 35, 083320, 2023 |
| **Affiliation** | University of Bremen |
| **Summary** | Demonstrates a deep neural network pipeline for predicting turbulent pipe flow fields, with emphasis on how DNN predictions are coupled back to the flow solver. Documents input feature selection, network architecture, and integration methodology. The approach is analogous to the neural network-based flow prediction implemented in this project. Of particular relevance for networking with Bremen-based researchers: this pipeline can be described as "a neural network flow prediction pipeline similar in spirit to [Groll's] pipe-flow paper, applied to the canonical wall hump separated-flow case." |

---

## 6. Solver Comparison Studies

### 6.1 OpenFOAM

| **Detail** | **Information** |
|---|---|
| **Type** | Open-source, finite-volume CFD framework |
| **Strengths** | Highly versatile; strong incompressible flow capability; large user/developer community; extensive turbulence model library (SA, k-ω SST, k-ε variants, RSM, LES/DES). Compressible solvers: rhoCentralFoam, sonicFoam, rhoSimpleFoam. |
| **Separation Benchmarks** | Widely used for backward-facing step, periodic hill, wall-mounted hump, airfoil stall. |
| **Limitations** | Compressible flow solvers less mature than incompressible ones. Limited built-in adjoint optimization (compared to SU2). Mesh quality sensitivity in separated regions. |

### 6.2 SU2

| **Detail** | **Information** |
|---|---|
| **Type** | Open-source, unstructured CFD solver (Stanford University) |
| **Strengths** | Designed for compressible/transonic aerodynamics. Built-in adjoint solvers for shape optimization. Strong for design and optimization tasks. SST turbulence model implementation. 2D bump-in-channel validation against NASA data. |
| **Separation Benchmarks** | Mexico rotor comparison showed better flow-field resolution than OpenFOAM. Validated on bump-in-channel (NASA benchmark). |
| **Limitations** | Smaller community than OpenFOAM. Fewer turbulence model options. Incompressible flow capabilities less developed. |

### 6.3 ANSYS Fluent / CFX

| **Detail** | **Information** |
|---|---|
| **Type** | Commercial, general-purpose CFD (ANSYS Inc.) |
| **Strengths** | Extensive turbulence model library, polyhedral meshing, SBES hybrid method, robust solvers. Industry standard. |
| **Separation Benchmarks** | Widely used for all major benchmarks. Publication bias toward ANSYS in industry-funded studies. |
| **Limitations** | Proprietary / expensive. Black-box implementations limit reproducibility. |

### 6.4 NASA CFL3D / FUN3D / OVERFLOW

| **Detail** | **Information** |
|---|---|
| **Type** | NASA in-house structured/unstructured solvers |
| **Strengths** | Reference implementations for NASA TMR benchmark cases. CFL3D used for massively-separated flow studies (2D hill, Ahmed body). FUN3D for unstructured grids. OVERFLOW for overlapping grids. |
| **Separation Benchmarks** | Primary solvers for NASA benchmark campaigns. Results serve as reference for other solver comparisons. |

### 6.5 Key Comparison Findings

- **OpenFOAM vs. SU2 (Mexico Rotor)**: SU2 showed better resolution of flow field and blade pressures under separated conditions; OpenFOAM was faster computationally. Definitive conclusions limited without grid independence.
- **OpenFOAM Compressible Solvers**: rhoCentralFoam vs. sonicFoam for nozzle separation—rhoCentralFoam provided better predictive performance for separation patterns.
- **General**: Solver accuracy for separation depends primarily on turbulence model choice, mesh quality, and numerical scheme rather than solver framework itself. All open-source solvers can achieve comparable accuracy with careful setup.

---

## 7. European and German Research Initiatives

### 7.1 DLR (German Aerospace Center)

| **Detail** | **Information** |
|---|---|
| **Institution** | Deutsches Zentrum für Luft- und Raumfahrt |
| **Research Areas** | Turbulent boundary layers with pressure gradients; transition prediction; hybrid RANS-LES methods (TAU code); high-lift configurations. |
| **Key Contributions** | Extensive experimental databases for turbulent boundary layers under adverse pressure gradients. Development of the TAU CFD code with advanced turbulence models. Participation in AIAA CFD validation challenges. |

### 7.2 ZARM – University of Bremen

| **Detail** | **Information** |
|---|---|
| **Institution** | Center of Applied Space Technology and Microgravity, University of Bremen |
| **Research Groups** | (1) Fluid Simulation and Modeling, (2) Multiphase Flows, (3) Thermo-fluid Dynamics |
| **Key Activities** | Transition to turbulence (wind turbines, cardiovascular systems); turbulent mixing; droplet breakup DNS; nonlinear adjoint optimization for shear flow turbulence; microgravity fluid dynamics (Drop Tower, ISS experiments); two-phase flows with free surfaces; spacecraft cryogenic fuel management; electric propulsion plasma flows. |
| **Tools Used** | FLOW-3D, Star CCM+, OpenFOAM, in-house HPC codes |
| **Relevance** | ZARM's fluid simulation group works on turbulence transition and DNS—directly relevant to separation prediction. Their microgravity experiments validate CFD models for spacecraft fuel management. HiWi opportunities likely in DNS/LES of transitional flows or CFD validation against Drop Tower experiments. |

### 7.3 ERCOFTAC (European Research Community on Flow, Turbulence and Combustion)

| **Detail** | **Information** |
|---|---|
| **Summary** | Maintains benchmark databases including the T3 series (flat-plate bypass transition), periodic hill, and various separated flow cases. Organizes workshops and comparison campaigns. ERCOFTAC T3A/T3B datasets are standard references for transition model validation. |

### 7.4 NATO STO (Science and Technology Organization)

| **Detail** | **Information** |
|---|---|
| **Summary** | Organized collaborative studies on predicting separated turbulent flows for military aerospace applications. Published technical reports evaluating RANS and hybrid RANS/LES methods against benchmark experiments including CRM-HL and tapered hump geometries. |

---

## 8. Research Gaps and Open Challenges

Based on the reviewed literature, the following research gaps have been identified:

### Gap 1: RANS Reliability for 3D Separation
> **Challenge:** RANS models (even SST k-ω) consistently fail to predict 3D separation onset and reattachment accurately. The tapered hump studies show RANS cannot capture Reynolds-number insensitivity of separation topology.  
> **Opportunity:** Systematic benchmarking of RANS variants across the full suite of 2D and 3D separation cases using consistent solver settings and grids. Quantifying model-form uncertainty in RANS predictions.

### Gap 2: Grid Sensitivity in Separated Regions
> **Challenge:** Most solver comparison studies lack rigorous grid convergence studies, making it impossible to distinguish modeling errors from discretization errors. The OpenFOAM vs. SU2 comparison explicitly noted this limitation.  
> **Opportunity:** Performing controlled grid-convergence studies on multiple benchmarks using the same turbulence model across multiple solvers.

### Gap 3: Transition-Separation Interaction
> **Challenge:** Many benchmark cases use tripped boundary layers, bypassing the complex interaction between boundary-layer transition and separation. In real aerospace applications (e.g., low-Re UAVs, turbomachinery), laminar separation bubbles and transition-driven separation are critical.  
> **Opportunity:** Benchmarking CFD solvers with transition models (γ-Re_θ, e^N) on untripped separation cases.

### Gap 4: ML Model Generalizability
> **Challenge:** Most ML-augmented turbulence models are trained and tested on similar flow configurations (e.g., periodic hill, bump). Generalization to unseen geometries and Reynolds numbers remains poor.  
> **Opportunity:** Developing multi-case training strategies and testing generalization systematically. Using physics-informed constraints to improve extrapolation.

### Gap 5: Wall-Modeled LES for Separated Flows
> **Challenge:** WMLES accuracy degrades significantly in separation and reattachment zones where the equilibrium wall-model assumptions break down. Non-equilibrium models improve results but at higher cost.  
> **Opportunity:** Benchmarking WMLES with various wall models on the NASA hump and periodic hill cases. Developing ML-enhanced wall models for non-equilibrium regions.

### Gap 6: Compressible/Transonic Separation Benchmarking
> **Challenge:** Far fewer benchmark studies exist for shock-induced and transonic separation compared to low-speed cases. The Bachalo-Johnson bump and Sandia ATB are exceptions but represent a narrow range of conditions.  
> **Opportunity:** Extending benchmark comparisons to transonic airfoils at incidence, scramjet inlets, and nozzle flows using modern high-fidelity data.

### Gap 7: Lack of Standardized Cross-Solver Comparison Framework
> **Challenge:** Existing solver comparisons use different grids, boundary conditions, convergence criteria, and post-processing methods, making fair comparison difficult.  
> **Opportunity:** Creating an open-source benchmarking framework that enforces consistent setup, automates solver execution, and generates standardized comparison metrics (Cp, Cf, separation/reattachment locations, velocity profiles).

### Gap 8: Spacecraft Wake and Base-Flow Separation
> **Challenge:** Flow separation behind spacecraft, re-entry capsules, and launch vehicle afterbodies is critical for thermal protection and aerodynamic stability, but benchmark data is sparse and mostly classified.  
> **Opportunity:** Leveraging ZARM's microgravity research and DLR's expertise to create open-access benchmark datasets for spacecraft wake flows.

---

## 9. References

### Foundational Experimental Benchmarks
1. Driver, D.M. & Seegmiller, H.L. (1985). "Features of a Reattaching Turbulent Shear Layer in Divergent Channel Flow." *AIAA Journal*, 23(2).
2. Bachalo, W.D. & Johnson, D.A. (1986). "Transonic, Turbulent Boundary-Layer Separation Generated on an Axisymmetric Flow Model." *AIAA Journal*, 24(3), 437–443.
3. Breuer, M., Peller, N., Rapp, Ch. & Manhart, M. (2009). "Flow over periodic hills—Numerical and experimental study over a wide range of Reynolds numbers." *Computers & Fluids*, 38(2), 433–457.
4. Fröhlich, J., Mellen, C.P., Rodi, W., Temmerman, L. & Leschziner, M.A. (2005). "Highly resolved large-eddy simulation of separated flow in a channel with streamwise periodic constrictions." *J. Fluid Mech.*, 526, 19–66.
5. Greenblatt, D., Paschal, K.B., Yao, C.-S., Harris, J., Schaeffler, N.W. & Washburn, A.E. (2006). "Experimental Investigation of Separation Control Part 1: Baseline and Steady Suction." *AIAA Journal*, 44(12).

### NASA Programs and Vision Documents
6. Slotnick, J., Khodadoust, A., Alonso, J., Darmofal, D., Gropp, W., Lurie, E. & Mavriplis, D. (2014). "CFD Vision 2030 Study: A Path to Revolutionary Computational Aerosciences." NASA/CR-2014-218178.
7. Rumsey, C.L. et al. (2017–2023). NASA Juncture Flow Experiment. Multiple NASA TMs and AIAA papers.
8. NASA Turbulence Modeling Resource (TMBWG): https://tmbwg.github.io/turbmodels (migrated from https://turbmodels.larc.nasa.gov/)

### Turbulence Modeling
9. Spalart, P.R. & Allmaras, S.R. (1992). "A One-Equation Turbulence Model for Aerodynamic Flows." AIAA Paper 92-0439.
10. Menter, F.R. (1994). "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications." *AIAA Journal*, 32(8), 1598–1605.
11. Spalart, P.R. et al. (1997). "Comments on the Feasibility of LES for Wings, and on a Hybrid RANS/LES Approach." 1st AFOSR Int. Conf. DNS/LES.
12. Shur, M.L. et al. (2008). "A hybrid RANS-LES approach with delayed-DES and wall-modelled LES capabilities." *Int. J. Heat and Fluid Flow*, 29(6), 1638–1649.

### Machine Learning and Data-Driven Methods
13. Ling, J., Kurzawski, A. & Templeton, J. (2016). "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *J. Fluid Mech.*, 807, 155–166.
14. Parish, E.J. & Duraisamy, K. (2016). "A paradigm for data-driven predictive modeling using field inversion and machine learning." *J. Comput. Phys.*, 305, 758–774.
15. Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). "Physics-informed neural networks." *J. Comput. Phys.*, 378, 686–707.
16. Srivastava, B. et al. (2024). "Augmenting RANS Turbulence Models Guided by Field Inversion and Machine Learning." NASA TM-20240012512.
17. McConkey, R., Yee, E. & Lien, F. (2021). "A curated dataset for data-driven turbulence modelling." *Scientific Data*, 8(1). DOI: 10.1038/s41597-021-01034-2.
18. Haghahenas, A., Hedayatpour, S. & Groll, R. (2023). "Prediction of particle-laden pipe flows using deep neural network models." *Physics of Fluids*, 35, 083320.

### Solver Documentation and Comparison
19. OpenFOAM Foundation. *OpenFOAM User Guide*. https://www.openfoam.com/
20. SU2 Development Team. *SU2: An Open-Source Suite for Multiphysics Simulation and Design*. https://su2code.github.io/
21. ANSYS Inc. *ANSYS Fluent Theory Guide*.

### European Research
22. ERCOFTAC Benchmark Databases. https://www.ercoftac.org/
23. DLR Institute of Aerodynamics and Flow Technology. Various technical reports on turbulent boundary layers and transition.
24. ZARM, University of Bremen. Research group pages: https://www.zarm.uni-bremen.de/

---

*This literature review was compiled as part of the project "CFD Solver Benchmark for Flow Separation Prediction" for the MSc Space Engineering program at the University of Bremen.*
