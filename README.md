# From Data to Decisions Under Uncertainty: Modern Bayesian Computing with PyMC 

Traditional machine learning models excel at making predictions but often fall short when quantifying uncertainty—a critical requirement for high-stakes decisions in production systems. This talk introduces PyMC, a probabilistic programming framework for Python, demonstrating how modern Bayesian methods can enhance your ML toolkit with built-in uncertainty quantification and interpretable results. We'll begin by demystifying Bayesian computation through live coding examples, showing how PyMC makes it as easy to code probabilistic models as writing them on a whiteboard. Once specified, models can be compiled to a high-performance back end like JAX or Numba, and fit using one of several modern inference methods. I will cover key features like PyMC's coords and dims system that prevents dimensionality bugs, `pymc.Data` containers for production model updates, and functions `pymc.do()` and `pymc.observe()` for facilitating causal inference workflows. You'll learn the complete Bayesian workflow—from prior specification through model validation—with emphasis on practical implementation over mathematical theory. 

## Presentation Outline

### 1. Introduction: The Uncertainty Problem (10 minutes)
- **Motivation**: Why traditional ML falls short in high-stakes decisions
- **Real-world scenarios**: Medical diagnosis, financial risk, autonomous systems
- **The uncertainty quantification gap**: Point estimates vs. distributions
- **Enter PyMC**: Probabilistic programming for practical applications

### 2. PyMC Fundamentals: From Whiteboard to Code (15 minutes)
- **Live Demo**: Simple linear regression model
  - Mathematical specification → PyMC code
  - Model definition with `pm.Model()` context
  - Prior specification and likelihood construction
- **The PyMC ecosystem**: ArviZ, PyTensor backends (JAX/Numba)
- **Modern inference**: NUTS sampling and why MCMC works

### 3. Production-Ready Features (15 minutes)
- **Coords and Dims System**
  - Preventing dimensionality bugs with named dimensions
  - Live coding: Multi-dimensional model with labeled axes
  - Model visualization with `pm.model_to_graphviz()`
  
- **Data Containers with `pymc.Data`**
  - Efficient model updates without recompilation
  - Live demo: Updating model with new observations
  - `pm.set_data()` for production pipelines

### 4. Causal Inference Workflows (10 minutes)
- **Interventions with `pymc.do()`**
  - Simulating counterfactual scenarios
  - Live coding: "What if we changed this parameter?"
  
- **Conditioning with `pymc.observe()`**
  - Moving from generative to inference models
  - Live demo: Parameter estimation from observed data

### 5. Complete Bayesian Workflow (15 minutes)
- **Prior Predictive Checks**: Does your model make sense?
- **Posterior Inference**: MCMC diagnostics and convergence
- **Posterior Predictive Checks**: Model validation
- **Prediction and Decision Making**: From uncertainty to action
- **Live coding walkthrough**: End-to-end bioassay example

### 6. Modern Computational Backends (10 minutes)
- **JAX compilation**: High-performance gradient computation
- **Numba acceleration**: JIT compilation for custom operations
- **When to use which backend**: Performance considerations
- **Live demo**: Backend switching and performance comparison

### 7. Resources and Next Steps (5 minutes)
- **PyMC Learning Resources**: Documentation, examples, community
- **Integration ecosystem**: Bambi, PyMC-Marketing, PyMC-Extras
- **Getting involved**: Contributing to open source probabilistic programming

---

*Total presentation time: ~80 minutes with Q&A*

### Key Learning Outcomes
- Understand when and why to use Bayesian methods over traditional ML
- Write production-ready probabilistic models with PyMC
- Implement uncertainty quantification in real-world scenarios
- Use modern PyMC features: coords/dims, Data containers, causal inference
- Apply complete Bayesian workflow from prior to prediction
