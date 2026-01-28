# veFIL Tokenomics Workbench (Python)

A rigorous cryptoeconomic modeling tool for vote-escrowed FIL (veFIL) token-locking mechanisms.

## Overview

This Python application provides a comprehensive simulation and analysis platform for evaluating veFIL tokenomics. It models:

- **Supply dynamics** - Token flows, inflation, reserve runway
- **Capital flows** - Net-new vs recycled vs cannibalized capital
- **Behavioral adoption** - Cohort-based participation modeling
- **Lending market impact** - Cannibalization scenarios
- **Adversarial scenarios** - Delta-neutral hedging, crisis behavior
- **Real-world externalities** - Storage pricing, hardware mismatch, network growth

## Quick Start

### Local Development

```bash
# Navigate to the python-app directory
cd python-app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

### Enabling the Expert Assistant (OpenAI API)

The P4 assistant uses the OpenAI API and reads your key from an environment variable.

```bash
# Never commit secrets. Set your key in the environment:
export OPENAI_API_KEY="sk-..."

# Optional: pick a model explicitly
export VEFIL_OPENAI_MODEL="gpt-5.2-mini"
```

In Streamlit Cloud, add `OPENAI_API_KEY` under **Settings → Secrets**.

### Running Simulations Programmatically

```python
from vefil.config.loader import load_config
from vefil.simulation.runner import SimulationRunner

# Load default configuration
config = load_config()

# Modify parameters as needed
config.simulation.time_horizon_months = 60
config.yield_source.reserve_annual_rate = 0.03  # 3% annual emission

# Run simulation
runner = SimulationRunner(config)
result = runner.run()

# Access results
print(f"Final locked: {result.final_metrics['final_locked']:,.0f} FIL")
print(f"Reserve runway: {result.final_metrics['reserve_runway_years']:.1f} years")
```

## Project Structure

```
python-app/
├── streamlit_app.py          # Main Streamlit UI (entry point)
├── setup.py                  # Package installation config
├── requirements.txt          # Dependencies for Streamlit Cloud
├── .streamlit/
│   ├── config.toml           # Streamlit theme/settings
│   └── style.css             # Custom CSS
├── src/vefil/                # Core Python package
│   ├── config/               # Configuration management
│   │   ├── schema.py         # Pydantic models (100+ parameters)
│   │   ├── loader.py         # YAML config loading
│   │   └── defaults.yaml     # Default parameter values
│   ├── engine/               # Deterministic simulation engines
│   │   ├── accounting.py     # Supply tracking & conservation
│   │   ├── rewards.py        # Yield source mechanics
│   │   ├── capital_flow.py   # Capital decomposition
│   │   └── lending.py        # Lending market dynamics
│   ├── behavior/             # Behavioral models
│   │   ├── adoption.py       # Lock demand modeling
│   │   ├── cohorts.py        # User segment definitions
│   │   └── opportunity_cost.py  # Alternative yield comparison
│   ├── adversarial/          # Risk scenarios
│   │   ├── hedging.py        # Delta-neutral analysis
│   │   ├── crisis.py         # Crisis behavior modeling
│   │   └── wrappers.py       # Liquid wrapper impact
│   ├── externalities/        # Real-world feedback loops
│   │   ├── storage_pricing.py  # Storage cost impact
│   │   ├── hardware.py       # Hardware depreciation
│   │   └── network_growth.py # Fee sustainability
│   ├── simulation/           # Orchestration
│   │   ├── runner.py         # Main simulation executor
│   │   └── monte_carlo.py    # Parameter sweeping
│   └── reporting/            # Output generation
│       ├── charts.py         # Plotly visualizations
│       └── export.py         # CSV/JSON/HTML export
└── docs/
    ├── ARCHITECTURE.md       # Detailed architecture docs
    └── DEPLOYMENT.md         # Streamlit Cloud deployment guide
```

## Key Features

### 1. Simulation Engine
- Discrete-time monthly simulation (configurable timestep)
- Conservation law validation (all FIL accounted for)
- Deterministic & reproducible (same seed = same results)

### 2. Yield Model
- **Reserve emissions**: Rewards funded from the 300M mining reserve
- Dynamic emission rate (default 2%, adjusts via policy to maintain APY floor)
- Configurable reward curve (k parameter controls how much longer locks are favored)

### 3. Behavioral Modeling
- Four cohorts: Retail, Institutional, Storage Providers, Treasuries
- Utility-driven participation based on opportunity costs
- Logistic response curves for realistic adoption dynamics

### 4. Risk Analysis
- Delta-neutral hedging profitability
- Crisis behavior predictions (diamond hands vs abandon)
- Liquid wrapper concentration risks

### 5. Externality Feedback
- Storage pricing impact from supply shocks
- 5-year lock vs 3-year hardware cycle mismatch
- Fee-based yield sustainability analysis

## Configuration

All parameters are defined in `src/vefil/config/defaults.yaml`. Key parameter groups:

| Group | Description | Example Parameters |
|-------|-------------|-------------------|
| `initial_supply` | Starting state | total, circulating, reserve |
| `yield_source` | Reward mechanics | type, reserve_annual_rate |
| `reward_curve` | Lock incentives | k (exponent), max_duration |
| `market` | Market conditions | liquidity_regime, volatility |
| `capital_flow` | Capital sources | net_new_fraction, cannibalized |
| `cohorts` | User segments | size_fraction, required_premium |
| `adversarial` | Risk scenarios | hedging rates, crisis thresholds |
| `externalities` | Real-world effects | network_growth_rate, hardware_depreciation |

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed parameter documentation.

## Deployment

This app is designed for deployment on **Streamlit Cloud**:

1. Push this `python-app` folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set the main file path to `streamlit_app.py`
5. Deploy

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Style

```bash
# Format code
black src/ streamlit_app.py

# Lint
ruff check src/ streamlit_app.py
```

### Adding New Features

1. Add engine/model in appropriate `src/vefil/` subdirectory
2. Update `Config` schema in `config/schema.py`
3. Add defaults to `config/defaults.yaml`
4. Integrate into `simulation/runner.py`
5. Add UI controls in `streamlit_app.py`
6. Update documentation

## License

MIT License - See LICENSE file for details.
