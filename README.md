# EV Battery Digital Twin — Sem 8 Project

This is my Semester 8 project on building a **digital twin for EV batteries**. The idea is to combine physics-based models with machine learning to predict battery temperature and track how the battery degrades over time. Basically, instead of just using ML blindly, I'm using known battery physics first and then using ML to fix whatever the physics model gets wrong.

---

## What's inside

Here's a quick overview of how I've organized the code:

- `dashboard/app.py` — the main UI built using Streamlit, this is what you interact with
- `src/physics/` — contains the thermal model and degradation model (the physics side)
- `src/data/` — handles loading data, building features, and splitting by battery cycles
- `src/ml/` — the ML residual model and some uncertainty estimation stuff
- `src/training/` — scripts to train and evaluate the model
- `src/evaluation/` — metrics and benchmark comparison helpers
- `tests/` — basic unit tests to make sure things don't break

---

## How to set it up

First install all the required packages:

```bash
pip install -r requirements.txt
```

> Make sure you're using Python 3.8+ and preferably inside a virtual environment.

---

## How to run things

**To open the dashboard:**
```bash
streamlit run dashboard/app.py
```

**To run the benchmark script:**
```bash
python scripts/run_benchmark.py
```

**To run the tests:**
```bash
pytest -q
```

---

## What I'm still working on (Sem 8 extensions)

These are the things I'm adding as part of my Sem 8 deliverables:

- **SOH and RUL prediction** — estimating State of Health and Remaining Useful Life using discharge capacity as the target. This will go in `src/ml/soh_rul_model.py`
- **Proper train/val/test split for SOH/RUL** — the split has to be time-ordered (can't shuffle battery cycle data randomly), and I'm also adding interval-based metrics to see how well the uncertainty estimates work
- **Anomaly detection at the cycle level** — flagging unusual cycles that might be safety concerns. This will go in `src/ml/anomaly.py`
- **Experiment tracking** — planning to log runs properly so I can compare what changes actually help (ablation study style)

---

## Notes

- The physics model runs first, and the ML model only learns the *residual* (i.e., what physics couldn't predict). This hybrid approach is the core idea of the project.
- The data splits are done cycle-aware, meaning I don't let future cycle data leak into training — which is important for battery degradation tasks.
- I'll keep updating this README as I add more stuff.