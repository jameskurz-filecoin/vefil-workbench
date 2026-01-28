# Deployment Guide

This guide covers deploying the veFIL Tokenomics Workbench to Streamlit Cloud.

## Prerequisites

- GitHub account
- This `python-app` folder pushed to a GitHub repository
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Streamlit Cloud Deployment

### Step 1: Prepare Repository

Ensure your GitHub repository contains the `python-app` folder with this structure:

```
python-app/
├── streamlit_app.py      # Entry point (required)
├── requirements.txt      # Dependencies (required)
├── setup.py              # Package config
├── .streamlit/
│   └── config.toml       # Theme settings
└── src/
    └── vefil/            # Python package
```

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select your repository
5. Configure:
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `python-app/streamlit_app.py`
   - **App URL**: Choose your subdomain (e.g., `vefil-workbench`)

### Step 3: Deploy

Click "Deploy!" and wait for the build to complete. First deployment takes 2-5 minutes.

### Step 4: Verify

Your app will be available at: `https://your-app-name.streamlit.app`

## Configuration Options

### Custom Domain (Optional)

Streamlit Cloud supports custom domains on paid plans:

1. Go to app settings
2. Click "Custom domain"
3. Add your domain and configure DNS

### Environment Variables

If you need secrets (API keys, etc.):

1. Go to app settings → "Secrets"
2. Add in TOML format:
   ```toml
   [secrets]
   api_key = "your-key-here"
   ```
3. Access in code:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

### Python Version

Streamlit Cloud uses Python 3.9+ by default. To specify a version, create `runtime.txt`:

```
python-3.11
```

## Troubleshooting

### Build Fails

**Check requirements.txt**: Ensure all dependencies are listed and versions are compatible.

```bash
# Test locally first
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Check imports**: All imports must use `from vefil...` (not `from src.vefil...`).

### App Crashes on Load

**Check config path**: The `defaults.yaml` file must be included in the package. Verify `setup.py` has:

```python
package_data={
    "vefil": ["config/defaults.yaml"],
},
include_package_data=True,
```

### Cold Starts

Free tier apps "sleep" after 7 days of inactivity. First access after sleep takes ~30 seconds to wake up. This is normal behavior.

### Memory Limits

Free tier has 1GB memory limit. For Monte Carlo with many runs:

```python
# Reduce runs for cloud deployment
config.simulation.monte_carlo_runs = 50  # Instead of 100
```

## Updating the App

### Automatic Updates

Streamlit Cloud automatically redeploys when you push to the connected branch:

```bash
git add .
git commit -m "Update simulation parameters"
git push origin main
# App redeploys automatically in ~1-2 minutes
```

### Manual Reboot

If the app gets stuck:

1. Go to app settings
2. Click "Reboot app"

### Rollback

To rollback to a previous version:

1. Revert your git commit locally
2. Push to trigger redeploy

Or use Streamlit's version history (paid plans only).

## Performance Optimization

### Caching

Add caching to expensive operations:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_simulation(config_hash: str):
    # Expensive computation
    return result
```

### Session State

Use session state for user-specific data:

```python
if 'simulation_result' not in st.session_state:
    st.session_state.simulation_result = None
```

### Lazy Loading

Only compute what's visible:

```python
if tab == "Stress Tests":
    # Only run stress test computation when tab is active
    stress_results = compute_stress_tests()
```

## Monitoring

### View Logs

1. Go to app settings
2. Click "Logs" to see recent output
3. Use `st.write()` or `print()` for debugging

### Analytics

Streamlit Cloud provides basic analytics:

- Viewer count
- Geographic distribution
- Usage patterns

## Alternative Deployment Options

If Streamlit Cloud doesn't meet your needs:

### Render.com

```yaml
# render.yaml
services:
  - type: web
    name: vefil-workbench
    runtime: python
    buildCommand: pip install -r python-app/requirements.txt
    startCommand: streamlit run python-app/streamlit_app.py --server.port $PORT
```

### Railway.app

```bash
# Install Railway CLI
railway login
railway init
railway up
```

### Docker (Self-Hosted)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY python-app/ .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

```bash
docker build -t vefil-workbench .
docker run -p 8501:8501 vefil-workbench
```

## Cost Comparison

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| Streamlit Cloud | 1 public app, 1GB RAM | $250/mo for private apps |
| Render | 750 hrs/mo, sleeps after 15min | $7/mo always-on |
| Railway | $5 free credit/mo | Pay per use |
| Self-hosted | Full control | Server costs |

## Security Notes

1. **No secrets in code**: Use environment variables or Streamlit secrets
2. **Public by default**: Free Streamlit apps are publicly accessible
3. **No user auth**: Add authentication for sensitive data (Streamlit Cloud paid feature)

