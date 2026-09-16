cd /Users/sgracey/Code/unifi-cam-proxy-kinda
source .venv/bin/activate
PYTHONPATH=. pytest tests/test_pairing.py -v
PYTHONPATH=. pytest tests/test_frigate_ingestion.py -v --log-cli-level=DEBUG
