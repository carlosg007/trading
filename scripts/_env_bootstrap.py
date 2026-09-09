# Load .env into the process environment for scripts run by systemd units.
# Kept minimal; the trading.env file in deploy/systemd/ is the primary
# environment for the daemon units and is NOT rediscovered by these loaders.
from pathlib import Path
from dotenv import load_dotenv

_repo = Path(__file__).resolve().parent.parent
_loaded = load_dotenv(_repo / ".env", override=False)
