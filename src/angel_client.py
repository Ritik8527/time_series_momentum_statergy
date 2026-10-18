"""
STEP 1 - Connect to Angel One SmartAPI.

What happens when you log in:
  1. You prove who you are with 4 things: API key, client code, PIN, and a 6-digit TOTP code.
  2. The TOTP code changes every 30 seconds. We do NOT type it; we generate it from your
     TOTP *secret* with the pyotp library (the same maths your authenticator app uses).
  3. Angel One returns a session (tokens). The SmartConnect object keeps them for us.

Run this file on its own to test your login:
    python src/angel_client.py
"""
import os
import sys


def _need(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing {name}. Add it to your .env file (see env.example.txt).")
    return value


def get_client():
    """Log in and return a ready SmartConnect client."""
    # imports live inside the function so other files can be tested without these packages
    from dotenv import load_dotenv
    import pyotp
    try:
        from SmartApi import SmartConnect
    except ImportError as e:
        raise SystemExit(
            f"Could not import SmartApi: {e}\n"
            "Fix: python -m pip install smartapi-python logzero websocket-client pycryptodome"
        )

    load_dotenv()                            # reads .env into environment variables
    api_key = _need("ANGEL_API_KEY")
    client_code = _need("ANGEL_CLIENT_CODE")
    pin = _need("ANGEL_PIN")
    totp_secret = _need("ANGEL_TOTP_SECRET")

    client = SmartConnect(api_key=api_key)
    otp = pyotp.TOTP(totp_secret).now()
    resp = client.generateSession(client_code, pin, otp)
    if not resp or not resp.get("status"):
        msg = resp.get("message") if isinstance(resp, dict) else resp
        raise RuntimeError(f"Angel One login failed: {msg}")
    return client


def logout(client) -> None:
    try:
        client.terminateSession(os.getenv("ANGEL_CLIENT_CODE"))
    except Exception as e:                   # logout problems should never crash a download
        print("logout warning:", e)


if __name__ == "__main__":
    c = get_client()
    code = os.getenv("ANGEL_CLIENT_CODE", "")
    print(f"Login OK for client {code[:2]}***{code[-2:]}")
    logout(c)
