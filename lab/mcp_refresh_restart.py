"""MCP Python SDK OAuth client: what happens to a stored refresh token after a restart?

Everything is in-memory: an httpx2 MockTransport plays the MCP server and the
authorization server. No network, no browser, no real credentials.

Case A (same process): the client learned the token lifetime, the token expires,
the next request triggers a proactive refresh. We record the refresh request body.
Case B (restart): a new provider loads the same tokens from storage. The token is
expired on the server, but the client never stored its expiry. We record whether the
client uses the stored refresh token or starts an interactive login.
"""
import asyncio
import json
import time
from importlib.metadata import version
from urllib.parse import parse_qs

import httpx2
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

MCP = "https://mcp.example.com/mcp"
AS = "https://auth.example.com"
LOG: list[dict] = []
STATE = {"valid_access": set(), "issued": 0}


class MemoryStorage(TokenStorage):
    def __init__(self):
        self.tokens, self.client = None, None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client

    async def set_client_info(self, client_info):
        self.client = client_info


def handler(request: httpx2.Request) -> httpx2.Response:
    url = str(request.url)
    if url.startswith(MCP):
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ")
        if token in STATE["valid_access"]:
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        return httpx2.Response(401, headers={"WWW-Authenticate":
            'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"'})
    if "oauth-protected-resource" in url:
        return httpx2.Response(200, json={"resource": MCP, "authorization_servers": [AS]})
    if "/.well-known/oauth-authorization-server" in url or "/.well-known/openid-configuration" in url:
        return httpx2.Response(200, json={
            "issuer": AS, "authorization_endpoint": f"{AS}/authorize", "token_endpoint": f"{AS}/token",
            "registration_endpoint": f"{AS}/register", "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"], "token_endpoint_auth_methods_supported": ["none"]})
    if url == f"{AS}/token":
        body = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        LOG.append({"event": "token_request", "grant_type": body.get("grant_type"),
                    "has_resource": "resource" in body, "resource": body.get("resource")})
        if body.get("grant_type") == "refresh_token" and "resource" not in body:
            # A strict RFC 8707 authorization server issues an audience-less token;
            # this resource server would reject it. Record and refuse here.
            return httpx2.Response(400, json={"error": "invalid_target"})
        STATE["issued"] += 1
        access = f"access-{STATE['issued']}"
        STATE["valid_access"] = {access}
        return httpx2.Response(200, json={"access_token": access, "token_type": "Bearer",
                                          "expires_in": 1, "refresh_token": "refresh-1"})
    LOG.append({"event": "unexpected", "url": url})
    return httpx2.Response(404)


class InteractiveLogin(Exception):
    pass


async def redirect_handler(url: str) -> None:
    LOG.append({"event": "interactive_login_started"})
    raise InteractiveLogin()


async def callback_handler():
    raise InteractiveLogin()


def provider(storage):
    meta = OAuthClientMetadata(redirect_uris=["http://localhost:3030/callback"], client_name="probe",
                               grant_types=["authorization_code", "refresh_token"])
    return OAuthClientProvider(server_url=MCP, client_metadata=meta, storage=storage,
                               redirect_handler=redirect_handler, callback_handler=callback_handler)


async def call(p, protocol_header=True):
    headers = {"MCP-Protocol-Version": "2025-06-18"} if protocol_header else {}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), auth=p) as client:
        try:
            r = await client.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=headers)
            return r.status_code
        except InteractiveLogin:
            return "interactive_login"


def seeded_storage():
    s = MemoryStorage()
    s.client = OAuthClientInformationFull(client_id="probe-client", redirect_uris=["http://localhost:3030/callback"],
                                          grant_types=["authorization_code", "refresh_token"],
                                          token_endpoint_auth_method="none")
    return s


async def case_same_process(protocol_header):
    LOG.clear()
    STATE["valid_access"], STATE["issued"] = set(), 0
    storage = seeded_storage()
    p = provider(storage)
    # Simulate an earlier login in this process: metadata discovered, token with known lifetime.
    STATE["valid_access"] = {"access-0"}
    storage.tokens = OAuthToken(access_token="access-0", expires_in=1, refresh_token="refresh-1")
    first = await call(p, protocol_header)          # 200, loads tokens (expiry unknown)
    p.context.update_token_expiry(storage.tokens)  # as if the client had received this token itself
    time.sleep(1.2)
    STATE["valid_access"] = set()                  # server-side expiry
    second = await call(p, protocol_header)
    return {"first_status": first, "after_expiry_status": second, "log": list(LOG)}


async def case_restart():
    LOG.clear()
    STATE["valid_access"], STATE["issued"] = set(), 0
    storage = seeded_storage()
    # Tokens persisted by a previous run; the access token has since expired server-side.
    storage.tokens = OAuthToken(access_token="access-old", expires_in=3600, refresh_token="refresh-1")
    status = await call(provider(storage))
    return {"status": status, "log": list(LOG),
            "stored_refresh_token_used": any(e.get("grant_type") == "refresh_token" for e in LOG)}


async def main():
    result = {"package": "mcp", "version": version("mcp"),
              "same_process_with_protocol_header": await case_same_process(True),
              "same_process_without_protocol_header": await case_same_process(False),
              "restart_expired_access_token": await case_restart(),
              "network_calls": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
