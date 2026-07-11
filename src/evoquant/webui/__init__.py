"""Offline control-plane dashboard + local API server (Phase 9).

The same `dashboard.html` is served here (stdlib http.server) and embedded
into the Cloudflare Worker. The Worker is the control plane; the Python
engine is the compute plane. Neither computes fitness or stores lockbox
data.
"""
