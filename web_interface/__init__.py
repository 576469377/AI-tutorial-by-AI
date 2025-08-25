"""
Web interface for AI Tutorial - Interactive Learning Dashboard

This module provides a web-based interface for the AI Tutorial,
making it accessible through a browser with interactive features.
"""

from .dashboard_app import create_dashboard_app, DashboardServer

__all__ = [
    'create_dashboard_app',
    'DashboardServer'
]