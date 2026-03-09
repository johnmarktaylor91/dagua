"""Module-level defaults — device, theme.

Minimal global state. Prefer passing these explicitly to Graph; these exist
only as convenience to avoid repeating device='cuda' on every call.

Usage:
    dagua.set_default_device('cuda')
    dagua.set_default_theme('dark')
"""
