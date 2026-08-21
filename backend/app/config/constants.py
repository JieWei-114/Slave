"""
Constants for backend API
Centralizes magic strings and action types for better maintainability
and to avoid typos in string comparisons
"""

# ============================================================
# HTTP Status Codes
# ============================================================

# Common status codes for better readability
HTTP_INTERNAL_ERROR = 500
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUEST = 429
HTTP_NOT_FOUND = 404
HTTP_BAD_GATEWAY = 502
