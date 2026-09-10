# Runtime configuration reload

Allow a running process to load, validate, and atomically activate a new YAML configuration, initially through SIGHUP or an equivalent local administrative trigger. The old immutable configuration should continue serving requests unless the entire replacement validates successfully, and in-flight requests should retain the configuration snapshot with which they began.

Reloading would let operators rotate provider credentials, change models, adjust timeouts, or respond to an outage without restarting the listener and interrupting streams. It preserves the existing file-as-UI model without requiring a database or management API.

The implementation should keep failure behavior boring: reject invalid replacements, retain the last known-good configuration, emit a clear audit-style log, and avoid partially applying routes. Remote administration and a web UI are separate concerns.
