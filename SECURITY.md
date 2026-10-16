# Security policy

## Supported versions

Fixes go into the latest release only (see the [changelog](CHANGELOG.md)).

## Reporting a vulnerability

Please report security problems privately, through GitHub's private vulnerability reporting: the repository's
**Security** tab → **Report a vulnerability**. Do not open a public issue for them.

A report is acknowledged within a week. Once the problem is confirmed, the fix and a new release follow as quickly as
the problem's severity warrants, and the advisory credits the reporter unless they prefer otherwise.

quantmet compiles one small C kernel (`quantmet/tc_detect.c`) on first use into a per-user cache directory; problems in
that path are in scope.
