`aquakin-ui` no longer leaves aquakin worker processes behind. A worker that
stopped reading was replaced but kept running (with JAX loaded) until the
editor closed, and a worker that timed out stayed a zombie; both are now
killed and reaped, and the pipes of every worker and study run are closed
when it ends.
