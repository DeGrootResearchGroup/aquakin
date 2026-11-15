"""aquakin's browser interface: a flowsheet editor (``aquakin-ui``).

An optional extra (``pip install "aquakin[ui]"``). The interface never imports
aquakin or JAX: it asks a long-lived ``aquakin serve`` worker process for the
unit catalogue and for checks, so the page starts without JAX and aquakin never
loads the interface.
"""
