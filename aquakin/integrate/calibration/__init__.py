"""The implementation of :func:`aquakin.calibrate`, one concern per module.

- ``config``: the public ``OptimizerConfig`` / ``LaplaceConfig`` /
  ``FreeICConfig``.
- ``observation``: ``observation_fn`` handling (``observe`` and friends), shared
  with the Fisher-information path.
- ``loss``: the loss and residual builders, missing-data masking, ``sigma`` and
  observation-unit handling.
- ``coords``: the free-initial-condition and simplex coordinates.
- ``result``: ``CalibrationResult`` and ``PredictiveBand``.
- ``problem``: the forward-model seam (``_ReactorForwardModel``) and the resolved
  ``_CalibrationProblem``.
- ``objective``: the compiled objective, residual and Jacobian.
- ``multistart``: starting points, optimizer bounds and the multistart loop.
- ``laplace``: the Laplace posterior and the at-bound checks.
- ``driver``: the fit run shared by ``calibrate`` and ``Plant.calibrate``.

Nothing here is public API: the public names are reached through
``aquakin.integrate`` (or ``aquakin``), and ``aquakin.integrate.calibrate``
re-exports every name it has always exported.
"""
