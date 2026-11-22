`Plant.set_temperature()` now sets the temperature of MBR and SBR units too;
it skipped them before. An SBR's kinetics run at this static temperature: it
does not follow its inlet temperature and a heat-balance model does not track
it, since its volume changes over the cycle.
