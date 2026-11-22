`IFASUnit` / `MBBRUnit` now default to `n_layers=16`, up from 4. The biofilm
discretization is first order and its error is governed by the penetration
ratio `L_f/delta` rather than thickness alone: at `L_f/delta = 4` four layers
put the substrate flux 26% high, where sixteen keep a typical carrier biofilm
within a couple of per cent. The class docstring carries the error table; lower
it again for a thin, fully penetrated film, where four layers are accurate to
well under a per cent and cheaper.
