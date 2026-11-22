`asm2d_tud` declares the units its parameters actually have rather than 1/d
or mol: `mAN` in g P/(g COD·d); `qPP` in (g P/g COD)²/d, since the poly-P
storage rate is inversely proportional to the stored fraction XPP/XPAO
(Murnleitner et al. 1997, which quotes it in g P/(g COD·d) by treating that
fraction as a pure ratio); `iTSS_PP` in g TSS/g P; `iNO3_N2` in g COD/g N; and
the `iCharge_*` factors in kmol per gram. The model now cites Meijer et al.
(2001), Murnleitner et al. (1997) and van Veldhuizen et al. (1999). Values and
kinetics are unchanged.
