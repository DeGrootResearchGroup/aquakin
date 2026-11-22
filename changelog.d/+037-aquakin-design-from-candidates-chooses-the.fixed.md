`aquakin.design_from_candidates` chooses the best set of measurements from
a pool of candidates under a budget, given each candidate's Fisher
information (for example the `groups=` output of
`fisher_information_from_fn`). It supports per-candidate costs, required
candidates, replicates and information already in hand. It solves the
relaxed problem for D and A, which bounds every design and gives the chosen
design's efficiency, and it runs greedy and exchange searches. Small pools
are searched exhaustively. Scoring a subset of parameters and averaging over
parameter draws work as in `design_criterion`.
