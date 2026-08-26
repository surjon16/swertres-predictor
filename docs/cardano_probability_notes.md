# Cardano and Probability Theory — Relevance Check

Girolamo Cardano (16th century) wrote *Liber de Ludo Aleae* ("Book on Games
of Chance") — the first systematic probability treatise, roughly a century
before Pascal and Fermat. He was also a compulsive gambler himself, which is
part of why he wrote it.

## His core methods, and how they map to what's in this repo

1. **The "circuit"** — his term for the complete set of equally likely
   outcomes (what we'd now call the sample space). For a die, that's 6
   faces; for Swertres, that's 10 digits per position or 1000 combos
   overall. This is exactly the uniform-distribution assumption
   `analysis/bias_test.py` checks.

2. **Fair games defined by equiprobability** — Cardano's central idea was
   that a game is "fair" precisely when every outcome in the circuit is
   equally likely, and *unfair* when observed frequencies deviate
   persistently from that. This is the chi-square goodness-of-fit test in
   embryonic form, four centuries early — `bias_test.py` independently
   arrives at Cardano's own diagnostic.

3. **Multiplication rule for independent events** — Cardano was among the
   first to state that the probability of two independent events both
   happening is the product of their individual probabilities. That's
   exactly the mechanism behind `top_k_combos()` in `analysis/nn_test.py`,
   which ranks 3-digit combos by multiplying per-position probabilities
   together.

4. **An early Law of Large Numbers** — Cardano noted that as trials
   increase, observed frequency converges toward true probability, and
   that small-sample deviations are expected noise, not evidence of
   unfairness. This is why `analysis/date_grouping_test.py`'s
   `same_day_month` slice (n≈23) was flagged as too thin to trust, while
   the full 9PM history (n≈7,959) is the trustworthy test — Cardano's own
   reasoning, not just modern statistical convention.

5. **He explicitly considered physical dice bias** — Cardano wrote about
   warped or weighted dice as a real, detectable cause of deviation from
   fairness. That's the same mechanical-equipment-wear hypothesis this
   project set out to test — a legitimate ~500-year-old question, not a
   fringe idea. Run Cardano's way (the chi-square test), it found no such
   deviation in this data.

## Net finding

There's no new method here that isn't already applied in this repo —
Cardano's toolkit *is* the chi-square/frequency approach used from the
start, just in its original form. This is a historical confirmation that
the project tested the right hypothesis with the right kind of tool from
the beginning; it doesn't unlock a new angle for extracting signal from
this dataset.

## Sources

- [Decoding Cardano's Liber de Ludo Aleae (PDF)](http://aurora.troja.mff.cuni.cz/~santolik/EVF503/Cardano_Liber_de_Ludo_Aleae.pdf)
- [Britannica: The Book on Games of Chance](https://www.britannica.com/topic/The-Book-on-Games-of-Chance)
- [Cardano, Gambling and the Dawn of Probability Theory](https://www.gameludere.com/2020/03/30/cardano-gambling-dawn-of-probability-theory/)
- [Chance Combinatorics: The Theory that History Forgot](https://sites.pitt.edu/~jdnorton/papers/chance_combinatorics_final.pdf)
