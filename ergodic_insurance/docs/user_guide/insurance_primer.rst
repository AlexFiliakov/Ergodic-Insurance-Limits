Commercial Insurance in One Page
================================

A five-minute primer on the handful of commercial-insurance terms you will meet
throughout this guide, the example notebooks, and the HJB optimization
experiment. It is written for readers who are already comfortable with the
*ergodic* idea -- time-average growth versus ensemble average -- but who have
never had to read a commercial insurance program. If a chart or notebook says
"buy down the SIR on a \$200M tower," this page is all the vocabulary you need to
follow it.

You do not need any of this to understand *why* insurance helps (see
:doc:`executive_summary`); you need it to follow *how* a real program is built,
priced, and optimized.

The one picture that unlocks everything: the tower
--------------------------------------------------

A commercial insurance program is rarely a single policy. It is a **stack of
layers**, drawn as a vertical *tower*, each layer covering a different band of
loss. The company keeps the very bottom band itself; insurers cover the bands
above, up to some maximum. Above the top, the company is exposed again.

Here is the exact tower used in the notebook 07 experiment, for a \$5M
manufacturer (an illustrative structure -- the dollar figures are specific to
that example):

.. code-block:: text

       $200M  +---------------------------+   <- top of program
              |  Layer 4: Catastrophic    |
              |  $150M xs $50M            |   rare, near-existential events
       $50M   +---------------------------+
              |  Layer 3: 2nd Excess      |
              |  $25M xs $25M             |   infrequent, severe losses
       $25M   +---------------------------+
              |  Layer 2: 1st Excess      |
              |  $20M xs $5M              |   large losses
       $5M    +---------------------------+
              |  Layer 1: Primary         |
              |  $4.75M xs $250K          |   the "working layer" (frequent)
       $250K  +---------------------------+
              |  Self-Insured Retention   |   the company pays this band itself
       $0     +---------------------------+

Read each layer as **"limit xs attachment"**: "\$20M xs \$5M" means a limit of
\$20M sitting *excess of* (on top of) a \$5M attachment point -- so it covers the
band from \$5M to \$25M.

How a single loss flows through the tower
-----------------------------------------

Follow one loss up the stack. Each band pays only its own slice:

* **Loss below \$250K** -- the company pays in full (this is the retention).
* **\$250K to \$5M** -- Layer 1 (Primary) responds, after the company's first \$250K.
* **\$5M to \$25M** -- Layer 2 (1st Excess) picks up the next band.
* **\$25M to \$50M** -- Layer 3 (2nd Excess).
* **\$50M to \$200M** -- Layer 4 (Catastrophic).
* **Above \$200M** -- uninsured; the company is fully exposed again.

So a \$30M loss is split: \$250K (company) + \$4.75M (Layer 1) + \$20M (Layer 2)
+ \$5M (Layer 3) = \$30M, with Layer 4 left untouched.

The vocabulary, defined against that picture
--------------------------------------------

**Self-Insured Retention (SIR) / Retention / Deductible**
   The amount of each loss the company keeps before any insurance pays -- the
   bottom band of the tower. In this framework the three terms are used
   interchangeably, and the SIR is **the one knob the optimizer turns**: raise it
   and you keep more risk but pay less premium; lower it and insurance does more
   work for a higher price. (In real contracts a true SIR and a deductible differ
   in who administers the claim and posts collateral; for our purposes they are
   the same lever.)

**Layer**
   One horizontal band of coverage, defined by an attachment point and a limit.

**Attachment point**
   The loss level at which a layer *starts* paying. Layer 2 above attaches at \$5M.

**Limit**
   The most a layer -- or the whole program -- will pay. Layer 2's limit is \$20M,
   so it covers the \$5M-\$25M band and no more.

**Primary, Excess, Catastrophic (Cat)**
   The **primary** layer sits just above the retention and absorbs the frequent,
   smaller losses (the "working layer"). **Excess** layers stack above it for
   larger, rarer losses. The **cat** layer at the top is for rare, near-existential
   events.

**Tower / Program**
   The full stack from the retention up to the top limit (\$200M here). A loss
   above the top is retained by the company.

**Premium**
   The annual price paid for a layer, or for the whole tower.

**Rate-on-line (ROL)**
   Premium divided by a layer's limit -- the price per dollar of coverage. It
   **falls steeply as you climb the tower**: the primary layer is hit most years
   (high ROL), while the cat layer almost never attaches (low ROL).

**Loss ratio**
   Expected losses divided by premium. A 65% loss ratio means 65 cents of every
   premium dollar is expected to return as losses; the other 35 cents is the
   insurer's **loading** for expenses, capital, and profit. Lower layers carry
   higher loss ratios; tail layers are loaded more heavily for their volatility.

**Frequency-severity model**
   How losses are generated: a **frequency** distribution for *how many* events
   occur per year (here, Poisson) and a **severity** distribution for *how big*
   each one is (lognormal for ordinary losses, heavy-tailed Pareto for
   catastrophes). You already know these as ordinary probability; insurance simply
   separates "how often" from "how bad."

Why this matters for ergodic optimization
-----------------------------------------

Everything above is structural plumbing. The reason the framework cares is that
the **retention is a growth decision, not merely a cost decision**. A lower SIR
drags on the survivors' growth -- you pay more premium every year -- but it
sharply cuts the chance of a loss large enough to end the company. Because a
single firm lives *one* path through time, avoiding ruin can raise its
time-average growth rate even when the premium exceeds expected losses -- the
central result this documentation develops. The HJB experiment in notebook 07
simply searches for the SIR, as a function of the firm's size and leverage, that
maximizes that time-average growth.

.. note::

   **Want the formal definitions?** The :doc:`glossary` lists every term
   alphabetically. For the live example this page is drawn from, see Part 3 of the
   `notebook 07 HJB experiment
   <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/blob/main/ergodic_insurance/notebooks/optimization/07_hjb_insurance_optimization.ipynb>`__.
