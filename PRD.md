# PRD: Size-aware "Fit insight" on the product page

**Status:** prototype tested offline · **Owner:** Aditya Singh · **Teams:** Storefront (PDP),
Outbound (returns & exchanges), Inbound (catalogue & size charts)

## 1. Problem

Shoppers can't try clothes on, so they guess a size. When the guess is wrong the
order comes back as a return or an exchange, the customer rates the product lower,
and they trust the next size chart less.

Evidence from 192,544 real Rent the Runway reviews (2010–2018, 105,571 customers,
5,850 items):

| Finding | Number |
|---|---|
| Orders where the size did not fit | **26.2%** (13.4% too small, 12.8% too large) |
| Rated 10/10 when it fit vs when it didn't | 71% vs 45–49% |
| Rated 6/10 or lower when it fit vs when it didn't | 4.4% vs 15.5–17.2% (**3.5–3.9x**) |
| Worst categories | shift 31.6%, skirt 31.4%, jacket 30.1% · best: gown 20.7% |
| Items (30+ reviews) that clearly run small or large | **46%** (25.5% small, 20.1% large) |
| Customers with a past fit issue who hit another | 31.7% vs 26.1% after a good fit |
| Fit complaints that say so in the review text | 75% of "too small", 68% of "too large" |

### The insight that shapes the solution

**The direction of a misfit depends on the size, not just the item.** In sizes
0–4, "too large" is 2.3x as common as "too small" (17.9% vs 7.8%). In sizes 17–20
it flips: "too small" is 1.7x as common (18.3% vs 10.5%). A single "runs small"
label on the product page is therefore wrong for many shoppers.

## 2. Users and jobs

- **Shopper choosing a size:** "Tell me which size will actually fit me, and why I should believe it."
- **Outbound / returns team:** fewer fit-related returns and exchanges to process.
- **Inbound / catalogue team:** know which items have misleading size charts, so they can fix them at source.

## 3. Hypothesis

If the product page shows **size-aware** fit advice at the moment a shopper picks a
size, backed by real review counts, then fewer orders will be the wrong size,
without lowering add-to-cart conversion.

## 4. Solution (v1)

When a size is selected, a "Fit insight" box shows one of three messages, plus the evidence:

- *Runs small at this size: consider one size up*
- *Runs large at this size: consider one size down*
- *True to size for most people at this size*

> e.g. "23% of 521 past renters in sizes 0–4 said too large."

Working prototype: [`prototype/index.html`](prototype/index.html).

**Out of scope for v1:** body-measurement input, virtual try-on, changing the size
chart itself.

## 5. Offline evidence

The model was trained on reviews before 2017 and tested on 69,126 later reviews.

| | Size-blind item label | Size-aware model |
|---|---:|---:|
| "Size up" shown and the order really was too small | 36.0% | **41.0%** |
| "Size down" shown and the order really was too large | 29.7% | **36.8%** |
| Share of all fit problems the advice reaches (same 23% of orders) | 28.3% | **33.3%** |
| Advice pointing the wrong way | 2.1–3.7% | 2.4–3.4% |
| Can advise on new items with no reviews | no | yes |

Base rates in the test period: 14.5% too small, 12.5% too large.

## 6. Success metrics

- **Primary:** fit-issue rate per order (proxy for fit-related returns and exchanges).
- **Secondary:** average rating, exchange rate, repeat purchase within 90 days.
- **Guardrails:** product page → add-to-cart conversion, cart abandonment, average
  order value, and complaints about wrong advice.

## 7. Experiment

- 50/50 split **by customer**, so nobody sees both versions.
- To detect a 1-point drop in fit-issue rate (27.1% → 26.1%) at 95% confidence and
  80% power: **30,630 orders per arm**. A 2-point drop needs 7,564 per arm.
  ([`results/experiment.json`](results/experiment.json))
- Decision rule: ship if the primary metric improves and no guardrail drops by
  more than its threshold. Otherwise iterate on the message or the margin.

## 8. Rollout and operations

1. **Pilot:** one category with high fit issues (shift dresses or jackets), 10% of traffic.
2. **A/B test** as above.
3. **SOP for the catalogue team:** every week, list items whose live fit-issue rate is
   above the category average, and review their size charts.
4. **Retrain monthly**, and track wrong-direction advice as its own metric.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Wrong-direction advice (about 3% of shown advice) | Show the evidence counts, not just the verdict; only show advice above the margin |
| New items have no reviews | The model uses category-level and size-level patterns (8.8% of advice in the test was on brand-new items) |
| The data is from a US rental site, not Indian e-commerce | Re-train on the platform's own returns and reviews before launch |
| Shoppers may size up "just in case" and still return | Track exchange rate and return reason, not just the fit flag |

## 10. Next steps

- Tag fit problems from review **text** for orders with no explicit fit flag (about 70%
  of complaints say it in words). Built: a TF-IDF baseline (macro-F1 0.68 on 69K later
  reviews) and a Gemini tagger that also names the body area (`src/llm_tagger.py`).
  Next: run it on return comments, and route "bust" or "length" issues to the catalogue team.
- Feed return reasons, not just reviews, into the model.
