# Historical Fraud Indicator Patterns (SYNTHETIC)

> Synthetic patterns drawn from a fictional portfolio's historical SIU (Special Investigations Unit)
> referrals. The `prior_*` facts come from the long-term claims memory and from earlier claims in the
> current batch.

### FI-01 · Repeat claim for the same peril within 12 months
- **Condition:** `prior_same_type_12m >= 1`
- **Action:** route_to_investigator
- **Severity:** high

A second claim for the same peril on the same policy within 12 months, especially water damage, has
historically been the strongest single predictor of an SIU referral. The investigator should check
whether the earlier damage was actually repaired and whether the claims overlap.

### FI-02 · High claim frequency
- **Condition:** `prior_claims_12m >= 3`
- **Action:** route_to_investigator
- **Severity:** high

Three or more earlier claims on the policy in the 12 months before this loss.

### FI-03 · Previously flagged policyholder
- **Condition:** `prior_flagged_12m >= 1`
- **Action:** request_documentation
- **Severity:** medium

The policy had a claim in the last 12 months that wasn't auto-approved. Ask for documents that show
the earlier issue was resolved (repair invoices, a closure letter).

### FI-04 · Round-number amounts
- **Condition:** `claim_amount >= 5000 and claim_amount % 5000 == 0`
- **Action:** note
- **Severity:** low

Large claims in exact round numbers are more often estimates than itemised losses. This is a weak
signal. Mention it in the briefing but don't escalate on it alone.

### FI-05 · Same-day reporting of high-value theft
- **Condition:** `claim_type == "theft" and report_lag_days == 0 and claim_amount >= 10000`
- **Action:** note
- **Severity:** low

High-value theft reported the same day is normal in itself. Combined with early inception or
round-number amounts, it strengthens an investigator referral.

### FI-06 · Narrative inconsistencies (judgement)
- **Action:** note
- **Severity:** low

Watch for descriptions that contradict the claim type, vague descriptions of high-value losses, or
wording copied from an earlier claim. This indicator has no automatic condition. The Anomaly &
Coverage agent may raise it using judgement, and it is shown to the adjuster as a model judgement,
not a verified rule.
