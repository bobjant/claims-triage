# Underwriting Rules — Homeowners Property Claims (SYNTHETIC)

> Synthetic reference material for the Claims Triage Assistant demo. Not real underwriting guidance.
>
> Each rule has an ID, a machine-checkable **Condition** written over the claim *facts* returned by the
> `check_coverage` skill, a recommended **Action**, and a **Severity**. Actions, from least to most
> severe: `note` (informational only) < `request_documentation` < `route_to_investigator`.
> When several rules fire, the most severe action wins.

### UW-01 · Early-inception claims
- **Condition:** `days_since_inception <= 7`
- **Action:** route_to_investigator
- **Severity:** high

Losses that occur within 7 days of policy inception must be reviewed by an investigator regardless of
the amount. Losses right after inception are a common sign that a policy was bought to cover a loss
that had already happened or was expected.

### UW-02 · Claim amount near the policy limit
- **Condition:** `pct_of_limit >= 98 and pct_of_limit <= 100`
- **Action:** route_to_investigator
- **Severity:** high

Claim amounts within 2% of the policy limit need investigator review. Claims set exactly at the limit
are more likely to be inflated to the maximum payable.

### UW-03 · Claim amount exceeds the policy limit
- **Condition:** `claim_amount > policy_limit`
- **Action:** route_to_investigator
- **Severity:** high

The payable amount is capped at the policy limit. An investigator must confirm the scope of the loss
and explain the cap to the policyholder.

### UW-04 · Loss outside the active policy period
- **Condition:** `loss_in_policy_period == False`
- **Action:** route_to_investigator
- **Severity:** high

A loss dated before `active_from` or after `active_to` is presumptively not covered. A human must
make the declination decision; the system never declines a claim automatically.

### UW-05 · Late reporting
- **Condition:** `report_lag_days > 30`
- **Action:** request_documentation
- **Severity:** medium

Losses reported more than 30 days after they occurred need a written explanation of the delay and
evidence that the damage was not made worse by the delay (mitigation receipts, photos).

### UW-06 · Desk-review threshold
- **Condition:** `claim_amount > 10000`
- **Action:** request_documentation
- **Severity:** medium

Claims above 10,000 can never be auto-approved. They need an itemised estimate or contractor quote
before settlement.

### UW-07 · Auto-approval eligibility
- **Action:** auto_approve
- **Severity:** info

A claim may be recommended for `auto_approve` only when it passed intake validation and **no** rule
in UW, COV or FI fires with an action more severe than `note`. The adjuster HITL gate still confirms
every auto-approval before settlement.

### UW-08 · Unknown policy
- **Condition:** `policy_found == False`
- **Action:** route_to_investigator
- **Severity:** high

A claim that cannot be matched to a policy in the register can't be assessed for coverage and may
point to identity or policy fraud.
