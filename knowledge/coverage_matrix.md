# Coverage Matrix — Homeowners Policy Form (SYNTHETIC)

> Synthetic reference material. Describes which perils the demo homeowners form covers and the key
> exclusions and sub-limits. Conditions are evaluated over the facts returned by `check_coverage`.

Covered perils for `coverage_type == "homeowners"`: fire, smoke, lightning, wind_hail, theft,
vandalism, and water_damage caused by sudden and accidental discharge (e.g. a burst pipe or failed
appliance hose). Auto perils (collision, auto_liability) are written on a separate auto form.

### COV-01 · Peril not covered by the homeowners form
- **Condition:** `coverage_type == "homeowners" and claim_type not in ["fire", "smoke", "lightning", "wind_hail", "theft", "vandalism", "water_damage"]`
- **Action:** route_to_investigator
- **Severity:** high

The claim type is not a covered peril on this form. A coverage specialist must confirm and issue any
declination. Check whether the customer has another policy (e.g. auto) that applies.

### COV-02 · Flood and surface-water exclusion
- **Condition:** `claim_type == "water_damage" and ("flood" in description or "groundwater" in description or "seepage" in description or "surface water" in description)`
- **Action:** request_documentation
- **Severity:** medium

Flood, surface water, groundwater and long-term seepage are excluded. Water damage from a sudden,
accidental discharge is covered. Ask for a plumber's or cause-of-loss report showing where the water
came from before deciding coverage.

### COV-03 · Jewelry and watch theft sub-limit
- **Condition:** `claim_type == "theft" and ("jewel" in description or "watch" in description) and claim_amount > 1500 and "jewelry" not in scheduled_items`
- **Action:** request_documentation
- **Severity:** medium

Theft of jewelry and watches is limited to 1,500 in total unless the items are scheduled on the
policy. Ask for purchase receipts or appraisals and a police report. The payout above the sub-limit is
not covered.

### COV-04 · Theft requires a police report
- **Condition:** `claim_type == "theft" and claim_amount > 1000`
- **Action:** request_documentation
- **Severity:** low

Theft claims above 1,000 need a police report reference number before settlement.
