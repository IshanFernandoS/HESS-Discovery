FUTURE FABRICATION SHORTLIST FROM THE FROZEN JOINT-SELECTION INDEX
20 November 2026

Primary file
------------
future_fabrication_high_joint_index_candidates.csv

Frozen inputs
-------------
- inputs/final_scores_3301.csv
- inputs/retention_scoped_records.csv

These copies make the shortlist generator self-contained. Their SHA-256 hashes
are recorded in provenance_and_summary.json and are checked during repository
verification.

The CSV contains prospective compositions meeting all three rules:

1. frozen joint-selection index I > 0;
2. top 10 joint-index rank within its cation order; and
3. absent from the manuscript's current 12-record retention cohort.

The resulting 59 rows include lower-order controls and higher-order prospective
fabrication candidates. HEC11 contributes only its three positive-index
compositions. "Absent from the current retention cohort" does not establish
that a chemistry has never appeared anywhere in the literature.

next_project_group separates 11 core higher-order prospects with I >= 0.50,
nine exploratory HEC9–HEC11 prospects with 0 < I < 0.50, and 39 lower-order
controls. This is a transparent grouping of the existing index, not a new score.

How to use the CSV
------------------
- Sort or filter by screening_tier and sequential_index for strong persistence.
- Filter project_role to "higher-order prospective fabrication" for HEC9–HEC11.
- Use higher_order_shortlist_rank to read the 20 HEC9–HEC11 rows in descending I.
- Use joint_rank_within_order when selecting matched same-order comparisons.
- Use fabrication_attention_flags to plan validation for Cr, Sc, Y, Re or HEC11.
- Treat application_search_tags as search prompts for literature/property work.
  They are not model-predicted properties or validated application claims.

The joint-selection index is conditional on the declared 3,301-candidate
universe. A high value indicates persistent passage through the existing
two-stage screening grid. It does not predict density, hardness, oxidation
resistance, conductivity, electromagnetic absorption or synthesis success.

Recommended next experimental design
------------------------------------
Select same-order candidates that create a deliberate contrast between the
joint index and EFA percentile, include at least one additional backbone, and
freeze the scores and endpoints before fabrication. Use replicated synthesis,
XRD, density and quantitative matrix-region EDS. Application-specific tests
should follow only after phase formation and cation retention are confirmed.

Reproduce
---------
From this directory, run:

python -X utf8 generate_shortlist.py

provenance_and_summary.json records input and output hashes and row counts.
No model was retrained and no application-performance model was introduced.

