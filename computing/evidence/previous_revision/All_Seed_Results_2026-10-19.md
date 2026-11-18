# All five seed results and source provenance

Tables S1-S8 report saved test metrics from validation-selected checkpoints; Tables S9-S12 report paired differences. Scratch values are accuracy; pretrained AG News/DBpedia use macro-F1, WNUT entity F1 and CIFAR accuracy. All entries are percentages. Diagnostic fail-fast/LR-assertion arms have no primary test endpoint and remain n.a. in the 240-job CSV. Original validation-only selection is unchanged.

## Table S1 AG News medium-4

Metric: accuracy.

| Seed | Stable | Single bad | Late | Rescue | Corrected |
|--:|--:|--:|--:|--:|--:|
| 42 | 92.2368 | 50.8816 | 68.8026 | 92.0132 | 92.9079 |
| 123 | 92.6447 | 36.9079 | 82.5263 | 92.5395 | 92.7105 |
| 456 | 92.4079 | 44.9211 | 63.2895 | 92.4342 | 92.3289 |
| 789 | 91.4868 | 50.3553 | 80.6579 | 92.4868 | 92.0658 |
| 2026 | 91.9868 | 41.7237 | 71.7368 | 92.4737 | 92.4211 |

## Table S2 AG News large-6

Metric: accuracy.

| Seed | Stable | Single bad | Late | Rescue | Corrected |
|--:|--:|--:|--:|--:|--:|
| 42 | 92.6053 | 48.4079 | 69.0789 | 92.4342 | 92.7105 |
| 123 | 92.6447 | 35.1447 | 78.6579 | 92.2237 | 92.3158 |
| 456 | 92.3816 | 37.0395 | 60.2105 | 92.2500 | 91.8158 |
| 789 | 91.7237 | 43.8947 | 61.6053 | 92.2105 | 92.3289 |
| 2026 | 92.4868 | 50.5263 | 73.0132 | 92.7500 | 92.4737 |

## Table S3 DBpedia medium-4

Metric: accuracy.

| Seed | Stable | Single bad | Late | Rescue | Corrected |
|--:|--:|--:|--:|--:|--:|
| 42 | 98.5971 | 37.0914 | 94.6057 | 98.5957 | 98.6543 |
| 123 | 98.6214 | 27.8714 | 93.8214 | 98.6700 | 98.6829 |
| 456 | 98.6257 | 26.0743 | 92.4586 | 98.6471 | 98.6743 |
| 789 | 98.6371 | 30.8171 | 92.6243 | 98.6414 | 98.6214 |
| 2026 | 98.5971 | 43.5786 | 91.0986 | 98.6357 | 98.7157 |

## Table S4 DBpedia large-6

Metric: accuracy.

| Seed | Stable | Single bad | Late | Rescue | Corrected |
|--:|--:|--:|--:|--:|--:|
| 42 | 98.7943 | 26.7257 | 87.9757 | 98.6714 | 98.6500 |
| 123 | 98.7500 | 26.4714 | 61.8986 | 98.6700 | 98.4586 |
| 456 | 98.8314 | 24.8686 | 86.1586 | 98.5857 | 98.5629 |
| 789 | 98.7529 | 32.0700 | 83.8486 | 98.6686 | 98.5186 |
| 2026 | 98.8314 | 28.8814 | 90.7957 | 98.6043 | 98.7071 |

## Table S5 AG News distilbert

Metric: macro-F1.

| Seed | Safe | Single bad | Late | Rescue |
|--:|--:|--:|--:|--:|
| 42 | 94.3634 | 94.3483 | 94.4156 | 94.4007 |
| 123 | 94.2930 | 94.2683 | 94.2816 | 94.3068 |
| 456 | 94.5304 | 94.5050 | 94.4923 | 94.4515 |
| 789 | 94.3067 | 94.3208 | 94.3334 | 94.3058 |
| 2026 | 94.4129 | 94.4671 | 94.3752 | 94.5189 |

## Table S6 DBpedia distilbert

Metric: macro-F1.

| Seed | Safe | Single bad | Late | Rescue |
|--:|--:|--:|--:|--:|
| 42 | 99.2814 | 99.2958 | 99.2886 | 99.3000 |
| 123 | 99.3056 | 99.2999 | 99.2870 | 99.2970 |
| 456 | 99.3100 | 99.3043 | 99.3000 | 99.3115 |
| 789 | 99.2857 | 99.2943 | 99.2799 | 99.2758 |
| 2026 | 99.2899 | 99.2971 | 99.3086 | 99.2713 |

## Table S7 WNUT-17 distilbert

Metric: entity F1.

| Seed | Safe | Single bad | Late | Rescue |
|--:|--:|--:|--:|--:|
| 42 | 43.2463 | 43.5912 | 43.4635 | 43.4635 |
| 123 | 42.4561 | 42.2650 | 42.3077 | 42.3077 |
| 456 | 39.6135 | 40.2644 | 39.9036 | 39.9036 |
| 789 | 43.5227 | 43.3638 | 43.2895 | 43.2895 |
| 2026 | 42.5532 | 42.9630 | 42.5873 | 42.5873 |

## Table S8 CIFAR-10 deit-tiny

Metric: accuracy.

| Seed | Safe | Single bad | Late | Rescue |
|--:|--:|--:|--:|--:|
| 42 | 96.2900 | 96.3000 | 96.2000 | 96.4100 |
| 123 | 96.3000 | 96.1000 | 96.2600 | 95.9700 |
| 456 | 95.9900 | 96.0500 | 96.0900 | 96.0600 |
| 789 | 96.0900 | 96.1700 | 96.1700 | 96.0600 |
| 2026 | 96.2700 | 96.2200 | 95.8300 | 96.2800 |

## Table S9 Paired differences AG News medium-4

All differences are percentage points. The repair and remaining-delay columns are exploratory.

| Seed | Stable - bad | Rescue - late | Corrected - rescue |
|--:|--:|--:|--:|
| 42 | +41.35526 | +23.21053 | +0.89474 |
| 123 | +55.73684 | +10.01316 | +0.17105 |
| 456 | +47.48684 | +29.14474 | -0.10526 |
| 789 | +41.13158 | +11.82895 | -0.42105 |
| 2026 | +50.26316 | +20.73684 | -0.05263 |

## Table S10 Paired differences AG News large-6

All differences are percentage points. The repair and remaining-delay columns are exploratory.

| Seed | Stable - bad | Rescue - late | Corrected - rescue |
|--:|--:|--:|--:|
| 42 | +44.19737 | +23.35526 | +0.27632 |
| 123 | +57.50000 | +13.56579 | +0.09211 |
| 456 | +55.34211 | +32.03947 | -0.43421 |
| 789 | +47.82895 | +30.60526 | +0.11842 |
| 2026 | +41.96053 | +19.73684 | -0.27632 |

## Table S11 Paired differences DBpedia medium-4

All differences are percentage points. The repair and remaining-delay columns are exploratory.

| Seed | Stable - bad | Rescue - late | Corrected - rescue |
|--:|--:|--:|--:|
| 42 | +61.50571 | +3.99000 | +0.05857 |
| 123 | +70.75000 | +4.84857 | +0.01286 |
| 456 | +72.55143 | +6.18857 | +0.02714 |
| 789 | +67.82000 | +6.01714 | -0.02000 |
| 2026 | +55.01857 | +7.53714 | +0.08000 |

## Table S12 Paired differences DBpedia large-6

All differences are percentage points. The repair and remaining-delay columns are exploratory.

| Seed | Stable - bad | Rescue - late | Corrected - rescue |
|--:|--:|--:|--:|
| 42 | +72.06857 | +10.69571 | -0.02143 |
| 123 | +72.27857 | +36.77143 | -0.21143 |
| 456 | +73.96286 | +12.42714 | -0.02286 |
| 789 | +66.68286 | +14.82000 | -0.15000 |
| 2026 | +69.95000 | +7.80857 | +0.10286 |

## Selection and provenance

Source manifest: tables/manuscript_table_sources.json. The all_condition_results.csv file records the exact result member, selection metric and selected epoch for every job. The paired_per_seed.csv file records both source members and realized optimizer budgets. Pretrained selection is also exported in pretrained_selection_and_primary_metric.csv. Selection scores and epochs agree with the saved histories. Checkpoint weights are absent from the archive.

The five-seed paired summaries use sample SD and descriptive 95% Student-t intervals. A two-sided exact sign test has p=0.0625 for five same-direction differences; this does not meet an unadjusted 0.05 threshold. No equivalence margin was prespecified. The historical three-seed sweep is a distinct protocol.
