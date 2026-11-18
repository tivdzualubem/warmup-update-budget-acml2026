# Primary-source comparison checked 7 November 2026

This check supports Table R1 and the contribution revision. It does not establish priority or reproduce another study. The 29 October feedback asks for a specific experimental distinction rather than novelty from early-update harm or longer training alone.

| Work and primary source | Scope checked | Consequence here |
|:--|:--|:--|
| Kingma and Ba (2015): https://arxiv.org/abs/1412.6980 | Standard bias-corrected Adam moments | Equation 2 is explanatory, not a new theorem |
| Shazeer and Stern (2018): https://proceedings.mlr.press/v80/shazeer18a.html | Adaptive update clipping | No claim to originate update-size control |
| Gotmare et al. (2019): https://arxiv.org/abs/1810.13243 | Warmup/later-layer changes and Figure 4 freezing comparison | Different intervention; no analogous mechanism measured here |
| Ma and Yarats (2021): https://doi.org/10.1609/aaai.v35i10.17069 | Untuned adaptive-optimizer warmup | Execution fault differs from warmup design |
| Kosson et al. (2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/059445c2d5b3ef918079851628fef1d6-Paper-Conference.pdf | Main update analysis; Appendix A/Figures 6-8 early-update and extended-training experiments | GPT-only shorthand is insufficient; center consumed rate,lag and budget |
| Kalra and Barkeshli (2024): https://arxiv.org/abs/2406.09405 | Conditioning/sharpness regimes and target-rate tolerance | No landscape mechanism identified by our histories |
| Abuduweili and Liu (2025): https://proceedings.mlr.press/v280/abuduweili25a.html | Nonzero second-moment initialization | Paired arms here retain matched zero initial moments |

Each distinction in the paper points to an equation, recorded contrast/history or an explicit lack of mechanism evidence. Recovery beyond the observed positive-LR horizon is unmeasured. Recent primary sources retained from the prior check concern Adam loss spikes, warmup theory and horizon effects; they support bounded background discussion. Repeat the current-source check before an actual venue submission.
