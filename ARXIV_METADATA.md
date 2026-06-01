# arXiv Metadata Draft

## Title

Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes

## Authors

Chenxi Tao, Seung-Kyum Choi

## Affiliations

George W. Woodruff School of Mechanical Engineering, Georgia Institute of Technology

## Corresponding Author

Seung-Kyum Choi, schoi@me.gatech.edu

## Suggested arXiv Categories

Primary category:

- `cs.CV` - Computer Vision and Pattern Recognition

Secondary categories:

- `cs.AI` - Artificial Intelligence
- `cs.RO` - Robotics

Notes:

- The submitted arXiv version uses `cs.CV` as the primary category, with `cs.AI` and `cs.RO` as cross-lists.

## Comments Field

Review article; 103 references; 9 main figures; empirical anchors on T-LESS/BOP, MVTec AD, and VisA.

## arXiv and DOI

- arXiv: `2605.30581`
- arXiv URL: <https://arxiv.org/abs/2605.30581>
- PDF: <https://arxiv.org/pdf/2605.30581>
- DOI: <https://doi.org/10.48550/arXiv.2605.30581>

## Abstract

Industrial visual sim-to-real is often described as transferring from synthetic images to real images, but industrial deployment usually involves a broader mismatch between available evidence and required decisions. A system may be built from CAD renderings, simulated RGB-D observations, normal reference images, synthetic defects, pretrained feature spaces, or language prompts, yet deployed under different sensors, lighting, materials, fixtures, calibration, production variation, and rare defect modes. This review reframes industrial visual sim-to-real as a domain-gap problem organized by prior availability. We distinguish CAD-available settings, where explicit object geometry can support rendering, calibration, pose estimation, segmentation, and test-time geometric verification; CAD-unavailable settings, where geometry is replaced by normal-reference appearance, feature distributions, teacher-student residuals, synthetic anomaly assumptions, foundation features, or vision-language priors; and boundary-prior settings, where approximate models, templates, reference views, or semantic correspondences preserve only part of the CAD role. This framing connects CAD-based detection and 6D pose-estimation literature with industrial anomaly and surface-inspection literature that is usually reviewed separately. To make the taxonomy concrete, we use empirical anchors on T-LESS/BOP, MVTec AD, and VisA. The anchors show that CAD render count alone does not close transfer; source-distribution design, detector capacity, and small real calibration can matter more. They also show that CAD at test time creates a distinct verification channel through mask, pose, and depth consistency, whereas CAD-unavailable inspection relies on calibrated normality and feature deviation. The review therefore argues against a single cross-task leaderboard and instead asks what prior grounds the deployment decision.

## Keywords

- industrial visual sim-to-real
- prior availability
- CAD-guided vision
- CAD-unavailable inspection
- boundary priors
- 6D object pose estimation
- industrial anomaly detection
- render-and-compare verification
- domain gap

## Suggested Short Description

A review of industrial visual sim-to-real organized by the prior available at deployment: CAD-guided geometry, CAD-unavailable inspection, and boundary-prior regimes.

## Plain-Language Summary

Industrial vision systems often rely on evidence that differs from real deployment conditions. This review argues that the most important distinction is what prior is available: a CAD model that can render and verify geometry, normal images and features that replace geometry, or weaker boundary priors such as templates and reference views. The paper connects CAD-based detection and pose-estimation methods with CAD-unavailable anomaly-inspection methods and uses empirical anchors to show why these regimes require different metrics and evaluation logic.

## BibTeX Template

```bibtex
@article{tao2026prioravailability,
  title={Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes},
  author={Tao, Chenxi and Choi, Seung-Kyum},
  journal={arXiv preprint arXiv:2605.30581},
  doi={10.48550/arXiv.2605.30581},
  eprint={2605.30581},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  year={2026}
}
```
