# Prior Availability in Industrial Visual Sim-to-Real

This repository supports the review article:

**Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes**  
Chenxi Tao and Seung-Kyum Choi  
George W. Woodruff School of Mechanical Engineering, Georgia Institute of Technology

## Overview

Industrial visual sim-to-real is often treated as a synthetic-to-real image-transfer problem. This review instead organizes the field by **prior availability**: what evidence is available to ground the deployment decision?

The review distinguishes three regimes:

- **CAD-guided regimes**, where explicit object geometry supports rendering, calibration, pose estimation, segmentation, and test-time verification.
- **CAD-unavailable regimes**, where geometry is replaced by normal-reference appearance, feature distributions, teacher-student residuals, synthetic anomaly assumptions, foundation features, or vision-language priors.
- **Boundary-prior regimes**, where approximate models, templates, reference views, prompts, or semantic correspondences preserve only part of the CAD role.

## Empirical Anchors

The paper uses representative empirical anchors rather than a single cross-task leaderboard:

- **T-LESS/BOP** for CAD-guided object detection, pose, and render-and-compare verification.
- **MVTec AD** for CAD-unavailable industrial anomaly detection and segmentation.
- **VisA** for CAD-unavailable anomaly detection under more varied object and defect categories.

These anchors are used to show why CAD-guided and CAD-unavailable settings require different metrics and evaluation logic.

## Keywords

industrial visual sim-to-real; prior availability; CAD-guided vision; CAD-unavailable inspection; boundary priors; 6D object pose estimation; industrial anomaly detection; render-and-compare verification; domain gap

## Citation

```bibtex
@article{tao2026prioravailability,
  title={Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes},
  author={Tao, Chenxi and Choi, Seung-Kyum},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

Replace `XXXX.XXXXX` after arXiv assigns the identifier.

## Contact

Chenxi Tao: ctao40@gatech.edu  
Seung-Kyum Choi: schoi@me.gatech.edu
