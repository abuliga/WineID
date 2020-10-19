# WineID
parafac2 in python for chromatogram analysis

This repository contains Python code for the automatic alignment of chromatograms. 
At this time, it uses the tensorly implementation of parafac2. 


* Planned Features
  * selection of informative UV-Vis spectral zones (user or automatic selection)
  * selection of informative, possibly non-uniform, time resolution (user or automatic selection)
  * time-partitioning of the chromatograms (user or automatic selection)
  * Parafac2 automatic alignment of (partial) chromatograms. Autonatic selection of parameters.
  * Parafac2 co-elution solver
  * Fit evaluation with quality metrics (core consinstency)
  * easy extraction of results for post-processing

* Bibliography: 
** Sarmento J. Mazivila, Santiago A. Bortolato, Alejandro C. Olivieri, MVC3_GUI: A MATLAB graphical user interface for third-order multivariate calibration. An upgrade including new multi-way models, Chemometrics and Intelligent Laboratory Systems, Volume 173, 2018, Pages 21-29,ISSN 0169-7439,https://doi.org/10.1016/j.chemolab.2017.12.012.
** PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model
** PARAFAC2—Part II. Modeling chromatographic data with retention time shifts
