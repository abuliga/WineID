# WineID
parafac2 in python for chromatogram analysis

This repository contains Python code for the automatic alignment of chromatograms. 
It uses the tensorly implementation of parafac2. 
To install tensorly for python, follow instrcutions at http://tensorly.org/stable/installation.html;

To test the code, just run the python scripts. You will have to change the data folder to use the scripts for real data.

* Planned Features (*bold* are implemented)
  * selection of informative UV-Vis spectral zones (user or automatic selection)
  * selection of informative, possibly non-uniform, time resolution (user or automatic selection)
  * time-partitioning of the chromatograms (user or automatic selection)
  * Parafac2 automatic alignment of (partial) chromatograms. Automatic selection of parameters.
  * Parafac2 co-elution solver
  * Fit evaluation with quality metrics (e.g. core consinstency)
  * easy extraction of results for post-processing

* Bibliography: 
  * José Manuel Amigo, Thomas Skov, and Rasmus Bro, ChroMATHography: Solving Chromatographic Issues with Mathematical Models and Intuitive Graphics
  * Sarmento J. Mazivila, Santiago A. Bortolato, Alejandro C. Olivieri, MVC3_GUI: A MATLAB graphical user interface for third-order multivariate calibration. An upgrade including new multi-way models, Chemometrics and Intelligent Laboratory Systems, Volume 173, 2018, Pages 21-29,ISSN 0169-7439,https://doi.org/10.1016/j.chemolab.2017.12.012.
  * Rasmus Bro  Claus A. Andersson  Henk A. L. Kiers, PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model
  * Rasmus Bro, Claus A. Andersson,  Henk A. L. Kiers, PARAFAC2—Part II. Modeling chromatographic data with retention time shifts
  * Jean Kossaifi, Yannis Panagakis, Anima Anandkumar and Maja Pantic, TensorLy: Tensor Learning in Python, Journal of Machine Learning Research, 2019. http://jmlr.org/papers/v20/18-277.html.
  * Lea G. Johnsena,José Manuel Amigoì, Thomas Skov and Rasmus Bro, Automated resolution of overlapping peaks in chromatographic data


