MATLAB code for statistical analysis with low-frequency diffusion data (in a divergence form model).

Author: Matteo Giordano, https://matteogiordano.weebly.com, and Sven Wang, https://sven-wang.weebly.com.

This repository is associated with the article "Statistical algorithms for low-frequency diffusion data: A PDE approach" by Matteo Giordano and Sven Wang. The abstract of the paper is:

"We consider the problem of making nonparametric inference in multi-dimensional diffusion models from low-frequency data. Statistical analysis in this setting is notoriously challenging due to the intractability of the likelihood and its gradient, and computational methods have thus far largely resorted to expensive simulation-based techniques. In this article, we propose a new computational approach which is motivated by PDE theory and is built around the characterisation of the transition densities as solutions of the associated heat (Fokker-Planck) equation. Employing optimal regularity results from the theory of parabolic PDEs, we prove a novel characterisation for the gradient of the likelihood. Using these developments, for the nonlinear inverse problem of recovering the diffusivity (in divergence form models), we then show that the numerical evaluation of the likelihood and its gradient can be reduced to standard elliptic eigenvalue problems, solvable by powerful finite element methods. This enables the efficient implementation of a large class of statistical algorithms, including (i) preconditioned Crank-Nicolson and Langevin-type methods for posterior sampling, and (ii) gradient-based descent optimisation schemes to compute maximum likelihood and maximum-a-posteriori estimates. We showcase the effectiveness of these methods via extensive simulation studies in a nonparametric Bayesian model with Gaussian process priors. Interestingly, the optimisation schemes provided satisfactory  numerical recovery while exhibiting rapid convergence towards stationary points  despite the problem nonlinearity; thus our approach may lead to significant computational speed-ups."

This repository contains the MATLAB code to replicate the numerical simulation study presented in the article. It contains the following:

1. GenerateObservations.m, code to generate the observations (discrete-time observations of a stochastic diffusion process).
2. SeriesDiscretisation.m, code to implement the parameter space discretisation via the Laplacian eigenpairs.
3. pCNSeries.m, code to implement posterior inference with Gaussian series priors defined Laplacian eigenbasis via the pCN algorithm for posterior samping. It requires the output of GenerateObservations.m and SeriesDiscretisation.m.
4. ULA.m, code to implement posterior inference with Gaussian series priors defined Laplacian eigenbasis via the ULA for posterior samping. It requires the output of GenerateObservations.m and SeriesDiscretisation.m.
5. GradDescent.m, code to compute the MAP estimate associated to Gaussian series priors defined Laplacian eigenbasis via the gradient descent algorithm. It requires the output of GenerateObservations.m and SeriesDiscretisation.m.
6. pCNMatern.m, code to implement posterior inference with the Matern process prior via the pCN algorithm for posterior samping. It requires the output of GenerateObservations.m and SeriesDiscretisation.m and the auxiliary function K_mat.m.
7. K_mat.m, auxiliary code for the Matérn covariance kernel, required by pCNMatern.m.

For questions or for reporting bugs, please e-mail Matteo Giordano (matteo.giordano@unito.it)

Please cite the following article if you use this repository in your research: Giordano, M., and Wang, S. (2024). Statistical algorithms for low-frequency diffusion data: A PDE approach.
