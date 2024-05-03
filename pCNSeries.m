% Copyright (c) 2024 Matteo Giordano and Sven Wang
%
% Codes accompanying the article "Statistical algorithms for low-frequency diffusion data: A PDE approach" 
% by Matteo Giordano andn Sven Wang

%%
% Contents:
%
% 1. Prior and initialisation
%
%   1.1 Prior specification
%   1.2 pCN initialisation
%
% 2. Bayesian nonparametric inference from low frequency diffusion data with 
%    truncated Gaussian series priors via the pCN algorithm for posterior 
%    sampling
%
%   2.1 pCN algorithm
%   2.2 Results
%
% Requires (part of) the output of GenerateObservations.m and
% SeriesDiscretisation.m

%%
% 1. Prior and initialisation

%%
% 1.1 Prior specification

%%
% Prior covariance matrix for Gaussian series prior draws

prior_regularity=1;
prior_variance=500;
prior_cov=zeros(J_basis,J_basis);
for j=1:J_basis
    prior_cov(j,j)=prior_variance*lambdas_basis(j)^(-2*prior_regularity);
end

%%
% Sample and plot a prior draw

theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)'; 
    % sample Fourier coefficients from prior

F_rand=zeros(1,mesh_nodes_num_prior);
for j=1:J_basis
    F_rand = F_rand+theta_rand(j)*e_basis(:,j)';
end

figure()
axes('FontSize', 13, 'NextPlot', 'add');
pdeplot(model_prior,'XYData',F_rand,'ColorMap',jet);
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

%%
%   1.2 pCN initialisation

%%
% choice of the initialiser

% Initialisation at 0
theta_init = zeros(J_basis,1);

% Initialisation at F_0
%theta_init = F0_coeff;

% Challenging initialisation at -F_0
%theta_init = -F0_coeff;

% Random initialisation Initialisation at a prior draw
%rng(1)
%theta_init = mvnrnd(zeros(J_basis,1),prior_cov,1)';

F_init_mesh_prior=zeros(1,mesh_nodes_num_prior);
for j=1:J_basis
    F_init_mesh_prior = F_init_mesh_prior+theta_init(j)*e_basis(:,j)';
end
f_init_mesh_prior=f_min + exp(F_init_mesh_prior);

% Specify f_init as a function of (location,state) to pass to elliptic 
% PDE solver
F_init_fun=@(location,state) griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),...
    F_init_mesh_prior,location.x,location.y);
f_init_fun=@(location,state) f_min+exp(F_init_fun(location,state));

tic

% Find Neumann eigenpairs for initial pCN state
specifyCoefficients(model,'m',0,'d',1,'c',f_init_fun,'a',0,'f',1); 
results = solvepdeeig(model,range);
lambdas_init = results.Eigenvalues;
J_init=length(lambdas_init);
u_init = results.Eigenvectors;
u_init(:,1) = 1/sqrt(vol);
u_init(:,2:J_init)=u_init(:,2:J_init)...
    *sqrt(mesh_nodes_num/vol);
toc

% Likelihood of initialisation point
loglik_init = 0;
for m=1:(sample_size-1) 
    trpdf = 0;
    for j=1:J_init
        trpdf = trpdf ... 
        + exp(-deltaT_obs*lambdas_init(j))*u_init(index_obs_mesh(m),j)...
        *u_init(index_obs_mesh(m+1),j);
    end
    loglik_init = loglik_init + log(abs(trpdf));
end
disp(['Log-likelihood of pCN initialisation = ', num2str(loglik_init)])
disp(['Log-likelihood of f0 = ', num2str(loglik0)])

%%
% 2. Bayesian nonparametric inference from low frequency diffusion data with 
%    truncated Gaussian series priors via the pCN algorithm for posterior 
%    sampling

%%
% 2.1 pCN algorithm

%%
% Run pCN

rng(1)
    % sets the seed

tic

% Parameters
stepsize=.0001; 
    % smaller stepsizes imply shorter moves for the pCN proposal
pCN_length=25000; 
    % number of pCN draws

% Auxiliary variables
pCN_acc_count = zeros(1,pCN_length); 
    % tracker of acceptance steps
unifs = rand(1,pCN_length); 
    % i.i.d. Un(0,1) random variables for Metropolis-Hastings updates
pCN_theta = zeros(J_basis,pCN_length); 
    % initialise matrix to store the pCN chain of Fourier coefficients
pCN_theta(:,1)=theta_init;
    % set pCN initialisation point
pCN_loglik = zeros(1,pCN_length); 
    % initialise tracker of the loglikelihood of the pCN chain
pCN_loglik(1) = loglik_init; 
    % set initial loglikelihood
loglik_current = pCN_loglik(1);
theta_current = pCN_theta(:,1);
u_current=u_init;

% pCN chain
for MCMC_index=2:pCN_length
    disp(["pCN step n. " num2str(MCMC_index)])
    
    % Construct pCN proposal
    theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)';
    theta_proposal=sqrt(1-2*stepsize)*theta_current+sqrt(2*stepsize)*theta_rand;
    F_proposal_mesh_prior=zeros(mesh_nodes_num_prior,1);
    for j=1:J_basis
        F_proposal_mesh_prior = F_proposal_mesh_prior+theta_proposal(j)*e_basis(:,j);
    end
        
    % Construct proposal diffusivity function to pass to PDE solver
    F_proposal_fun=@(location,state) griddata(mesh_nodes_prior(1,:),...
        mesh_nodes_prior(2,:),F_proposal_mesh_prior,location.x,location.y);
    f_proposal_fun=@(location,state) f_min+exp(F_proposal_fun(location,state));

    % Solve Neumann eigenvalue problem corresponding to proposal
    specifyCoefficients(model,'m',0,'d',1,'c',f_proposal_fun,'a',0,'f',1); 
    results = solvepdeeig(model,range);
    lambdas_proposal = results.Eigenvalues;
    J_proposal=length(lambdas_proposal);
    u_proposal = results.Eigenvectors;
    u_proposal(:,1) = 1/sqrt(vol);
    u_proposal(:,2:J_proposal)=u_proposal(:,2:J_proposal)...
        *sqrt(mesh_nodes_num/vol);
    
    % Compute log-likelihood of proposal
    loglik_proposal = 0;
    for m=1:(sample_size-1)
        trpdf = 0;
        for j=1:J_proposal
            trpdf = trpdf ... 
            + exp(-deltaT_obs*lambdas_proposal(j))*...
            u_proposal(index_obs_mesh(m),j)*u_proposal(index_obs_mesh(m+1),j);
        end
        loglik_proposal = loglik_proposal + log(abs(trpdf));
    end
    
    % Compute acceptance probability
    alpha=exp(loglik_proposal-loglik_current);

    % Metropolis-Hastings update
    if unifs(MCMC_index)<alpha % accept proposal
        theta_current = theta_proposal;
        loglik_current = loglik_proposal;
        pCN_acc_count(1,MCMC_index)=pCN_acc_count(1,MCMC_index-1)+1;
        u_current = u_proposal;
    else
        pCN_acc_count(1,MCMC_index)=pCN_acc_count(1,MCMC_index-1);
    end
    pCN_theta(:,MCMC_index) = theta_current;
    pCN_loglik(MCMC_index) = loglik_current;

end

toc

%%
% 2.2 Results

%%
% Acceptance ratio and loglikelihood along the MCMC chain

% Acceptance ratio
accept_ratio=pCN_acc_count./(1:pCN_length);
figure()
axes('FontSize', 20, 'NextPlot', 'add');
plot(accept_ratio(1:pCN_length),'-','LineWidth',5)
yline(.3,'r','LineWidth',2)
legend('Acceptance ratio','Fontsize',20)
xlabel('pCN step','FontSize', 20)

% Log-likelihood
figure()
axes('FontSize', 20, 'NextPlot', 'add');
plot(pCN_loglik,'-', 'LineWidth',2)
%title('Loglikelihood along the MCMC chain','FontSize',20)
xlabel('pCN step','FontSize', 20)
yline(loglik0,'r','LineWidth',2)
ylim([loglik0-25,loglik0+25])
legend('Log-likelihood of f_{\vartheta_m}','Log-likelihood of f_0',...
    'Fontsize',20)

%%
% MCMC average and estimation error

burnin=2500;

% Compute MCMC average
theta_mean = mean(pCN_theta(:,(burnin+1):pCN_length),2);
F_mean_mesh_prior=zeros(mesh_nodes_num_prior,1);
for j=1:J_basis
    F_mean_mesh_prior = F_mean_mesh_prior +theta_mean(j)*e_basis(:,j);
end

% Plot MCMC average and estimation erorr
figure()
pdeplot(model_prior,'XYData',F0_mesh_prior,'ColorMap',jet)
%title('True F_0','FontSize',20)
clim([min(F0_mesh_prior),max(F0_mesh_prior)])
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

figure()
pdeplot(model_prior,'XYData',F_mean_mesh_prior,'ColorMap',jet)
%title('Posterior mean estimate (via pCN)','FontSize',20)
clim([min(F0_mesh_prior),max(F0_mesh_prior)])
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20;

figure()
axes('FontSize', 13, 'NextPlot', 'add');
pdeplot(model_prior,'XYData',F0_mesh_prior-F_mean_mesh_prior','ColorMap',hot)
%title('Estimation error','FontSize',20)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20;

% Approximate L^2 distance between f_0 and posterior mean
pCN_estim_error = norm(theta_mean - F0_coeff);
disp(['L^2 estim error = ', num2str(pCN_estim_error)])
disp(['L^2 approximation error via projection = ', num2str(approx_error)])
disp(['L^2 norm of F_0 = ', num2str(F0_norm)])