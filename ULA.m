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
%   1.2 ULA initialisation
%
% 2. Bayesian nonparametric inference from low frequency diffusion data with 
%    truncated Gaussian series priors via the ULA for posterior sampling
%
%   2.1 ULA
%   2.2 Results
%
% Requires (part of) the output of GenerateObservations.m and
% SeriesDiscretisation.m

%%
% 1. Prior and initialisation

%%
% 1.1 Prior specification

%%
% Barycenter values of basis elements for faster construction of directions

e_basis_bary=zeros(mesh_elements_num,J_basis);
for j=1:J_basis
    e_basis_bary(:,j) = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),...
            e_basis(:,j)',barycenters(1,:),barycenters(2,:));
end

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
% 1.2 ULA initialisation

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
%    truncated Gaussian series priors via the ULA for posterior sampling

%%
%   2.1 ULA

%%
% Run ULA

rng(1)

stepsize=.000025;
ULA_length=10000;
ULA_theta=zeros(J_basis,ULA_length);
    % initialises vector to store the MCMC samples of theta
ULA_theta(:,1)=theta_init;
    % initialisation point
ULA_pdf=zeros(1,ULA_length-1);
    % vector to store the posterior densities along the ULA
ULA_loglik=zeros(1,ULA_length-1);
    % vector to store the loglikelihood along the ULA
inv_prior_cov=diag(lambdas_basis.^(2*prior_regularity))/prior_variance;
    % inverse prior covariance matrix
Gauss_incr=mvnrnd(zeros(J_basis,1),eye(J_basis,J_basis),ULA_length)';
    % samples independent Brownian increments

tic 

for ULA_step=1:(ULA_length-1)
    %disp(['ULA step n. ', num2str(ULA_step)])

    F_current_mesh_prior=(e_basis*ULA_theta(:,ULA_step))';
    F_current_bary = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),...
            F_current_mesh_prior,barycenters(1,:),barycenters(2,:));


    % Solve elliptic eigenvalue problem for the conductivity f_current,
    % normalise eigenfunction, compute their gradient and the pointwise
    % products
    F_current_fun=@(location,state) griddata(mesh_nodes_prior(1,:), ...
        mesh_nodes_prior(2,:),F_current_mesh_prior,location.x,location.y);
    f_current_fun=@(location,state) f_min+exp(F_current_fun(location,state));
    specifyCoefficients(model,'m',0,'d',1,'c',f_current_fun,'a',0,'f',1); 
    results = solvepdeeig(model,range); 
    lambdas = results.Eigenvalues; 
    J = length(lambdas);
    u = results.Eigenvectors; 
    u(:,1) = 1/sqrt(vol); 
    u(:,2:J) = u(:,2:J)*sqrt(mesh_nodes_num/vol);
    u_obs=u(index_obs_mesh(1:sample_size-1),:);
    u_next_obs=u(index_obs_mesh(2:sample_size),:);
    Dxu = zeros(J,mesh_elements_num); 
    Dyu = zeros(J,mesh_elements_num);
    for j=2:J
        [Dxu(j,:),Dyu(j,:)] = pdegrad(mesh_nodes,mesh_elements,u(:,j));
    end

    % Compute the transition densities
    trds=u_obs.*u_next_obs*exp(-deltaT_obs*lambdas);

    % Compute the posterior density of current gradient descent step
    ULA_loglik(ULA_step) = sum(log(abs(trds)));
    ULA_pdf(ULA_step)=ULA_loglik(ULA_step) ...
        -.5*ULA_theta(:,ULA_step)'*inv_prior_cov ...
        *ULA_theta(:,ULA_step);
  
    % Compute all the exponential factors, inner products and products of
    % eigenfunctions
    h_bary=repmat(exp(F_current_bary)',1,J_basis).*e_basis_bary;
    exp_fact = deltaT_obs*exp(-deltaT_obs*lambdas).*ones(J,J);
    inn_prods=zeros(J_basis,J,J);
    u_prods=zeros(sample_size-1,J,J);
    Frech_derivs=zeros(sample_size-1,J_basis);
    for j=1:J
        for k=j:J
            if abs(lambdas(j)-lambdas(k))>.001 
               exp_fact(j,k)= exp(-deltaT_obs*lambdas(j))...
               *(exp(deltaT_obs*(lambdas(j)-lambdas(k)))-1)...
               /(lambdas(j)-lambdas(k));
               exp_fact(k,j)=exp_fact(j,k);
            end
            inn_prods(:,j,k) = sum(repmat(mesh_elements_area',1,J_basis).*h_bary ...
                    .*repmat((Dxu(j,:).*Dxu(k,:))',1,J_basis),1) ...
                    + sum(repmat(mesh_elements_area',1,J_basis).*h_bary ...
                    .*repmat((Dyu(j,:).*Dyu(k,:))',1,J_basis),1);
            inn_prods(:,k,j) = inn_prods(:,j,k);
            u_prods(:,j,k) = u_obs(:,j).*u_next_obs(:,k);
            u_prods(:,k,j)=u_prods(:,j,k);
            Frech_derivs = Frech_derivs - exp_fact(j,k)*u_prods(:,j,k)*inn_prods(:,j,k)';
        end
    end
   
    % Compute the gradient
    grad=sum(Frech_derivs./trds)' - inv_prior_cov*ULA_theta(:,ULA_step);

    % ULA update
    ULA_theta(:,ULA_step+1) = ULA_theta(:,ULA_step)+stepsize*grad...
    +sqrt(2*stepsize)*Gauss_incr(:,ULA_step+1);

end

toc

%%
% 2.2 Results

%%
% Plot the posterior density and likelihood along the ULA chain

% Posterior density of ground truth
post_pdf0 = loglik0 -.5*F0_coeff'*inv_prior_cov*F0_coeff;

figure()
axes('FontSize', 20, 'NextPlot','add')
plot(ULA_pdf,'-','LineWidth',5)
yline(post_pdf0,'r','LineWidth',2)
legend('log\pi(\vartheta_m|X^{(n)})','log\pi(\theta_0|X^{(n)})','FontSize',20)
xlabel('ULA step','FontSize',20)

%%
% MCMC average and estimation error

burnin=250;

% Compute MCMC average
theta_mean = mean(ULA_theta(:,burnin+1:ULA_length),2);
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
%title('Posterior mean estimate (via ULA)','FontSize',20)
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
ULA_estim_error = norm(theta_mean-F0_coeff);
disp(['L^2 estim error = ', num2str(ULA_estim_error)])
disp(['L^2 approximation error via projection = ', num2str(approx_error)])
disp(['L^2 norm of f_0 = ', num2str(F0_norm)])