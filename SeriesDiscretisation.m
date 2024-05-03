% Copyright (c) 2024 Matteo Giordano and Sven Wang
%
% Codes accompanying the article "Statistical algorithms for low-frequency diffusion data: A PDE approach" 
% by Matteo Giordano andn Sven Wang

%%
% Contents:
%
% 1. Discretisation of the parameter space via the Dirichlet-Laplacian
%    eigenbasis
%
% 2. Projection of F_0 onto the eigenbasis
%
% Requires (part of) the output of GenerateObservations.m

%%
% 1. Discretisation of the parameter space via the Neumann-Laplacian
%    eigenbasis

%%
% Mesh for computation of the Neumann-Laplacian eigenpairs

model_prior = createpde(); 
geometryFromEdges(model_prior,geom);
    % requires 'geom' from GenerateObservations.m
generateMesh(model_prior,'Hmax',0.075);
mesh_nodes_prior=model_prior.Mesh.Nodes;
    % mesh nodes
mesh_nodes_num_prior=size(mesh_nodes_prior); 
mesh_nodes_num_prior=mesh_nodes_num_prior(2);
mesh_elements_prior = model_prior.Mesh.Elements;
    % triangular mesh elements
mesh_elements_num_prior = size(mesh_elements_prior); 
mesh_elements_num_prior = mesh_elements_num_prior(2); 
    % number of triangular mesh elements
[~,mesh_elements_area_prior] = area(model_prior.Mesh); 
    % area of triangular mesh elements

% Compute barycenters of triangular mesh elements
barycenters_prior = zeros(2,mesh_elements_num_prior);
for i=1:mesh_elements_num_prior
    barycenters_prior(:,i)=mean(mesh_nodes_prior(:,mesh_elements_prior(1:3,i)),2);
end

F0_mesh_prior = F0(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:));
F0_bary_prior=F0(barycenters_prior(1,:),barycenters_prior(2,:));

%%
% Solve elliptic eigenvalue problem for the Neumann-Laplacian

tic

% Specity homogeneous Dirichlet boundary conditions
applyBoundaryCondition(model,'neumann','Edge',...
    1:model.Geometry.NumEdges,'g',0,'q',0); 
% Specify coefficients for eigenvalue equation
specifyCoefficients(model_prior,'m',0,'d',1,'c',1,'a',0,'f',1);
range_prior = [-1,750]; 
    % range to search for eigenvalues
results = solvepdeeig(model_prior,range_prior); 
    % solve eigenvalue equation
lambdas_basis = results.Eigenvalues; 
    % extract eigenvalues
J_basis = length(lambdas_basis); 
    % number of eigenvalues (dimension of discretised parameter space)
e_basis = results.Eigenvectors; 
    % extract eigenfunctions

toc

% Normalisation of eigenfunction
u0(:,1) = 1/sqrt(vol); 
e_basis(:,1:J_basis) = e_basis(:,1:J_basis)*sqrt(mesh_nodes_num_prior/vol);

%%
% 2. Projection of F_0 onto the eigenbasis

F0_coeff=zeros(J_basis,1); 
    % initialises vector to store the Fourier coefficients of F_0 in the 
    % Neumann-Laplacian eigenbasis

% Compute the Fourier coefficients of F_0
for j=1:J_basis
    ej_basis_interp=scatteredInterpolant(mesh_nodes_prior(1,:)',...
        mesh_nodes_prior(2,:)',e_basis(:,j));
    ej_basis_bary_prior=ej_basis_interp(barycenters_prior(1,:),...
        barycenters_prior(2,:));
    F0_coeff(j)=sum(mesh_elements_area_prior.*F0_bary_prior.*ej_basis_bary_prior);
end

% Compute projection of F_0
F0_proj=zeros(1,mesh_nodes_num_prior);
for j=1:J_basis
    F0_proj = F0_proj+F0_coeff(j)*e_basis(:,j)';
end

% Plot F_0, its projection and the approximation error
figure()
pdeplot(model_prior,'XYData',F0_mesh_prior,'ColorMap',jet)
title('True F_0','FontSize',20)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

figure()
pdeplot(model_prior,'XYData',F0_proj,'ColorMap',jet)
title('Projection of F_0','FontSize',20)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

figure()
pdeplot(model_prior,'XYData',F0_mesh_prior-F0_proj,'ColorMap',hot)
title('Approximation error','FontSize',20)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

% Compute piecewise constant approximations of the projection of F_0 at
% the triangle baricenters
F0_proj_bary_prior = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),F0_proj,...
    barycenters_prior(1,:),barycenters_prior(2,:));

% Approximate L^2 distance between F_0 and its projection
approx_error = sqrt(sum((F0_bary_prior-F0_proj_bary_prior).^2.*mesh_elements_area_prior));
disp(['L^2 approximation error via projection = ', num2str(approx_error)])

% L^2 norm of f0
F0_norm=norm(F0_coeff);
disp(['L^2 norm of f_0 = ', num2str(F0_norm)])