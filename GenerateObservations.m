% Copyright (c) 2024 Matteo Giordano and Sven Wang
%
% Codes accompanying the article "Statistical algorithms for low-frequency diffusion data: A PDE approach" 
% by Matteo Giordano andn Sven Wang

%%
% Contents:
%
% 1. Discretely-sampled bidimensional reflected diffusion
%
%   1.1 Domain and ground truth
%   1.2 Continuous diffusion path
%   1.3 Low-frequency observations
%
% 2. Computation of the likelihood of the ground truth
%
%   2.1 Neumann elliptic igenvalue problem for the ground truth
%   2.3 Likelihood of the ground truth

%%
% 1. Discretely-sampled bidimensional reflected diffusion
%
% Let O be the disk (in R^2) with radius r. Let X=(X_t, t>=0) be the 
% reflected Markov diffusion process on O arising as the solution of the 
% SDE
%
%   dX_t = grad[f_0(X_t)]dt + sqrt[2f_0(X_t)] I dW_t + v(X_t)dL_t, 
%   X_0  = x_0,
%
% where f:O -> [f_min,+infty), f_min > 0, is the unknown (sufficiently 
% smooth) diffusivity, I is the identity matrix of R^2, (W_t, t >= 0) is a 
% standard Brownian motion in R^2, and (v(X_t)L_t, t >= 0) models reflection at
% the boundary. The diffusion X has generator
%
%   L_{f_0}[phi] = div[f_0 grad[phi]]
%
% and invariant distribution equal to U(O), the uniform distribution on O.
%
% We observe disctretely sampled values of the diffusion X at low 
% frequency:
%
%   X^{(n)} = (X_0,X_{deltaT_obs},...,X_{ndeltaT_obs})
%
% where n >=1 and deltaT_obs>0 is the fixed time interval between subsequent
% observations
%
% The following code generates n observations (X_0,X_{deltaT_obs},...,
% X_{ndeltaT_obs}) and evaluates the likelihood L_n(f_0) of f_0, given by
% the product of the transition densties
%
%   p_{deltaT_obs,f_0}(X_{(i-1)deltaT_obs},X_{ideltaT_obs}), i=1,...,n.
%

%%
%   1.1 Domain and ground truth

%%
% Specify the domain and create a triangular mesh

% Display more digits
format long

% Specify radius of disk
r = 1/sqrt(pi); 
    % chosen to normalise first Neumann-Laplacian eigenfunction to 1
vol=pi*r^2;
    % volume of disk

% Create PDE model
model = createpde();

% Define circular domain O with centre (0,0) and radius r
O = [1,0,0,r]';

% Create geometry
geom = decsg(O); 
geometryFromEdges(model,geom);

% Create mesh
generateMesh(model,'Hmax',0.05);
mesh_nodes = model.Mesh.Nodes; 
    % matrix whose columns contain the (x,y) coordinates of the mesh nodes
mesh_nodes_num = size(mesh_nodes); mesh_nodes_num=mesh_nodes_num(2); 
    % number of nodes
mesh_elements = model.Mesh.Elements; 
    % matrix whose columns contain the 6 node indices identifying each triangle
mesh_elements_num = size(mesh_elements); 
mesh_elements_num = mesh_elements_num(2); 
    % number of triangles
[~,mesh_elements_area] = area(model.Mesh);

% Compute barycenters of triangles
barycenters = zeros(2,mesh_elements_num);
for i=1:mesh_elements_num
    barycenters(:,i) = mean(mesh_nodes(:,mesh_elements(1:3,i)),2);
end

% Plot mesh
%figure()
%pdeplot(model,'ElementLabels','on') 
    % plots triangular mesh and displays elements names
%hold on
%pdemesh(mesh_nodes,mesh_elements(:,1),'EdgeColor','green') 
    % highlights first triangle in the mesh
%figure()
%pdeplot(model,'ElementLabels','on','NodeLabels','on') 
    % plots triangular mesh and displays nodes and triangles names
%hold on
%plot(mesh_nodes(1,:),mesh_nodes(2,:),'ok','MarkerFaceColor','blue') 
    % plots the nodes in the mesh
%plot(barycenters(1,:),barycenters(2,:),'ok','Marker','*',...
% 'Color','black','LineWidth',1) 
    % plots barycenters

%%
% Specify true diffusivity f_0

% Specify f_0 and its gradient
f_min=0.1;
f0 = @(x,y) f_min +1 +10*exp(-(7.25*x-1.5).^2-(7.25*y-1.5).^2) ...
   + 10*exp(-(7.25*x+1.5).^2-(7.25*y+1.5).^2);
f0_mesh = f_min + 1 + 10*exp(-(7.25*mesh_nodes(1,:)-1.5).^2 ...
    - (7.25*mesh_nodes(2,:)-1.5).^2) ...
    + 10*exp(-(7.25*mesh_nodes(1,:)+1.5).^2-(7.25*mesh_nodes(2,:)+1.5).^2);
F0 = @(x,y) log(f0(x,y)-f_min);
F0_mesh=log(f0_mesh-f_min);
Gradf0 = @(x,y) -145*exp(-(7.25*x-1.5).^2-(7.25*y-1.5).^2)...
    *[7.25*x-1.5,7.25*y-1.5]...
    -145*exp(-(7.25*x+1.5).^2-(7.25*y+1.5).^2)...
    *[7.25*x+1.5,7.25*y+1.5];

% Plot f0
figure()
pdeplot(model,'XYData',f0_mesh,'ColorMap',jet)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

% Plot F0
figure()
pdeplot(model,'XYData',F0_mesh,'ColorMap',jet)
axis equal
colorbar('Fontsize',20)
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

%%
%   1.2 Continuous diffusion path

%%
% Reflected continuous sample paths simulation

rng(1)
    % set the seed
    
tic 
    % tracker of run time

% Discretisation of time interval for simulation
simulation_size = 5e8; 
deltaT_sim = .000005; 
    % time interval between simulated values
T = simulation_size*deltaT_sim; 
    % time horizon for simulation

% Generate Euler Euler–Maruyama approximations
x_0 = [0,0]; 
%randrho = r*rand(); 
    % random radius for initial condition
%randtheta = 2*pi*rand(); 
    % random angle for initial condition
%x_0 = [randrho*cos(randtheta),randrho*sin(randtheta)]; 
    % uniformly distributed initial condition
X = zeros(2,simulation_size);
X(:,1) = x_0;
        % initialisation of diffusion path

for i=2:simulation_size
    X_old = X(:,i-1);

    % Sample candidate for next observation
    X_new = X_old + Gradf0(X_old(1),X_old(2))'*deltaT_sim...
        + sqrt(2*f0(X_old(1),X_old(2)))...
        *mvnrnd(zeros(2,1),deltaT_sim*eye(2))';

    % If new point X_next is outside O (disk of radius r) use boundary 
    % reflection until the new point is inside
    while norm(X_new)>r 
        coefficients=polyfit([X_new(1), X_old(1)],[X_new(2), X_old(2)], 1); 
            % line passing through X_old and X_new (which is outside disk)
        slope = coefficients(1);
        intercept = coefficients(2);
        [xout,yout] = linecirc(slope,intercept,0,0,r); 
            % intrsection of the line with circle
        d1=sqrt((xout(1)-X_new(1))^2+(yout(1)-X_new(2))^2); 
        d2=sqrt((xout(2)-X_new(1))^2+(yout(2)-X_new(2))^2); 
            % as there are two intersections, calculate the distance from 
            % X_new to these intersections and choose the closest one
        if d1<d2
            xintersec(1)=xout(1);
            xintersec(2)=yout(1);
        else
            xintersec(1)=xout(2);
            xintersec(2)=yout(2);
        end
        
        % Find reflected point
        direction=X_new-xintersec'; 
            % vector from the intersection point to X_new
        C = xintersec(1)/r; S = xintersec(2)/r;
        rotmatr=[C, S;-S, C];
            % rotation matrix associated to xinteresec
        dir_refl=rotmatr*direction;
            % direction vector in the coordinate system of xinteresec
        dir_refl=[-dir_refl(1);dir_refl(2)];
            % reflected direction vector in the coordinate system of 
            % xinteresec
        invrotmatr=[C, -S;S, C];
            % inverse rotation matrix associated to xintersec
        dir_refl=invrotmatr*dir_refl;
            % reflected direction vector in the original coordinate system
        X_refl=xintersec'+dir_refl;
            % reflected point
        
        X_new = X_refl;
    end

     X(:,i)=X_new;
end

toc

% Plot continuous diffusion path
figure()
axes('FontSize', 20, 'NextPlot','add')
plot(X(1,1:min(simulation_size,2.5e5)),X(2,1:min(simulation_size,2.5e5)))
axis equal
plot(X(1,1),X(2,1),'Marker','*','Color','red','LineWidth',2)
hold on
ang=0:0.001:2*pi; 
xcirc=r*cos(ang);
ycirc=r*sin(ang);
plot(xcirc,ycirc,'LineWidth',2);
legend('X_t','X_0','FontSize',20)
axis equal
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 

%%
% 1.3. Low-frequency observations

%%
% Discretely sampled observations and projection onto mesh for faster
% evaluation of the transition densities

% Sample diffusion at discrete time intervals
deltaT_obs = .05; 
    % time interval between observations
deltaT_ratio = floor(deltaT_obs/deltaT_sim);
max_sample_size = floor(T/deltaT_obs); 
    % maximal possible number of discretely sampled values
sample_size = max_sample_size;
    % number of discretely sampled values, must be <= max_sample_size

disp(['Sample size = ', num2str(sample_size)])

index_obs = 1 + deltaT_ratio*(0:sample_size-1); 
    % indices of observations
X_obs = X(:,index_obs);
    % discretely-sampled observations

% Projection of observations onto mesh by finding closest points
index_obs_mesh = findNodes(model.Mesh,'nearest',X_obs); 
    % indices of mesh elements closest to observed diffusion values
X_obs_mesh = mesh_nodes(:,index_obs_mesh); 
    % projection of observations onto mesh

% Plot observations
figure()
hold on
plot(X_obs(1,1:min([length(X_obs),sample_size,250])),...
    X_obs(2,1:min([length(X_obs),sample_size,250])),'.-');
plot(X_obs(1,1),X_obs(2,1),'Marker','*','Color','red','LineWidth',2)
plot(xcirc,ycirc,'Color','#EDB120','Linewidth',2);
legend('X_{iD}','X_0','FontSize',20)
axis equal
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20; 
hold off

%%
% 2. Computation of the likelihood of the ground truth

%%
% 2.1. Neumann elliptic igenvalue problem for the ground truth

%%
% Solve elliptic eigenvalue problem for the true conductivity f0

% Specify f_0 as a function of (location,state) to pass to elliptic PDE solver
f0_fun=@(location,state) f0(location.x,location.y);

tic

% Solve elliptic eigenvalue problem associated to f_0
applyBoundaryCondition(model,'neumann','Edge',...
    1:model.Geometry.NumEdges,'g',0,'q',0); 
    % zero Neumann conditions
specifyCoefficients(model,'m',0,'d',1,'c',f0_fun,'a',0,'f',1); 
    % coefficients for eigenvalue equations
range = [-1,250]; 
    % range of search for eigenvalues
results = solvepdeeig(model,range); 
    % solve eigenvalue problem
lambdas0 = results.Eigenvalues; 
    % extract eigenvalues
J0 = length(lambdas0);
    % number of eigenvalues
u0 = results.Eigenvectors; 
    % extract eigenfunctions

toc

%%
%L^2 normalisation of eigenfunction

% In MATLAB R2023a, the eigenfunction are normalised in a way that the sum of 
% the squared values are equal to 1.
%for i=1:1
%    sum(u0(:,i).^2) % = 1 for each eigenfunction
%    mean(u0(:,i).^2) % = 1/mesh_size for each eigenfunction
%end

% Normalisation of first eigenfunction
u0(:,1) = 1/sqrt(vol); 
    % normalisation of first (constant) eigenfunction to obtain 
    % a proper transition pdf

% Normalise the other eigenfunctions so that mean(u0(:,j).^2) =
% 1/sqrt(vol).
u0(:,2:J0) = u0(:,2:J0)*sqrt(mesh_nodes_num/vol);

% Plot eigenfunctions
figure()
pdeplot(model,'XYData',u0(:,1)); 
%title('e_{f_0,0} (= constant)','FontSize',20)
axis equal
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20;
clim([min(u0(:,J0)),max(u0(:,J0))])

figure()
pdeplot(model,'XYData',u0(:,2)); 
%title('e_{f_0,1}','FontSize',20)
axis equal
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20;
clim([min(u0(:,J0)),max(u0(:,J0))])

figure()
pdeplot(model,'XYData',u0(:,J0)); 
%title('u_{f_0,J}','FontSize',15)
axis equal
xticks([-.5,-.25,0,.25,.5])
yticks([-.5,-.25,0,.25,.5])
ax = gca;
ax.FontSize = 20;
clim([min(u0(:,J0)),max(u0(:,J0))])

% Plot the eigenvalues
figure()
axes('FontSize', 20, 'NextPlot','add')
plot(lambdas_basis,'.','Linewidth',3)
xlabel('j', 'FontSize', 15);
ylabel('\lambda_j', 'FontSize', 15);

%%
% 2.2 Likelihood of the ground truth

%%
% Compute likelihood the along the observation path for the true conductivity

loglik0=0;
for m=1:(sample_size-1) 
    
    trpdf = 0;
    for j=1:J0
        trpdf = trpdf + exp(-deltaT_obs*lambdas0(j))*...
              u0(index_obs_mesh(m),j)*u0(index_obs_mesh(m+1),j);
    end
    loglik0 = loglik0 + log(abs(trpdf));
end
disp(['Log-likelihood of f_0 = ', num2str(loglik0)])

%%
% Compute log-likelihood of constant function for comparison

% Find Neumann eigenvalues for constant diffusivity
specifyCoefficients(model,'m',0,'d',1,'c',f_min+1,'a',0,'f',1); 
results = solvepdeeig(model,range);
lambdas_const = results.Eigenvalues;
J_const=length(lambdas_const);
u_const = results.Eigenvectors;
u_const(:,1) = 1/sqrt(vol);
u_const(:,2:length(lambdas_const))=u_const(:,2:length(lambdas_const))...
    *sqrt(mesh_nodes_num/vol);

% Compute the likelihood
loglik_const = 0;
for m=1:(sample_size-1) 
    trpdf = 0;
    for j=1:J_const
        trpdf = trpdf ... 
        + exp(-deltaT_obs*lambdas_const(j))*u_const(index_obs_mesh(m),j)...
        *u_const(index_obs_mesh(m+1),j);
    end
    loglik_const = loglik_const + log(abs(trpdf));
end
disp(['Log-likelihood of constant diffusivity = ', num2str(loglik_const)])
disp(['Log-likelihood of f0 = ', num2str(loglik0)])
