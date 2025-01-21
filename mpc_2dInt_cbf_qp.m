%% Discrete-time MPC controller with CBF constraints
% Recreation of 2D Double integrator example in 'Safety-Critical Model 
% Predictive Control with Discrete-TimeControl Barrier Function'
% Jun Zeng*, Bike Zhang* and Koushil Sreenath

% Vary N, gamma to recreate plots in Fig. 4.

%% MPC controllers with CBF constraints
clc;clear all;

%% Import CasADi
%% ADD CASADI PATH HERE
%% addpath('')
import casadi.*

%% System dynamics 

x1 = MX.sym('x1');
x2 = MX.sym('x2');
x3 = MX.sym('x3');
x4 = MX.sym('x4');
u1 = MX.sym('u1');
u2 = MX.sym('u2');

x = [x1;x2;x3;x4];
u = [u1;u2];

T = 0.2;
Tfinal = 100;

% Continuous dynamics 
Ac = [0 1 0 0;...
    0 0 0 0;...
    0 0 0 1;...
    0 0 0 0];
Bc = [0 0; 1 0;0 0;0 1];
Cc = eye(4);
Dc = [0 0;0 0;0 0;0 0]; 
sysc = ss(Ac, Bc, Cc, Dc);

% Discrete dynamics
sysd = c2d(sysc, T);

X0 = MX.sym('X0', 4);
U = MX.sym('U', 2);

X = sysd.A*X0 + sysd.B*U;
Y = sysd.C*X0;

F = Function('F', {X0, U}, {X}, {'x0', 'p'}, {'x_next'});

%% MPC problem formulation

% optimization parameters
N = 8;      % 1s look-ahead
Nx = 4;     % number of states
Nu = 2;     % number of inputs
Nm = 2;     % number of measurements 
gamma = 0.8;% cbf hyperparameter

% obstacle location
x_obs = -2;
y_obs = -2.25; 
r_obs = 1.5;

% weights
Q = 10*eye(4);
R = eye(2);
Pw = 100*eye(4);

% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];
P = {};

% "Lift" initial conditions
Xk = MX.sym('X0', 4);
w = {w{:}, Xk};
lbw = [lbw; -5; -inf; -5; -inf];    % State lower bounds
ubw = [ubw;  5; inf; 5; inf];       % State upper bounds
w0  = [w0; -5; 0; -5; 0];           % initial guess

% Add parameters for initial state
Pk = MX.sym('P0',4);
P = {P{:}, Pk};
g = {g{:}, Pk-Xk};
lbg = [lbg; 0; 0; 0; 0];         % Equality constraint on initial condition
ubg = [ubg; 0; 0; 0; 0];

for k = 0:N-1
    
    % New NLP variable for control
    Uk = MX.sym(['U_' num2str(k)], Nu);
    w = {w{:}, Uk};
    lbw = [lbw; -1; -1];    % control constraint lower bound 
    ubw = [ubw;  1; 1];     % control constraint upper bound
    w0 = [w0;  0; 0];
    
    % Current outputs
    Yk_curr = sysd.C*Xk;    % Perfect full state feedback
    
    % current barrier function
    hk_curr = (Yk_curr(1,1) - x_obs)^2 + (Yk_curr(3,1) - y_obs)^2 - r_obs^2;
    
    % Objective function
    J = J + Xk'*Q*Xk + Uk'*R*Uk;
    
    % Next point
    Fk =  F('x0', Xk, 'p', Uk);
    Xk_next = Fk.x_next;
      
    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 4);
    w = {w{:}, Xk};
    lbw = [lbw; -5; -inf; -5; -inf];
    ubw = [ubw;  5; inf; 5; inf];
    w0 = [w0; 0; 0; 0; 0];
    
    % barrier function at next point
    hk_next = (Xk(1,1) - x_obs)^2 + (Xk(3,1) - y_obs)^2 - r_obs^2;
    
    % barrier function constraints
    g = {g{:}, hk_next - hk_curr + gamma*hk_curr};
    lbg = [lbg; 0];         % CBF constraint lower bound
    ubg = [ubg; inf];       % CBF constraint upper bound
    
    % Add equality constraint
    g = {g{:}, Xk_next-Xk};
    lbg = [lbg; 0; 0; 0; 0];    % System dynamics equality constraints
    ubg = [ubg; 0; 0; 0; 0];
       
end

J = J + Xk'*Pw*Xk;    % terminal cost

% Create an NLP solver
opts=struct;
opts.print_time = 0;        % quiet run (looks like command only for ipopt)
opts.ipopt.print_level=0;
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}), 'p', vertcat(P{:}));
solver = nlpsol('solver', 'ipopt', prob, opts);

%% Main simulation

t0 = 0;             % initial time
x0 = [-5;0;-5;0];   % initial states
u0 = zeros(2,1);    % initial control
tsim = [];          % sim time
tsim = [tsim t0];   % first time instant

mpciter = 0;        % main mpc iteration
xx(:,1) = x0;       % final state trajectory
xxOpt = [];         % solution trajectory during each optimization window
u_cl=[];            % final control input 
u_cl(:,1) = u0';

args.w0 = [x0;repmat(zeros(2+4,1),N,1)];    % initial guess

main_loop = tic;
while mpciter < Tfinal/T

    args.p = x0;
    
    sol = solver('x0', args.w0, 'lbx', lbw, 'ubx', ubw,...
        'lbg', lbg, 'ubg', ubg,'p', args.p);
    
    xxOpt(:,mpciter+1) = full(sol.x);
    u_cl(:,mpciter+2) = xxOpt(5:6,mpciter+1)';
    
    xnext = F('x0',x0,'p', u_cl(:,mpciter+2));
    
    xx(:,mpciter+2) = full(xnext.x_next);
    args.w0 = xxOpt(:,mpciter+1);
    x0 = xx(:,mpciter+2);
    tsim(mpciter+2) = tsim(mpciter+1)+T;
    
    mpciter = mpciter + 1;
    
end
main_loop_time = toc(main_loop)

%% Plots

figure(1)
hold on
plot(xx(1,:), xx(3,:),'k-','LineWidth',1.5);    % system trajectory
grid on; box on;
xlabel('X [m]'); ylabel('Y [m]');
hold on
plot(x_obs+r_obs*cos([0:0.01:2*pi]),...
     y_obs+r_obs*sin([0:0.01:2*pi]), 'r--','LineWidth',1.5);    % obstacle
% ylim([-5 0]);
% xlim([-5 0]);

 %%



