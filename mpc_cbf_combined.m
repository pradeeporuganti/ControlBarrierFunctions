%% LKA MPC controllers with CLF, CBF constraints
clc;clear all;close all;

%% Import CasADi
%%%% ADD CASADI PATH HERE
% addpath('')
import casadi.*

%% General
dt = 0.1;         
Tfinal = 4.5;     % Final simulation time (s)
t = 0:dt:Tfinal;  % Simulation time steps (s)

%% Reference trajectory
ref = load('ref.mat');
ref.kappa = (ref.lb1_k + ref.lb2_k) / 2;    % avg curvature

%% Vehicle parameters

% General system information
nx = 7;          % Number of states
nm = 2;          % Number of measurements
nc = 1;          % Number of inputs

% System Parameters
Izz = 3092;     % Vehicle yaw moment of inertia (kg.m^2)
Cf = 6e4;       % Front tires cornering stiffness (N/rad)
Cr = 5e4;       % Rear tires cornering stiffness (N/rad)
rst = 1/15;     % Steering ratio 
lf = 1.1579;    % Distance between vehicle CG to front axle (m)
lr = 1.689;     % Distance between vehicle CG to rear axle (m)
m = 1767.2;     % Vehilce mass (kg)
ux = 30;        % Vehicle longitudanal velocity (m/s) [60 mph]
umin = -2*0.5236; % 2*Limits from Borelli paper
umax = 2*0.5236;

%% Nonlinear system model

% Model variables
% y       = MX.sym('y');      % Lateral distance: -4 to 4     (m)  
uy      = MX.sym('uy');     % Lateral velocity: -inf to inf (m/s)
psi     = MX.sym('psi');    % Heading/ Yaw: -0.4 to 0.4 (rad)  
r       = MX.sym('r');      % Yaw rate: -inf to inf (rad/s)
X       = MX.sym('X');      % Inertial X location: -inf to inf (m)
Y       = MX.sym('Y');      % Inertial Y location: -inf to inf (m)
e1      = MX.sym('e1');     % error in orientation (rad)
e2      = MX.sym('e2');     % lateral deviation (m)
d       = MX.sym('d');      % control input: steering angle: -0.5236*2 to 0.5236*2 (rad)
rho     = MX.sym('rho');    % road curvature (rad/m)

% theta   = MX.sym('theta');  % Steering angle (rad): -0.5236 to 0.5236 (rad) -30 deg to 30 deg
% u       = MX.sym('u');      % Rate of change of steering angle/ input: -0.3491 to 0.3491 (rad/s)
                              % Control amplitude limits from Borelli paper
                            
x = [uy; psi; r; X; Y; e1; e2];

% Nonlinear dynamics (used for MPC)
xdot = [(-(2*Cf+2*Cr)/(m*ux))*uy+((((2*Cr*lr)-(2*Cf*lf))/(m*ux))-ux)*r+((2*Cf*rst)/m)*d;...
     r;...
     ((2*Cr*lr-2*Cf*lf)/(Izz*ux))*uy + (-((2*Cf*lf^2)+(2*Cr*lr^2))/(ux*Izz))*r+((2*Cf*lf*rst)/Izz)*d;...   
     ux*cos(psi) - uy*sin(psi);...
     ux*sin(psi) + uy*cos(psi);...
     r - ux*rho;...
     uy + ux*e1];

% C = [0 0 0 0 0 1 0;...
%     0 0 0 0 0 0 1];
C = eye(7);
 
f = Function('f', {x, d, rho}, {xdot});

%% Safety filter optim setup (not used)

% % variables for CBF
% u       = MX.sym('u');       % 'Safe' signal (rad)
% % Ps      = MX.sym('Ps',4);  % actor position parameters
% X_act   = MX.sym('X_act');   % actor X pos [m]
% Y_act   = MX.sym('Y_act');   % actor Y pos [m]
% p       = 1e-21;              % penalty on class-K functions
% H       = 10;                % weight for safety filter optim
% 
% % decision variable
% w1    = {u};
% 
% % Nonlinear system dynamics (used for safety filter)
% fx = [(-(2*Cf+2*Cr)/(m*ux))*uy+((((2*Cr*lr)-(2*Cf*lf))/(m*ux))-ux)*r;...
%      r;...
%      ((2*Cr*lr-2*Cf*lf)/(Izz*ux))*uy + (-((2*Cf*lf^2)+(2*Cr*lr^2))/(ux*Izz))*r;...   
%      ux*cos(psi) - uy*sin(psi);...
%      ux*sin(psi) + uy*cos(psi);...
%      r - ux*rho;...
%      uy + ux*e1];
%  
% gx = [((2*Cf*rst)/m);...
%     0;...
%     ((2*Cf*lf*rst)/Izz);...
%     0;...
%     0;...
%     0;...
%     0];
% 
% xd = fx + gx*u;             % System dynamics in control affine form
% 
% bx = (X-X_act)^2 + (Y-Y_act)^2 - 25;    % Safety constraint
% y1 = dot(gradient(bx, x),fx);               % Lfb
% y2 = dot(gradient(bx, x),gx);               % Lgb
% y3 = dot(gradient(y1, x),fx);               % Lf2b
% y4 = dot(gradient(y1, x),gx);               % LgLfb
% 
% % Creating functions for each
% bar     = Function('bar', {x, rho, X_act, Y_act}, {bx});
% Lfb     = Function('Lfb', {x, rho, X_act, Y_act}, {y1});
% Lgb     = Function('Lgb', {x, rho, X_act, Y_act}, {y2});
% Lf2b    = Function('Lf2b', {x, rho, X_act, Y_act}, {y3});
% LgLfb   = Function('LgLfb', {x, rho, X_act, Y_act}, {y4});

%% Integrator 

% Fixed step Runge-Kutta 4 integrator
M = 4;                   % RK4 steps per interval
DT = dt/4;
X0 = MX.sym('X0', 7);
U = MX.sym('U');
R = MX.sym('R');
X = X0;

for j=1:M
    
     k1 = f(X, U, R);
     k2 = f(X + DT/2 * k1, U, R);
     k3 = f(X + DT/2 * k2, U, R);
     k4 = f(X + DT * k3, U, R);
     
     X = X + DT/6*(k1 +2*k2 +2*k3 +k4);
     
end

F = Function('F', {X0, U, R}, {X}, {'x0', 'd', 'rho'}, {'x_next'});

%% Dynamics check

% actual = load('fromCourse.mat');
% swa = actual.swa.Data;
% 
% X_init = zeros(nx,1);
% X = [];
% X = [X X_init];
% 
% for i = 2:length(actual.swa.Time)
% 
%     Fk = F('x0', X(:,i-1), 'p', actual.swa.Data(i-1));
%     Xk_next = full(Fk.x_next);
%     X = [X Xk_next];
%     
% end

%% MPC problem setup 

Q = zeros(2,2);             % weighing matrices (states)
Q(1,1) = 1000; Q(2,2) = 1000;
N = 20;                      % 
gamma = 0.8;                  % cbf hyperparameter
r_obs = 5;

R = zeros(1,1); R(1,1) = 1; % weighing matrices (controls)

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
Xk = MX.sym('X0', 7);
w = {w{:}, Xk};
lbw = [lbw; -inf; -inf; -inf; -inf; -inf; -inf; -inf];
ubw = [ubw;  inf; inf; inf; inf; inf; inf; inf];
w0 = [w0; 0; 0; 0; 0; 0; 0; 0];

% Add parameters for initial state
Pk = MX.sym('P0',7);
P = {P{:}, Pk};
g = {g{:}, Pk-Xk};
lbg = [lbg; 0; 0; 0; 0; 0; 0; 0];
ubg = [ubg; 0; 0; 0; 0; 0; 0; 0];

% Formulate the NLP
for k=0:N-1
    
    % New NLP variable for control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};
    lbw = [lbw; -0.5236*5];
    ubw = [ubw;  0.5236*5];
    w0 = [w0;  0];
       
    % Current outputs
    Yk_curr = C*Xk;
    
    % Reference path parameters
    refk = MX.sym(['ref_' num2str(k)],1);
    P = {P{:}, refk};
    
    % obstacle parameters
    obsk = MX.sym(['obs_' num2str(k)],2);
    P = {P{:}, obsk};
    
    % current barrier function
    hk_curr = (Yk_curr(4,1) - obsk(1,1))^2 + (Yk_curr(5,1) - obsk(2,1))^2 - r_obs^2;
    
    % New NLP variable for parameters
    J = J+(Yk_curr(6:7,1))'*Q*(Yk_curr(6:7,1))+Uk'*R*Uk;
   
    % Next point
    Fk =  F('x0', Xk, 'd', Uk, 'rho', refk);
    Xk_next = Fk.x_next;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 7);
    w = {w{:}, Xk};
    lbw = [lbw; -inf; -inf; -inf; -inf; -inf; -inf; -inf];
    ubw = [ubw;  inf; inf; inf; inf; inf; inf; inf];
    w0 = [w0; 0; 0; 0; 0; 0; 0; 0];
    
    % barrier function at next point
    hk_next = (Xk(4,1) - obsk(1,1))^2 + (Xk(5,1) - obsk(2,1))^2 - r_obs^2;
    
    % barrier function constraints
    g = {g{:}, hk_next - hk_curr + gamma*hk_curr};
    lbg = [lbg; 0];         % CBF constraint lower bound
    ubg = [ubg; inf];       % CBF constraint upper bound
    
    % Add equality constraint
    g = {g{:}, Xk_next-Xk};
    lbg = [lbg; 0; 0; 0; 0; 0; 0; 0];
    ubg = [ubg; 0; 0; 0; 0; 0; 0; 0];
    
end

% Create an NLP solver
opts=struct;
opts.print_time = 0;
opts.ipopt.print_level=0;
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}), 'p', vertcat(P{:}));
solver = nlpsol('solver', 'ipopt', prob, opts);

%% Main simulation with Receding Horizon Control

t0 = 0;             % initial time
x0 = [0; 0; 0; -8.44; -0.8; 0; 0];    % initial states
u0 = zeros(1,1);    % initial control
tsim = [];          % sim time
tsim = [tsim t0];   % first time instant

mpciter = 0;        % main mpc iteration
xx(:,1) = x0;       % final state trajectory
xxOpt = [];         % solution trajectory during each optimization window
u_cl=[];            % final control input (rate of change of steering)
u_cl(1) = u0;

u_safe = [];
u_safe(1) = u0;

args.w0 = [x0;repmat(zeros(1+7,1),N,1)];    % initial guess
check = []; check1 = []; check2 = [];
cost = []; safety = []; actorX = []; actorY = [];

main_loop = tic;
while mpciter < Tfinal/dt
    %% Solve MPC problem for RHC
%     args.p = look_ahead(xx(:,mpciter+1), tsim(mpciter+1),...
%         dt, N, actual, Tfinal);     % current reference trajectory
    curv_prev = curv_preview(N-1, x0(4));
    par = zeros(3*length(curv_prev),1);
    par(1:3:end) = curv_prev;
    
    % current location of other vehicle 
    act_X = interp1(ref.t_act, ref.act_x, tsim(mpciter+1):dt:tsim(mpciter+1)+(N-1)*dt);
    act_Y = interp1(ref.t_act, ref.act_y, tsim(mpciter+1):dt:tsim(mpciter+1)+(N-1)*dt);
    act = [act_X; act_Y]; act = act(:)';
    actorX = [actorX act_X(1)];
    actorY = [actorY act_Y(1)]; 
    
    [ind, ~] = find(par == 0);
    par(ind) = act;
    
    % Current ego-vehicle state + N step curvature preview
%     args.p = [x0;curv_prev(2:end)];
    args.p = [x0; par];
    
    % Solve N-step optim problem to get u_des N-step policy
    sol = solver('x0', args.w0, 'lbx', lbw, 'ubx', ubw,...
        'lbg', lbg, 'ubg', ubg,'p', args.p);
    
    % N*size(w) optim solution used as init for next optim cycle
    xxOpt(:,mpciter+1) = full(sol.x);
%     cost = [cost ;full(sol.f)];             % extract cost
    u_cl(mpciter+2) = xxOpt(8,mpciter+1);   % receding horizon control input
    
    %% Point-wise safety filter optimization
    
%     % Cost function
%     J1    = 0.5*(u-u_cl(mpciter+2))*H*(u-u_cl(mpciter+2));
%     
%     % terms for the control-input constraint 
%     t1    = full(Lf2b(x0, curv_prev(1), act_X, act_Y));
%     t2    = full(LgLfb(x0, curv_prev(1), act_X, act_Y));
%     t3    = full(Lfb(x0, curv_prev(1), act_X, act_Y));
%     t4    = full(Lgb(x0, curv_prev(1), act_X, act_Y));
%     t5    = full(bar(x0, curv_prev(1), act_X, act_Y));
%     
%     % constraint; choosing linear/ quadratic class-K functions
% %     con    = t1 + t2*u + p*(t3+t4) + p*(t3 + t4 + p*t5)^2;
%     con = t1 + t2*u + 2*p*t3*t5 + 2*t3^2 + 2*p^2*t3*t5^2 +...
%         p^3*t5^4;
%     g1     = {con};
%     
%     % bounds on constraint
%     lbg1   = [0];
%     ubg1   = [inf];
%     
%     % bounds on decision variable
%     lbw1   = [umin];
%     ubw1   = [umax];
%     
%     % setup optim problem 
%     qp = struct('f', J1, 'x', vertcat(w1{:}), 'g', vertcat(g1{:}));
%     sol = qpsol('sol', 'qpoases', qp);
%     
%     % solve optim problem
%     safe_sol = sol('x0', u_cl(mpciter+2), 'lbx', lbw1, 'ubx', ubw1,...
%             'lbg', lbg1, 'ubg', ubg1);
%         
%     u_safe(mpciter+2)   = full(safe_sol.x);
%     constra(mpciter+1)  = full(safe_sol.g);
%     cost(mpciter+1)     = full(safe_sol.f);
%     safety(mpciter+1)   = t5;
    
    %% simulate dynamics + extract signals + init for next iter 
    
    xnext = F('x0', x0, 'd', u_cl(mpciter+2), 'rho', curv_prev(1));
    xx(:,mpciter+2) = full(xnext.x_next);
        
    args.w0 = xxOpt(:,mpciter+1);
    x0 = xx(:,mpciter+2);
    tsim(mpciter+2) = tsim(mpciter+1)+dt;
    
    mpciter = mpciter + 1;
    
end
main_loop_time = toc(main_loop)

%% plots

th = 0:pi/50:2*pi;

for i=1:length(actorY)
    
    f = figure(1);
    f.Position(3:4) = [520 985];
    movegui(f,'south');
    plot(xx(5,i), xx(4,i), 'o','MarkerFaceColor','b');
    hold on
    plot(5*sin(th)+xx(5,i),  5*cos(th)+xx(4,i), 'b-');
    hold on
    plot(ref.ego_Y, ref.ego_X, 'g-','LineWidth',1.5);
    hold on
    plot(actorY(1,i), actorX(1,i), 'o','MarkerFaceColor','r');
    hold on
    plot(5*sin(th)+actorY(1,i),  5*cos(th)+actorX(1,i), 'r-');
    hold on
    plot(ref.lb1_y, ref.lb1_x, 'k-','LineWidth', 1.5);
    hold on
    plot(ref.lb2_y, ref.lb2_x, 'k-','LineWidth', 1.5);
    grid on; box on;
    xlabel('Y [m]'); ylabel('X [m]');
    xlim([-40 40]); ylim([-20 140])
    %legend('MPC output', 'human', 'other vehicle','lane boundary');
    hold off;
    pause(.1);
    
end

%% 
