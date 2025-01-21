%% LKA MPC controllers with CLF, CBF constraints
clc;clear all;close all;

%% Import CasADi
%% ADD CASADI PATH HERE
%% addpath('')
import casadi.*

%% General Simulation Parameters

dt = 0.1;         % Time step (s) 
Tfinal = 10;     % Final simulation time (s)
t = 0:dt:Tfinal;  % Simulation time steps (s)
N = 10;           % 1s (N*dt) lookahead 

%% Environment parameters

lw = 3.7;         % Lane width [m]
v_act = 15;       % Other vehicle speed [m/s]

%% Reference trajectory

x_ref = 0:1:400;                        % X (inertial) [m]
y_ref = -lw/2.*(ones(size(x_ref)));     % Y (inertial) [m]
rho   = zeros(size(x_ref));             % lane curvature (straight road)

%% Vehicle parameters

% General system information
nx = 7;          % Number of states
nm = 2;          % Number of measurements
nc = 1;          % Number of inputs

% System Parameters
Izz = 3092;     % Vehicle yaw moment of inertia (kg.m^2)
Cf  = 6e4;      % Front tires cornering stiffness (N/rad)
rst = 1/15;     % Steering ratio 
Cr  = 5e4;      % Rear tires cornering stiffness (N/rad)
Cd  = 0.3;      % max decel
Ca  = 0.5;      % max acceleration
f0  = 0.1;      % aero drag coeff
f1  = 5;        % aero drag coeff
f2  = 0.25;     % aero drag coeff
lf  = 1.1579;   % Distance between vehicle CG to front axle (m)
lr  = 1.689;    % Distance between vehicle CG to rear axle (m)
m   = 1767.2;   % Vehilce mass (kg)
acc_g   = 9.81; % acceleration due to gravity

% constant longitudanal velocity used for lateral only analysis
% ux = 30;      % Vehicle longitudanal velocity (m/s) [60 mph]

% Control limits
umin = -3*0.5236;       % steering limit 
umax = 3*0.5236;        % (3 times higher than borelli paper to allow for evasive action)
amax = m*Ca*acc_g;      % acceleartion limits 
amin = -m*Cd*acc_g;     % (same as the A.Ames ACC CBF paper)

%% Vehicle dimensions

vl = 4.7;       % Length
vb = 1.8;       % Width

%% Nonlinear system model

% Model variables
% y       = MX.sym('y');      % Lateral distance: -4 to 4     (m)  
ux      = MX.sym('ux');     % Longitudanal velocity (m/s)
uy      = MX.sym('uy');     % Lateral velocity: -inf to inf (m/s)
psi     = MX.sym('psi');    % Heading/ Yaw: -0.4 to 0.4 (rad)  
r       = MX.sym('r');      % Yaw rate: -inf to inf (rad/s)
X       = MX.sym('X');      % Inertial X location: -inf to inf (m)
Y       = MX.sym('Y');      % Inertial Y location: -inf to inf (m)
e1      = MX.sym('e1');     % error in orientation (rad)
e2      = MX.sym('e2');     % lateral deviation (m)
d       = MX.sym('d');      % control input: steering angle: -0.5236*2 to 0.5236*2 (rad)
Fw      = MX.sym('Fw');     % control input: Wheel force: 
rho     = MX.sym('rho');    % road curvature (rad/m)

% theta   = MX.sym('theta');  % Steering angle (rad): -0.5236 to 0.5236 (rad) -30 deg to 30 deg
% u       = MX.sym('u');      % Rate of change of steering angle/ input: -0.3491 to 0.3491 (rad/s)
                              % Control amplitude limits from Borelli paper
                            
x = [ux; uy; psi; r; X; Y; e1; e2];
u = [Fw; d];

% Nonlinear dynamics (used for MPC)
xdot = [-((f0 + f1*ux + f2*ux^2)/m) + Fw/m;...
     (-(2*Cf+2*Cr)/(m*ux))*uy+((((2*Cr*lr)-(2*Cf*lf))/(m*ux))-ux)*r+((2*Cf*rst)/m)*d;...
     r;...
     ((2*Cr*lr-2*Cf*lf)/(Izz*ux))*uy + (-((2*Cf*lf^2)+(2*Cr*lr^2))/(ux*Izz))*r+((2*Cf*lf*rst)/Izz)*d;...   
     ux*cos(psi) - uy*sin(psi);...
     ux*sin(psi) + uy*cos(psi);...
     r - ux*rho;...
     uy + ux*e1];

C = [1 0 0 0 0 0 0 0;...
    0 0 0 0 0 0 1 0;...
    0 0 0 0 0 0 0 1];
 
f = Function('f', {x, u, rho}, {xdot});

%% Integrator 

% Fixed step Runge-Kutta 4 integrator
M   = 4;                   % RK4 steps per interval
DT  = dt/4;
X0  = MX.sym('X0', 8);
U   = MX.sym('U', 2);
R   = MX.sym('R');
X   = X0;

for j=1:M
    
     k1 = f(X, U, R);
     k2 = f(X + DT/2 * k1, U, R);
     k3 = f(X + DT/2 * k2, U, R);
     k4 = f(X + DT * k3, U, R);
     
     X = X + DT/6*(k1 +2*k2 +2*k3 +k4);
     
end

F = Function('F', {X0, U, R}, {X}, {'x0', 'd', 'rho'}, {'x_next'});

%% CLF-CBF parameters

delta   = MX.sym('delta');   % optim variable to relax CLF constraint

% lambda  = 0.1; 
gamma1 = 1;                  % optim hyperparameter for MPC collision avoidance
gamma2 = 1;                  % optim hyperparameter for safety filter 1
gamma3 = 0.5;                % optim hyperparameter for safety filter 2
w_slack = 2e-2;              % weight of CLF slack
H       = 1e-4.*eye(2, 2);   % weight for safety filter optim

%% MPC problem setup 

Q = zeros(2,2);             % weighing matrices (states)
Q(1,1) = 1e5; Q(2,2) = 10; Q(3,3) = 10;

R = zeros(2,2); 
R(1,1) = 1e-2; R(2,2) = 1; % weighing matrices (controls)

Th = 1;

% Controls
% du = MX.sym('du');          % -0.3491 to 0.3491 (rad) -20deg to 20deg
                            % limits from Borelli paper

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
Xk = MX.sym('X0', 8);
w = {w{:}, Xk};
lbw = [lbw; 10; -inf; -inf; -inf; -inf; -inf; -inf; -inf];
ubw = [ubw;  30; inf; inf; inf; inf; inf; inf; inf];
w0 = [w0; 25; 0; 0; 0; 0; 0; 0; 0];

% Add parameters for initial state
Pk = MX.sym('P0',8);
P = {P{:}, Pk};
g = {g{:}, Pk-Xk};
lbg = [lbg; 0; 0; 0; 0; 0; 0; 0; 0];
ubg = [ubg; 0; 0; 0; 0; 0; 0; 0; 0];

% Formulate the NLP
for k=0:N-1
    
    % New NLP variable for control
    Uk = MX.sym(['U_' num2str(k)], 2);
    w = {w{:}, Uk};
    lbw = [lbw; amin; umin];
    ubw = [ubw; amax; umax];
    w0 = [w0; 0; 0];
       
    % Current outputs
    Yk_curr = C*Xk;
    
    % Reference path parameters
    refk = MX.sym(['ref_' num2str(k)],1);
    P = {P{:}, refk};
    
    % obstacle parameters
    obsk = MX.sym(['obs_' num2str(k)],2);
    P = {P{:}, obsk};
    
    % Current barrier function
    hk_curr = (Yk_curr(2,1) - obsk(1,1))^4/vl^4 + (Yk_curr(3,1) - obsk(2,1))^4/vb^4 - 1;
    
    % New NLP variable for parameters
    J = J+(Yk_curr - [30;0;0])'*Q*(Yk_curr - [30;0;0])+Uk'*R*Uk;
   
    % Next point
    Fk =  F('x0', Xk, 'd', Uk, 'rho', refk);
    Xk_next = Fk.x_next;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 8);
    w = {w{:}, Xk};
    lbw = [lbw; 10; -inf; -inf; -inf; -inf; -inf; -inf; -inf];
    ubw = [ubw;  30; inf; inf; inf; inf; inf; inf; inf];
    w0 = [w0; 25; 0; 0; 0; 0; 0; 0; 0];
    
    % barrier function at next point
    hk_next = (Xk(5,1) - obsk(1,1))^4/vl^4 + (Xk(6,1) - obsk(2,1))^4/vb^4 - 1;
    
    % barrier function constraints
    g = {g{:}, hk_next - hk_curr + gamma1*hk_curr};
    lbg = [lbg; 0];         % CBF constraint lower bound
    ubg = [ubg; inf];       % CBF constraint upper bound

    % Add equality constraint
    g = {g{:}, Xk_next-Xk};
    lbg = [lbg; 0; 0; 0; 0; 0; 0; 0; 0];
    ubg = [ubg; 0; 0; 0; 0; 0; 0; 0; 0];
    
end

% Create an NLP solver
opts=struct;
opts.print_time = 0;
opts.ipopt.print_level=0;
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}), 'p', vertcat(P{:}));
solver = nlpsol('solver', 'ipopt', prob, opts);

%% Main simulation with Receding Horizon Control

% init stuff
t0   = 0;             % initial time
x0   = [25; 0; 0; 0; 0; -lw/2; 0; 0];    % initial states
u0   = [0; 0];        % initial control
tsim = [];          % sim time
tsim = [tsim t0];   % first time instant

args.w0 = w0;       % init optim guess

mpciter = 0;        % main mpc iteration
xx(:,1) = x0;       % final state trajectory
xxOpt   = [];         % solution trajectory during each optimization window

u_cl      = [];            % final control input
u_cl(:,1) = u0;

u_safe       = [];
u_safe(:, 1) = u0;

% del      = [];
% del(:,1) = [del delta0];

X_act = 100; Y_act = -lw/2;
actorY = []; actorY = [actorY Y_act];
actorX = []; actorX = [actorX X_act];

% Main loop
main_loop = tic;
while mpciter < Tfinal/dt
    %% Solve MPC problem for RHC
    
    % optim parameters
    % Current ego-vehicle state + N step curvature preview
%     args.p = [x0; zeros(N,1);  X_act+15*(tsim+dt:dt:tsim+(N*dt))'; -lw/2.*ones(N,1)];
    
    % Input parameters 
    % Using workaround for straight road (0 curvature)
    % look at "mpc_cbf_combined.m" for correct implementation
    
    curv_prev = -1.*ones(N,1); % placeholder for curvature
    par = zeros(length(vertcat(P{:}))-length(x0),1);
    par(1:3:end) = curv_prev;  % input curvature values into parameter array
    
    act_X = X_act+v_act*(tsim+dt:dt:tsim+(N*dt))'; act_Y = -lw/2.*ones(N,1);
    act = [act_X act_Y]'; act = act(:);  % Alternating X-Y predictions of actor
    
    [ind, ~] = find(par == 0); par(ind) = act;           % input into parameter array
    % replace curvature placeholder with true curvature look-ahead
    [ind, ~] = find(par == -1); par(ind) = 0;                 
    
    args.p = [x0; par];       % final input parameters for optim problem
      
    % Solve N-step optim problem to get u_des N-step policy
    sol = solver('x0', args.w0, 'lbx', lbw, 'ubx', ubw,...
        'lbg', lbg, 'ubg', ubg,'p', args.p);
    
    % N*size(w) optim solution used as guess for next optim cycle
    xxOpt(:,mpciter+1) = full(sol.x);
    u_cl(:,mpciter+2) = xxOpt(9:10,mpciter+1);   % receding horizon control input
    
    %% Safety filter
    
    % Cost function
    J1    = 0.5*([Fw;d]-u_cl(:,mpciter+2))'*H*([Fw;d]-u_cl(:,mpciter+2)); % + w_slack*delta^2;
    
    % decision variable
    w1    = {Fw; d};
    
%     stb_con = (x0(1)-30)*((2*(Fw-(f0 + f1*x0(1) + f2*x0(1)^2))/m) +(lambda*(x0(1)-30))) - delta;
    saf_con = (1/m)*(Th + ((x0(1)-v_act)/(Cd*acc_g)))*((f0 + f1*x0(1) + f2*x0(1)^2)-Fw) + (v_act-x0(1))...
        + gamma2*((X_act-x0(5)) - Th*x0(1) - (0.5*(x0(1)-v_act)^2/(Cd*acc_g)));
%     g1 = {stb_con; saf_con};
    g1 = {saf_con};
    
    % bounds on constraint
    lbg1   = [0];
    ubg1   = [inf];
    
    % bounds on decision variable
    lbw1   = [amin; umin];
    ubw1   = [amax; umax];
    
    % setup optim problem 
    qp = struct('f', J1, 'x', vertcat(w1{:}), 'g', vertcat(g1{:}));
    sol = qpsol('sol', 'qpoases', qp);
   
    % solve current optim problem
    safe_sol = sol('x0', [u_cl(:,mpciter+2)], 'lbx', lbw1, 'ubx', ubw1,...
            'lbg', lbg1, 'ubg', ubg1);
    
    u_safe(:,mpciter+2)   = full(safe_sol.x(1:2,1));    % filtered "safe" RHC
    
    %% simulate dynamics + extract signals + init for next iter 
    
    xnext           = F('x0', x0, 'd', u_safe(:,mpciter+2), 'rho', 0);
    xx(:,mpciter+2) = full(xnext.x_next);
    args.w0         = xxOpt(:,mpciter+1);
    x0              = xx(:,mpciter+2);
    tsim(mpciter+2) = tsim(mpciter+1)+dt;
    mpciter = mpciter + 1;
    
    %% Update actor position
    
    X_act = X_act + v_act*dt;
    Y_act = -lw/2;
    actorX = [actorX X_act]; actorY = [actorY Y_act];
    
end
main_loop_time = toc(main_loop)

%% Animation
th =linspace(0,2*pi);

% v = VideoWriter('cbf_combined.avi');
% open(v);

for i=1:length(xx(5,:))
    
    f = figure(1);
    f.Position(3:4) = [520 985];
    movegui(f,'south');
    plot(xx(6,i), xx(5,i), 'o','MarkerFaceColor','b');
    hold on
    plot(xx(6,i)+ vb*sin(th) ,xx(5,i) + vl*cos(th),'b-');
%     plot(5*sin(th)+xx(6,i),  5*cos(th)+xx(5,i), 'b-');
    hold on
    plot(y_ref, x_ref, 'g-','LineWidth',1.5);
%     plot(ref.ego_Y, ref.ego_X, 'g-','LineWidth',1.5);
    hold on
    plot(actorY(1,i), actorX(1,i), 'o','MarkerFaceColor','r');
    hold on
    plot(actorY(1,i)+vb*sin(th),  actorX(1,i)+vl*cos(th), 'r-');
    hold on
    plot(lw/2+y_ref, x_ref, 'k-','LineWidth', 1.5);
    hold on
    plot(-lw/2+y_ref, x_ref, 'k-','LineWidth', 1.5);
    hold on 
    plot((1.5*lw)+y_ref, x_ref, 'k-','LineWidth', 1.5);
    grid on; box on;
    xlabel('Y [m]'); ylabel('X [m]');
    xlim([-40 40]); ylim([-40 430])
    %legend('v_{ego}', 'ref', 'other vehicle','lane boundary');
    hold off;
    pause(.1);
    
%     mov = getframe(gcf);
%     writeVideo(v,mov);
end

%%


