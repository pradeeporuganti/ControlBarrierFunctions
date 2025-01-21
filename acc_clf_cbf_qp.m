%% ACC controller with CLF, CBF and input constraints

% Recreation of Adaptive Cruise Control (ACC) example in 'Control Barrier
%Function Based Quadratic Programs for Safety Critical Systems'
% Aaron D. Ames, Xiangru Xu, Jessy W. Grizzle, Paulo Tabuada
clc;clear all;close all;

%% Import CasADi
%% ADD CASADI PATH HERE
%% addpath('')
import casadi.*

%% Problem parameters

% vehicle parameters
v_lead = 14;     % lead vehicle speed [m/s]
v_des  = 24;     % ego vehicle velocity set by driver [m/s]
m      = 1650;   % ego vehicle mass 
accg   = 9.81;   % acceleration due to gravity
f0     = 0.1;    % road load parameter
f1     = 5;      % road load parameter
f2     = 0.25;   % road load parameter
ca     = 0.3;    % maximum acceleration pararmeter
cd     = 0.3;    % maximum deceleration parameter
T      = 1.8;    % look ahead [s]

% control limits
u_max = ca*m*accg;
u_min = -cd*m*accg;

% clf params
clf_rate = 5;
cbf_rate = 5;

% optim weights
w_inp = (2/(m^2));
w_slack = 2e-2;

% sim parameters
dt = 0.02;
sim_t = 20;
init = [0; 20; 100];

%% System dynamics

p = MX.sym('p');
v = MX.sym('v');
z = MX.sym('z');
u = MX.sym('u');
d = MX.sym('d');

x = [p;v;z];

Fr   = f0 + f1 + f2*v^2;
xdot = [v;...
    (u-Fr)/m;...
    v_lead-v];

f = Function('f', {x,u}, {xdot});

%% Integrator

% Fixed step Runge-Kutta 4 integrator
M = 4;                   % RK4 steps per interval
DT = dt/4;
X0 = MX.sym('X0', 3);
U = MX.sym('U');
X = X0;

for j=1:M
    
     k1 = f(X, U);
     k2 = f(X + DT/2 * k1, U);
     k3 = f(X + DT/2 * k2, U);
     k4 = f(X + DT * k3, U);
     
     X = X + DT/6*(k1 +2*k2 +2*k3 +k4);
     
end

F = Function('F', {X0, U}, {X}, {'x0','p'}, {'x_next'});

%% Setup
iter = 0;
xsim = [];
xsim = [xsim init];
slk = [];
usim = [];
usim = [usim;0];

while iter < sim_t/dt

    % decision variables
    w = {u, d};
    Fr = f0 + f1 + f2*xsim(2,iter+1)^2;
    
    % weights (note: input output linearization is done)
    Hacc = 2*[w_inp 0;...
        0 w_slack];

    Facc = -2*[Fr/m;0];

    % objective
    J = 0.5*([u;d]'*Hacc*[u;d])+ Facc'*[u;d];

    % constraints
    g1 = (xsim(2,iter+1)-v_des)*((2/m)*(u-Fr) + clf_rate*(xsim(2,iter+1)-v_des)) - d;                  % stability constraint
    g2 = (1/m)*(T + ((xsim(2,iter+1)-v_lead)/(cd*accg)))*(Fr-u) + (v_lead-xsim(2,iter+1))...
        + cbf_rate*(xsim(3,iter+1) - T*xsim(2,iter+1) - (0.5*(xsim(2,iter+1)-v_lead)^2/(cd*accg)));    % Safety constraint
    g  = {g1, g2};

    % bounds on constraints
    lbg = [-inf; 0];
    ubg = [0; inf];

    % bounds on decision variables
    lbw = [u_min; -inf];
    ubw = [u_max; inf];
    
    % setup optim problem
    qp = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
    solver = qpsol('solver', 'qpoases', qp);

    % solve optim problem
    sol = solver('x0', [0;0], 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
    
    % get solution
    xx = full(sol.x);
    usim = [usim; xx(1)];
    slk = [slk xx(2)];
    
    % apply control input on system
    x = F('x0', xsim(:,iter+1), 'p', xx(1));
%     x = full(f(xsim(:,iter+1), xx(1)));
    xsim = [xsim full(x.x_next)];   
%     xsim = [xsim x];

    iter = iter + 1;
    
end

for i = 1:length(xsim)
   
    % V(x)
    V(i) = (xsim(2,i) - v_des)^2;

    % B(x)
    B(i) = (xsim(3,i) - T*xsim(2,i) - ...
        (0.5*(xsim(2,i)-v_lead)^2/(cd*accg)));
    
end
%% Plots

t = 0:dt:sim_t;

figure(1)
plot(t, xsim(2,:),'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('v_{ego} [m/s]');
ylim([10 28]);

figure(2)
plot(t, xsim(3,:),'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('z [m]');
ylim([20 100]);

figure(3)
plot(t, usim,'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('u [N]');
ylim([-5000 5000]);

figure(4)
plot(t(2:end), slk,'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('slack');
ylim([0 600]);

figure(5)
plot(t, V,'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('CLF');
ylim([0 100]);

figure(6)
plot(t, B,'LineWidth',1.5);
grid on; box on;
xlabel('t [s]'); ylabel('CBF');
ylim([0 60]);

%%








