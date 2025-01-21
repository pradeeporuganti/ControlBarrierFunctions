%% LKA MPC controllers with CLF, CBF constraints
clc;clear all;close all;

%% Import CasADi
%% ADD CASADI PATH HERE
%% addpath('')
import casadi.*

%% General
dt = 0.1;         
Tfinal = 4.5;     % Final simulation time (s)
t = 0:dt:Tfinal;  % Simulation time steps (s)

%% Reference trajectory
ref = load('ref.mat');
ref.kappa = (ref.lb1_k + ref.lb2_k) / 2;    % avg curvature

%% Reference control input
load('u_cl.mat');

%% Vehicle parameters

% General system information
nx = 8;          % Number of states
nm = 8;          % Number of measurements
nc = 2;          % Number of inputs

% System Parameters
p.Izz = 3092;     % Vehicle yaw moment of inertia (kg.m^2)
p.Cf  = 6e4;      % Front tires cornering stiffness (N/rad)
p.rst = 1/15;     % Steering ratio 
p.Cr  = 5e4;      % Rear tires cornering stiffness (N/rad)
p.Cd  = 0.3;      % max decel
p.Ca  = 0.5;      % max acceleration
p.f0  = 0.1;      % aero drag coeff
p.f1  = 5;        % aero drag coeff
p.f2  = 0.25;     % aero drag coeff
p.lf  = 1.1579;   % Distance between vehicle CG to front axle (m)
p.lr  = 1.689;    % Distance between vehicle CG to rear axle (m)
p.m   = 1767.2;   % Vehilce mass (kg)
p.acc_g   = 9.81;     % acceleration due to gravity
p.v0  = 15;       % Velocity of other actor (constant)

% Control limits
umin = -5*0.5236;       % steering limit 
umax = 5*0.5236;        % (3 times higher than borelli paper to allow for evasive action)
amax = p.m*p.Ca*p.acc_g;      % acceleartion limits 
amin = -p.m*p.Cd*p.acc_g;     % (same as the A.Ames ACC CBF paper)

%% Sensor parameters

variance = 1e-3;        % sensor parameters currently chosen at random

%% Nonlinear system model

% Model variables
ux      = MX.sym('ux');     % Longitudanal velocity (m/s)
uy      = MX.sym('uy');     % Lateral velocity: -inf to inf (m/s)
psi     = MX.sym('psi');    % Heading/ Yaw: -0.4 to 0.4 (rad)  
r       = MX.sym('r');      % Yaw rate: -inf to inf (rad/s)
X       = MX.sym('X');      % Inertial X location: -inf to inf (m)
Y       = MX.sym('Y');      % Inertial Y location: -inf to inf (m)
Fw      = MX.sym('Fw');     % control input: Force at the wheel (N)
d       = MX.sym('d');      % control input: steering angle (rad)
rho     = MX.sym('rho');    % road curvature (rad/m)
                            
x = [ux; uy; psi; r; X; Y];
u = [Fw; d];

% Nonlinear dynamics 
xdot = [-((p.f0 + p.f1*ux + p.f2*ux^2)/p.m) + Fw/p.m;...
     (-(2*p.Cf+2*p.Cr)/(p.m*ux))*uy+((((2*p.Cr*p.lr)-(2*p.Cf*p.lf))/(p.m*ux))-ux)*r+((2*p.Cf*p.rst)/p.m)*d;...
     r;...
     ((2*p.Cr*p.lr-2*p.Cf*p.lf)/(p.Izz*ux))*uy + (-((2*p.Cf*p.lf^2)+(2*p.Cr*p.lr^2))/(ux*p.Izz))*r+((2*p.Cf*p.lf*p.rst)/p.Izz)*d;...   
     ux*cos(psi) - uy*sin(psi);...
     ux*sin(psi) + uy*cos(psi)];

C = [1 0 0 0 0 0;...    % backtrack from engine speed sensor [y1]
    1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1;...     % GPS Y-pos [y6]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] (9x6 matrix)

% Linearized system dynamics at init
A = [ -20.0000         0         0         0         0         0;...
         0   -4.1497         0  -29.4350         0         0;...
         0         0         0    1.0000         0         0;...
         0    0.3229         0   -4.8098         0         0;...
    1.0000         0         0         0         0         0;...
         0    1.0000   30.0000         0         0         0];
     
% obsv(A,C) = 6 (full)
 
f = Function('f', {x, u}, {xdot});

%% Decomposing output space
% Note: rank(obsv(A,Ci)) = 6
% Can build Generalized Observer Scheme of fault detection

% Remove engine speed sensor
C1 = [1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1;...     % GPS Y-pos [y6]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] 

% Remove wheel speed sensor
C2 = [1 0 0 0 0 0;...    % backtrack from engine speed sensor [y1]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1;...     % GPS Y-pos [y6]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] 

% Remove IMU   
C3 = [1 0 0 0 0 0;...    % backtrack from engine speed sensor [y1]
    1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1;...     % GPS Y-pos [y6]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] 

% Remove yaw rate 
C4 = [1 0 0 0 0 0;...    % backtrack from engine speed sensor [y1]
    1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1;...     % GPS Y-pos [y6]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] 

% Remove GPS (2 rows)
C5 = [1 0 0 0 0 0;...    % backtrack from engine speed sensor [y1]
    1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % Lidar X-pos [y7]
    0 0 0 0 0 1;...     % Lidar Y-pos [y8]
    0 0 1 0 0 0];       % Lidar orientation [y9] 

% Remove Lidar (3 rows)

C6 = [1 0 0 0 0 0;...   % backtrack from engine speed sensor [y1]
    1 0 0 0 0 0;...     % backtrack from wheel speed sensor [y2]
    0 0 1 0 0 0;...     % IMU for orientation [y3]
    0 0 0 1 0 0;...     % Yaw rate sensor [y4]
    0 0 0 0 1 0;...     % GPS X-pos [y5]
    0 0 0 0 0 1];       % GPS Y-pos [y6]

%% Integrator 

% Fixed step Runge-Kutta 4 integrator
M = 4;                   % RK4 steps per interval
DT = dt/4;
X0 = MX.sym('X0', 6);
U = MX.sym('U', 2);
X = X0;

for j=1:M
    
     k1 = f(X, U);
     k2 = f(X + DT/2 * k1, U);
     k3 = f(X + DT/2 * k2, U);
     k4 = f(X + DT * k3, U);
     
     X = X + DT/6*(k1 +2*k2 +2*k3 +k4);
     
end

F = Function('F', {X0, U}, {X}, {'x0', 'u'}, {'x_next'});

%% Simulation

% init stuff
x0 = [30; 0; 0; 0; -8.44; -0.8];    % initial states
t0 = 0;

tsim = []; actorX = []; actorY = [];
tsim = [tsim; t0];
xx(:,1) = x0;

x_est(:, 1) = x0; x_est1(:, 1) = x0; x_est2(:, 1) = x0; x_est3(:, 1) = x0; 
x_est4(:, 1) = x0; x_est5(:, 1) = x0; x_est6(:, 1) = x0; 

y(:, 1) = C*x0; y1(:, 1) = C1*x0; y2(:, 1) = C2*x0; y3(:, 1) = C3*x0; 
y4(:, 1) = C4*x0; y5(:, 1) = C5*x0; y6(:, 1) = C6*x0;

r1 = []; r2 = []; r3 = []; r4 = []; r5 = []; r6 = [];

iter = 0;

% init estimate covariance matrix for EKF array
P = eye(6,6);   
P1 = P;
P2 = P;
P3 = P;
P4 = P;
P5 = P;
P6 = P;

while  iter < Tfinal/dt
    
    xnext = F('x0', xx(:,iter+1), 'u', u_cl(:, iter+1));
    xx(:,iter+2) = full(xnext.x_next);
    
    w = sqrt(variance).*randn(length(x), 1);    % Gaussian white noise
    y(:, iter+2) = C*(xx(:, iter+2) +  w);       
    
    if tsim(iter+1) >= 3 & tsim(iter+1) < 4 
        
        % GPS malfunction
        y(:, iter+2) = C*(xx(:, iter+2) +  w) + [0;0;0;0;10;10;0;0;0]; 
        
    end 
      
    y1(:, iter+2) = y(2:end, iter+2);                       % es
    y2(:, iter+2) = [y(1, iter+2); y(3:end, iter+2)];       % ws
    y3(:, iter+2) = [y(1:2, iter+2); y(4:end, iter+2)];     % imu
    y4(:, iter+2) = [y(1:3, iter+2); y(5:end, iter+2)];     % yaw
    y5(:, iter+2) = [y(1:4, iter+2); y(7:end, iter+2)];     % gps
    y6(:, iter+2) = y(1:6, iter+2);                         % lidar
    
    % Assume no processs noise
    % Full estimate from EKF
    [x_hat, P] = ekf(F, x_est(:, iter+1), u_cl(:, iter+1), y(:, iter+2), p, P, C);
    x_est(:, iter+2) = full(x_hat);
    
    % Estimates from observer array
    [x_hat1, P1] = ekf(F, x_est1(:, iter+1), u_cl(:, iter+1), y1(:, iter+2), p, P1, C1);
    x_est1(:, iter+2) = full(x_hat1);
    
    [x_hat2, P2] = ekf(F, x_est2(:, iter+1), u_cl(:, iter+1), y2(:, iter+2), p, P2, C2);
    x_est2(:, iter+2) = full(x_hat2);
    
    [x_hat3, P3] = ekf(F, x_est3(:, iter+1), u_cl(:, iter+1), y3(:, iter+2), p, P3, C3);
    x_est3(:, iter+2) = full(x_hat3);
    
    [x_hat4, P4] = ekf(F, x_est4(:, iter+1), u_cl(:, iter+1), y4(:, iter+2), p, P4, C4);
    x_est4(:, iter+2) = full(x_hat4);
    
    [x_hat5, P5] = ekf(F, x_est5(:, iter+1), u_cl(:, iter+1), y5(:, iter+2), p, P5, C5);
    x_est5(:, iter+2) = full(x_hat5);
    
    [x_hat6, P6] = ekf(F, x_est6(:, iter+1), u_cl(:, iter+1), y6(:, iter+2), p, P6, C6);
    x_est6(:, iter+2) = full(x_hat6);
    
    % Residual generation
    r1(:,iter+1) = y1(:, iter+2) - C1*(x_est1(:, iter+2));
    r2(:,iter+1) = y2(:, iter+2) - C2*(x_est2(:, iter+2));
    r3(:,iter+1) = y3(:, iter+2) - C3*(x_est3(:, iter+2));
    r4(:,iter+1) = y4(:, iter+2) - C4*(x_est4(:, iter+2));
    r5(:,iter+1) = y5(:, iter+2) - C5*(x_est5(:, iter+2));
    r6(:,iter+1) = y6(:, iter+2) - C6*(x_est6(:, iter+2));
    
%     % Fault/ attack detection logic
%     sensor(:, iter+1) = detection(r1(:,iter+1), r2(:,iter+1), r3(:,iter+1),...
%         r4(:,iter+1), r5(:,iter+1), r6(:,iter+1));
    
    % current location of other vehicle 
    act_X = interp1(ref.t_act, ref.act_x, tsim(iter+1));
    act_Y = interp1(ref.t_act, ref.act_y, tsim(iter+1));
    act = [act_X; act_Y]; act = act(:)';
    actorX = [actorX act_X(1)];
    actorY = [actorY act_Y(1)]; 
    
    tsim(iter+2) = tsim(iter+1)+dt;
    iter = iter + 1;
    
end 

%% Animation

% th = 0:pi/50:2*pi;
% 
% for i=1:length(actorY)
%     
%     f = figure(1);
%     f.Position(3:4) = [520 985];
%     movegui(f,'south');
%     plot(xx(6,i), xx(5,i), 'o','MarkerFaceColor','b');
%     hold on
%     plot(5*sin(th)+xx(6,i),  5*cos(th)+xx(5,i), 'b-');
%     hold on
%     plot(ref.ego_Y, ref.ego_X, 'g-','LineWidth',1.5);
%     hold on
%     plot(actorY(1,i), actorX(1,i), 'o','MarkerFaceColor','r');
%     hold on
%     plot(5*sin(th)+actorY(1,i),  5*cos(th)+actorX(1,i), 'r-');
%     hold on
%     plot(ref.lb1_y, ref.lb1_x, 'k-','LineWidth', 1.5);
%     hold on
%     plot(ref.lb2_y, ref.lb2_x, 'k-','LineWidth', 1.5);
%     grid on; box on;
%     xlabel('Y [m]'); ylabel('X [m]');
%     xlim([-40 40]); ylim([-20 140]);
%     hold off;
%     pause(.1);
%     
% end
% 
% %% Plots 
% 
% figure(2)
% plot(xx(6,:), xx(5,:),'b-','LineWidth',1.5);
% hold on
% plot(x_est(6,:), x_est(5,:),'r--','LineWidth',1.5);
% grid on; box on; 

% Residual
figure(3)
subplot(611)
plot(tsim(1:end-1), r1);
grid on; box on;
subplot(612)
plot(tsim(1:end-1), r2);
grid on; box on;
ylim([-10 10]);
subplot(613)
plot(tsim(1:end-1), r3);
grid on; box on;
ylim([-10 10]);
subplot(614)
plot(tsim(1:end-1), r4);
grid on; box on;
ylim([-10 10]);
subplot(615)
plot(tsim(1:end-1), r5);
grid on; box on;
ylim([-10 10]);
subplot(616)
plot(tsim(1:end-1), r6);
grid on; box on;
ylim([-10 10]);

%%