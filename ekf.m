function [x_hat, P] = ekf(F, x_prev, u_curr, y, p, P, H)


    x_prior = F(x_prev, u_curr);
    
    A = [-(p.f1 + 2*p.f2*x_prev(1,1)) 0 0 0 0 0;...
        ((2*p.Cf+2*p.Cr)/(p.m*x_prev(1,1)^2))*x_prev(2,1)+((((-2*p.Cr*p.lr)+(2*p.Cf*p.lf))/(p.m*x_prev(1,1)^2))-1)*x_prev(4,1)...
         (-(2*p.Cf+2*p.Cr)/(p.m*x_prev(1,1))) 0 ((((2*p.Cr*p.lr)-(2*p.Cf*p.lf))/(p.m*x_prev(1,1)))-x_prev(1,1)) 0 0;...
        0 0 0 1 0 0;...
        ((-2*p.Cr*p.lr+2*p.Cf*p.lf)/(p.Izz*x_prev(1,1)^2))*x_prev(2,1)+(((2*p.Cf*p.lf^2)+(2*p.Cr*p.lr^2))/(x_prev(1,1)^2*p.Izz))*x_prev(4,1)...
         ((2*p.Cr*p.lr-2*p.Cf*p.lf)/(p.Izz*x_prev(1,1))) 0 (-((2*p.Cf*p.lf^2)+(2*p.Cr*p.lr^2))/(x_prev(1,1)*p.Izz)) 0 0;...
        cos(x_prev(3,1)) -sin(x_prev(3,1)) -x_prev(1,1)*sin(x_prev(3,1))-x_prev(2,1)*cos(x_prev(3,1)) 0 0 0;...
        sin(x_prev(3,1)) cos(x_prev(3,1)) x_prev(1,1)*cos(x_prev(3,1))-x_prev(2,1)*sin(x_prev(3,1)) 0 0 0];
    
    R = 1e-2.*eye(size(H,1));
    
    P_prior = A*P*A';
    
    K = P_prior*H'*inv((H*P_prior*H')+R);
    
    x_hat = x_prior + (K*(y -(H*x_prior))); 
    P = (eye(6)-(K*H))*P_prior;

end 