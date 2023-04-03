function a1_20060920
% CISC371, Fall 2020, Assignment #1, Question #1: scalar optimization

% Anonymous functions for objective functions and gradients
f1 =@(t) (exp(3*t)+5*exp(-2*t));
g1 =@(t) (3*exp(3*t)-10*exp(-2*t));
h1 =@(t) (9*exp(3*t)+20*exp(-2*t));
f2 =@(t) (log(t).^2-2+log(10-t).^2-t.^(0.2));
g2 =@(t) ((10*log(10 - t) + 10*log(t))*t + 10*t^(1/5) - 100*log(t) - t^(6/5))/(5*t^2 - 50*t);
h2 =@(t) ((100 - 50*log(t) - 50*log(10 - t))*t^2 + (1000*log(t) - 1000)*t + 400*t^(1/5) - 5000*log(t) - 80*t^(6/5) + 4*t^(11/5) + 5000)/(25*t^4 - 500*t^3 + 2500*t^2);
f3 =@(t) (-3*t.*sin(0.75*t)+exp(-2*t));
g3 =@(t) (- 2*exp(-2*t) - 3*sin((3*t)/4) - (9*t*cos((3*t)/4))/4);
h3 =@(t) (4*exp(-2*t) - (9*cos((3*t)/4))/2 + (27*t*sin((3*t)/4))/16);

% Unify objective and gradient for standard call to optimization code
fg1  =@(t) deal((f1(t)), (g1(t)));
fgh1 =@(t) deal((f1(t)), (g1(t)), (h1(t)));
fg2  =@(t) deal((f2(t)), (g2(t)));
fgh2 =@(t) deal((f2(t)), (g2(t)), (h2(t)));
fg3  =@(t) deal((f3(t)), (g3(t)));
fgh3 =@(t) deal((f3(t)), (g3(t)), (h3(t)));
fg3A  =@(t) deal((f3A(t)), (g3A(t)));
fgh3A =@(t) deal((f3A(t)), (g3A(t)), (h3A(t)));


% Compute the quadratic approximations and search estimates
[fcoef_f1_quad, tstat_f1_quad] = quadapprox(fgh1, 1)
[fcoef_f2_quad, tstat_f2_quad] = quadapprox(fgh2, 9.9)
[fcoef_f3_quad, tstat_f3_quad] = quadapprox(fgh3, 2*pi)


[tmin_f1_fixed,fmin_f1_fixed,ix_f1_fixed] = steepfixed(fg1, 1, 1/100)
[tmin_f2_fixed,fmin_f2_fixed,ix_f2_fixed] = steepfixed(fg2, 9.9, 0.039)
[tmin_f3_fixed,fmin_f3_fixed,ix_f3_fixed] = steepfixed(fg3, 2*pi, pi/50)

[tmin_f1_back,fmin_f1_back,ix_f1_back] = steepline(fg1, 1, 1/10, 0.5, 0.5)
[tmin_f2_back,fmin_f2_back,ix_f2_back] = steepline(fg2, 9.9, 0.39, 0.5, 0.5)
[tmin_f3_back,fmin_f3_back,ix_f3_back] = steepline(fg3, 2*pi, pi/5, 0.5, 0.5)

%% PLOT FUNCTIONS
% plot f1
f1_b1 = 0;
f1_b2 = 1;
tv = linspace(-10, 10, 500);
figure(1)
axis([0 1.01 4 22]);
hold on;
plot(tv,f1(tv),'b-', tv, polyval(fcoef_f1_quad,tv), 'r-'); % plot f1 and quadratic
plot(f1_b2, f1(f1_b2), 'k+'); %plot for initial estimate
plot(tstat_f1_quad, f1(tstat_f1_quad), 'r*') % plot quadratic estimate
plot(tmin_f1_fixed, f1(tmin_f1_fixed), 'g*') % plot fixed step estimate
plot(tmin_f1_back, f1(tmin_f1_back), 'y*') % plot backtracking estimate
plot(fminbnd(f1, f1_b1, f1_b2), f1(fminbnd(f1, f1_b1, f1_b2)), 'ko') % plot actual local minimizer
title("f1(t) (blue) and local quadratic (red)")
xlabel('t');
ylabel('f(t)');

% plot f2
f2_b1 = 6;
f2_b2 = 9.9;
tv = linspace(0, 10, 500);
figure(2)
axis([0 10 0 10]);
hold on;
plot(tv,f2(tv),'b-', tv, polyval(fcoef_f2_quad,tv), 'r-'); % plot f2 and quadratic
plot(f2_b2, f2(f2_b2), 'k+'); %plot for initial estimate
plot(tstat_f2_quad, f2(tstat_f2_quad), 'r*') % plot quadratic estimate
plot(tmin_f2_fixed, f2(tmin_f2_fixed), 'g*') % plot fixed step estimate
plot(tmin_f2_back, f2(tmin_f2_back), 'y*') % plot backtracking estimate
plot(fminbnd(f2, f2_b1, f2_b2), f2(fminbnd(f2, f2_b1, f2_b2)), 'ko') % plot actual local minimizer
title("f2(t) (blue) and local quadratic (red)")
xlabel('t');
ylabel('f(t)');

% plot f3
f3_b1 = 0;
f3_b2 = 2*pi;
tv = linspace(-10, 10, 500);
figure(3)
axis([0 8 -8 25]);
hold on;
plot(tv,f3(tv),'b-', tv, polyval(fcoef_f3_quad,tv), 'r-'); % plot f3 and quadratic
plot(f3_b2, f3(f3_b2), 'k+'); %plot for initial estimate
plot(tstat_f3_quad, f3(tstat_f3_quad), 'r*') % plot quadratic estimate
plot(tmin_f3_fixed, f3(tmin_f3_fixed), 'g*') % plot fixed step estimate
plot(tmin_f3_back, f3(tmin_f3_back), 'y*') % plot backtracking estimate
plot(fminbnd(f3, f3_b1, f3_b2), f3(fminbnd(f3, f3_b1, f3_b2)), 'ko') % plot actual local minimizer
title("f3(t) (blue) and local quadratic (red)")
xlabel('t');
ylabel('f(t)');

end

function [fcoef, tstat] = quadapprox(funfgh, t1)
% [FCOEF,TSTAT]=QUADAPPROX(FUNFGH,T1) finds the polynomial coefficients
% FCOEF of the quadratic approximation of a function at a scalar point
% T1, using the objective value, gradient, and Hessian from unction FUNFGH
% at T1 to complete the approximation. FCOEF is ordered for use in POLYVAL.
% The stationary point of the quadratic is returned as TSTAT.
%
% INPUTS:
%         FUNFGH - handle to 3-output function that computed the
%                  scalar-valued function, gradient, and 2nd derivative
%         T1     - scalar argument
% OUTPUTS:
%         FCOEF  - 1x3 array of polynomial coefficients
%         TSTAT  - stationary point of the approximation
% ALGORITHM:
%     Set up and solve a 3x3 linear equation. If the points are colinear
%     then TSTAT is empty

% Initialize the outputs
fcoef = [];
tstat = [];

% set up a linear equation
[f d1 d2] = funfgh(t1);

% generating 3x3 matrix based on a^2 + a + 1
m3 = [t1(:).^2 t1(:).^1 t1(:).^0];
m3(2, :) = [2*t1 1 0]; % replace second row with first deriv
m3(3, :) = [2 0 0]; % replace third row with second deriv
% generating function matrix
b3 = f;
b3(2) = d1;
b3(3) = d2;
b3 = b3';
% perform matrix left division 
fcoef = (m3 \ b3)';
% calculate estimation 
tstat = -fcoef(2)/(2*fcoef(1));

end
function [tmin,fmin,ix]=steepfixed(objgradf,t0,s,imax_in,eps_in)
% [TMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,T0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point T0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of minimizer
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% set convergence criteria to those supplied, if available
if nargin >= 4 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end


if nargin >= 5 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search point, objective, gradient
tmin = t0;
[fmin gval] = objgradf(tmin);
ix = 0;

while (norm(gval)>epsilon & ix<imax)
    d = -gval; % calculate direction
    tmin = tmin + s*d; % calculate next step
    [fmin gval] = objgradf(tmin); % get function and gradient at new step
    ix = ix + 1;
end
end

function [tmin,fmin,ix]=steepline(objgradf,t0,s0,beta,alpha,imax_in,eps_in)
% [TMIN,FMIN]=STEEPLINE(OBJGRADF,T0,S,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point T0 and using constant stepsize S. Backtracking is
% controlled by reduction ratio BETA. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         T0       - initial estimate of minimizer
%         S        - stepsize, positive scalar value
%         BETA     - backtracking hyper-parameter, 0<beta<1
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 6 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 7 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Limit BETA to the interval (0,1)
beta  = max(1e-6, min(1-(1e-6), beta));
alpha = max(1e-6, min(1-(1e-6), alpha));


% Initialize: objective, gradient, unit search vector
tmin = t0;
[fmin gofm] = objgradf(tmin);
dvec = -gofm;
wmat = tmin;
ix = 0;

% terminates when gradiant close to 0 or max iterations hit
while (abs(gofm)>epsilon & ix<imax)
    s = s0;
    alpha = gofm / 2; % update alpha 
    [fback ~] = objgradf(tmin + s*dvec); % get function at step value
    while (fback > fmin + alpha*s*dvec) % comparison
        s = beta*s; % scale step size
        [fback ~] = objgradf(tmin + s*dvec); % update function value 
    end
    tmin = tmin + s*dvec; % take new step
    [fmin gofm] = objgradf(tmin); % get function and gradient at new step
    dvec = -gofm; % get new direction
    ix = ix + 1;
end
end