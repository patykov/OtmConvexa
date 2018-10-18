% Questao 4.11

%% a) Plot f(x)
[x1,x2] = meshgrid(-pi:.2:pi);
fx = 0.7*(x1).^4 -8*(x1).^2 +6*(x2).^2 +cos(x1*x2) -8*x1;

h1 = figure();
surf(x1,x2,fx)
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
zlabel('f(x)', 'FontSize', 20)
save_pdf(h1, 'q4_11a');
pause;


%% b) Contour plot of f(x)
h2 = figure();
contour(x1,x2,fx)
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
save_pdf(h2, 'q4_11b');
pause;


%% d) Fletcher’s inexact line search

x0 = [-pi;pi];
d0 = [1.0;-1.3];

[alphas_star, min_xs, min_fs] = fletcher_inexact_line_search(x0,d0);

h3 = figure();
contour(x1,x2,fx); hold on;
scatter(min_xs(1,:), min_xs(2,:))
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
save_pdf(h3, 'q4_11d-contour_alphas_1');
pause;

all_alphas = 0:0.01:4.8332;

y1 = zeros(1, length(all_alphas));
for i=1:length(all_alphas)
    alpha = all_alphas(i);
    y1(i) = f(x0 + alpha*d0);
end

h4 = figure();
plot(all_alphas, y1)
xlabel('$\alpha$','Interpreter','latex', 'FontSize', 20)
ylabel('$f(x_0 + \alpha d_0)$','Interpreter','latex', 'FontSize', 20)
save_pdf(h4, 'q4_11d-function');
pause;


all_alphas_star = [];
all_min_xs = [];
all_min_fs = [];
for k=1:10
    [alphas_star, min_xs, min_fs] = fletcher_inexact_line_search(x0,d0);
    
    if length(all_min_fs) > 0
        if (max(min_fs) > min(all_min_fs))
            break;
        end
    end
    all_alphas_star = cat(1, all_alphas_star, alphas_star);
    all_min_xs = cat(2, all_min_xs , min_xs);
    all_min_fs = cat(1, all_min_fs, min_fs);
    x0 = min_xs(:, end);

end

h5 = figure();
contour(x1,x2,fx); hold on;
scatter(all_min_xs(1,:), all_min_xs(2,:))
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
save_pdf(h5, 'q4_11d-contour_alphas_all');
pause;



%% e) Repeat (d) for different values

x0 = [-pi;pi];
d0 = [1.0;-1.1];

[alphas_star, min_xs, min_fs] = fletcher_inexact_line_search(x0,d0);

h6 = figure();
contour(x1,x2,fx); hold on;
scatter(min_xs(1,:), min_xs(2,:))
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
save_pdf(h6, 'q4_11e-contour_alphas_1');
pause;

all_alphas = 0:0.01:4.8332;

y1 = zeros(1, length(all_alphas));
for i=1:length(all_alphas)
    alpha = all_alphas(i);
    y1(i) = f(x0 + alpha*d0);
end

h7 = figure();
plot(all_alphas, y1)
xlabel('$\alpha$','Interpreter','latex', 'FontSize', 20)
ylabel('$f(x_0 + \alpha d_0)$','Interpreter','latex', 'FontSize', 20)
save_pdf(h7, 'q4_11e-function');
pause;


all_alphas_star = [];
all_min_xs = [];
all_min_fs = [];
for k=1:10
    [alphas_star, min_xs, min_fs] = fletcher_inexact_line_search(x0,d0);
    
    if length(all_min_fs) > 0
        if (max(min_fs) > min(all_min_fs))
            break;
        end
    end
    all_alphas_star = cat(1, all_alphas_star, alphas_star);
    all_min_xs = cat(2, all_min_xs , min_xs);
    all_min_fs = cat(1, all_min_fs, min_fs);
    x0 = min_xs(:, end);

end

h8 = figure();
contour(x1,x2,fx); hold on;
scatter(all_min_xs(1,:), all_min_xs(2,:))
xlabel('x1', 'FontSize', 20)
ylabel('x2', 'FontSize', 20)
save_pdf(h8, 'q4_11e-contour_alphas_all');
pause;

close all;

%% c) Compute the gradient of f(x)

function gradient = g(x)
gradient = [(14*x(1)^3)/5 - x(2)*sin(x(1)*x(2)) - 16*x(1) - 8;
      12*x(2) - x(1)*sin(x(1)*x(2))];
end

function y = f(x)
    y = 0.7*(x(1))^4 -8*(x(1))^2 +6*(x(2))^2 +cos(x(1)*x(2)) -8*x(1);
end