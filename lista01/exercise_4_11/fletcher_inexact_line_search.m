

%%% Entradas:
%%% x0 - valor pr�vio de busca;
%%% d0 - dire��o da busca;

%%% Sa�das:
%%% alfa_estrela - alfa que minimiza f(x0+alfa*d0) para o 'x0' pr�vio.
%%% x1 - x que soluciona o problema;
%%% f_x0 - valor de f(x) para o x0 (valor pr�vio);
%%% f_x1 - valor de f(x) para o x encontrado acima;


function [alphas_star, min_xs, min_fs] = fletcher_inexact_line_search(x0,d0)
    max_iter = 100;

    % Step 1
    xk = x0;
    dk = d0;

    rho = 0.1;
    theta = 0.7;
    tau = 0.1;
    chi = 9;

    alpha_L = 0;
    alpha_U = 10^99;

    % Step 2
    fL = f(xk+alpha_L*dk);
    fL_ = g(xk+alpha_L*dk)'*dk;

    % Step 3
    g0 = g(x0);
    alpha_0 = (norm(g0)^2)/(g0'*H(x0)*g0);

    alphas_star = [];
    alphas_star = cat(1, alphas_star, alpha_0);

    for k=1:max_iter
        % Step 4
        f0 = f(xk+alpha_0*dk);
        
        % Step 5
        if(f0>fL+rho*(alpha_0-alpha_L)*fL_)
            
            if(alpha_0<alpha_U)
                alpha_U = alpha_0;
            end

            alfa_0_ = alpha_L + ((alpha_0-alpha_L)^2*fL_)/(2*(fL-f0+(alpha_0-alpha_L)*fL_));

            if(alfa_0_<alpha_L+tau*(alpha_U-alpha_L))
                alfa_0_ = alpha_L+tau*(alpha_U-alpha_L);
            end

            if(alfa_0_>alpha_U-tau*(alpha_U-alpha_L))
                alfa_0_ = alpha_U-tau*(alpha_U-alpha_L);
            end

            alpha_0 = alfa_0_;

            alphas_star = cat(1, alphas_star, alpha_0);

        else
            % Step 7
            f0_ = g(xk+alpha_0*dk)'*dk;

            if(f0_ < theta*fL_)

                delta_alfa_0 = ((alpha_0-alpha_L)*f0_)/(fL_-f0_);

                if(delta_alfa_0<tau*(alpha_0-alpha_L))
                    delta_alfa_0 = tau*(alpha_0-alpha_L);
                end

                if(delta_alfa_0>chi*(alpha_0-alpha_L))
                    delta_alfa_0 = chi*(alpha_0-alpha_L);
                end

                alfa_0_ = alpha_0 + delta_alfa_0;

                alpha_L = alpha_0;
                alpha_0 = alfa_0_;
                fL = f0;
                fL_ = f0_;

                alphas_star = cat(1, alphas_star, alpha_0);
         
            else
                % Step 8
                min_xs = [];
                min_fs = [];
                for i=1:length(alphas_star)
                    x = (xk + alphas_star(i)*dk);
                    min_xs = cat(2, min_xs, (x));
                    min_fs = cat(1, min_fs, f(x));
                end
                break

            end


        end

    end

end

function z = f(x)
    z = 0.7*(x(1))^4 -8*(x(1))^2 +6*(x(2))^2 +cos(x(1)*x(2)) -8*x(1);
end

function gradiente = g(x)
    gradiente = [
        (14*x(1)^3)/5 - x(2)*sin(x(1)*x(2)) - 16*x(1) - 8;
        12*x(2) - x(1)*sin(x(1)*x(2))
    ];
end

function hessiana = H(x)
    hessiana = [
        (42*x(1)^2)/5 - x(2)^2*cos(x(1)*x(2)) - 16,   - sin(x(1)*x(2)) - x(1)*x(2)*cos(x(1)*x(2));
         - sin(x(1)*x(2)) - x(1)*x(2)*cos(x(1)*x(2)), 12 - x(1)^2*cos(x(1)*x(2))
     ];
end