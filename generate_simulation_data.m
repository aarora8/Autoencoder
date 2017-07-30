% generating data for experiments with support =4
clear; clc;
h = 512;
n = 100;
A_star = randn(n,h);
for i =1:h
    colnorm=sqrt(sum(A_star(:,i).^2,1));
    A_star(:,i) = A_star(:,i)./colnorm;
end

coherence_mat = A_star'*A_star;
mu_by_root_n= 0;
for i = 1:n
    for j = 1:h
        if(i~=j)
            if(abs(coherence_mat(i,j))>mu_by_root_n)
                mu_by_root_n = abs(coherence_mat(i,j));
            end
        end
    end
end

% creating a randomly initialised  weight matrix
% this weight matrix will be in the columnwise ball distance of A_star
W = zeros(h,n); % weight matrix
var_weight = 1; 
W_T = W';
ball_distance = 2;
for i =1:h
    W1 = normrnd(0,var_weight,[n,1]);
    colnorm=sqrt(sum(W1.^2,1));
    W1 = (ball_distance)*W1./colnorm;
    W_T(:,i) = A_star(:,i) - W1;
end
W_T_initial = W_T;
W_initial = W_T';
clear x y i j coherence_mat colnorm var_weight W1 W W_T
result = strcat('../simulation_data_',int2str(h),'.mat');
save (result,'n','h','mu_by_root_n','A_star', 'W_initial','W_T_initial','ball_distance'); 
