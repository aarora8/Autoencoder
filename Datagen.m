clear
clc
 
h = 256;
S = 4;
n = 100;
m_1 = -1/4096;

A_star = randn(n,h);
A_star = orth(A_star')';

for i =1:h
    colnorm=sqrt(sum(A_star(:,i).^2,1));
    A_star(:,i) = A_star(:,i)./colnorm;
end

coherence_mat = A_star'*A_star;
mu_max= 0;
for i = 1:n
    for j = 1:h
        if(i~=j)
            if(abs(coherence_mat(i,j))>mu_max)
                mu_max = abs(coherence_mat(i,j));
            end
        end
    end
end
Num_datapoints = 7200;
 
Y_mat = zeros(n,7000);
X_mat = zeros(h,7000);
Y_test = zeros(n,200);
X_test = zeros(h,200);
var_x_star = 1/(h*log(n));
for i = 1:Num_datapoints  
    x = zeros(h,1);
    x(1:S) = normrnd(- 1/4096,var_x_star,[S 1]);
    if(i<=7000)
        y = A_star*x;
        Y_mat(:,i) = y;
        X_mat(:,i) = x;
    end
    if(i>7000)
        y = A_star*x;
        Y_test(:,i-7000) = y;
        X_test(:,i-7000) = x;
    end
end
mu_by_root_n = mu_max;

clear x y i j mu_max var_x_star Num_datapoints coherence_mat colnorm S n h 

