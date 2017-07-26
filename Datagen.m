clear
clc
 
N = 256;
S = 4;
M = 100;
 
 
A_star = randn(M,N);
A_star = orth(A_star')';
% matrixNorm = A_star.'*A_star;
% matrixNorm = sqrt(diag(matrixNorm)).';
% A_star = A_star./repmat(matrixNorm, [M,1]);

for i =1:N
    colnorm=sqrt(sum(A_star(:,i).^2,1));
    A_star(:,i) = A_star(:,i)./colnorm;
end
 
A = A_star;
coherence_mat = A'*A;
mu_max= 0;row_max = -1; col_max = -1;
for i = 1:M
    for j = 1:N
        if(i~=j)
            if(abs(coherence_mat(i,j))>mu_max)
                mu_max = abs(coherence_mat(i,j));
                row_max = i;
                col_max = j;
            end
        end
    end
end
Num_datapoints = 7200;
 
Y_mat = zeros(M,7000);
X_mat = zeros(N,7000);
Y_test = zeros(M,200);
X_test = zeros(N,200);
var_x_star = 1/(N*log(M));
for i = 1:Num_datapoints  
    x = zeros(N,1);
    x(1:S) = normrnd(- 1/4096,var_x_star,[S 1]);
%     x(1:S) = x(1:S) - 1/256;
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


m_1 = -1/4096;
var_x = var_x_star;
m_2 = var_x_star + m_1^2;
mu_by_root_n = mu_max/sqrt(M);

