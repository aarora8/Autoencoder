N = size(Y_mat,2);
h = size(X_mat,1);
eta = 0.003; 
delta = 0.8;
epsilon_i = abs(m_1)*S*(delta+mu_by_root_n);
lambda_2 = -1;
C = (1 - delta)^2;
S = 4;

q_i = S/h;
q_j = S/h;
term_l1_1 = C*h*S;
term_l1_2 = h*q_i*(1 - delta)^2;
lambda_1 = term_l1_1 + term_l1_2;
g_mat = zeros(size(X_mat,1),size(Y_mat,1)); 
W = zeros(size(X_mat,1),size(Y_mat,1)); 

var_x_star = 1;
W_T = W';
% W_tilda = zeros(size(X_mat,1),size(Y_mat,1)); 
% W_tilda = W(:,1:4);
% W*y - epsilon_i;
for i =1:size(X_mat,1)
    W1 = normrnd(0,var_x_star,[size(Y_mat,1),1]);
    colnorm=sqrt(sum(W1.^2,1));
    W1 = (2)*W1./colnorm;
    W_T(:,i) = A_star(:,i) - W1;
end

W = W_T';
diff = W'*X_test(:,1) - A_star*X_test(:,1);
diff_norm = norm(diff,2);

W_diff = zeros(size(X_mat,1),1);
for i =1:size(X_mat,1)
    W1 = W_T(:,i) - A_star(:,i);
    colnorm=sqrt(sum(W1.^2,1));
    W_diff(i,1) = colnorm;
end

gradient_val = [];
gmat_val = [];
num_iter = 15;
W_diff2 = zeros(size(X_mat,1),num_iter);
for iter =1:num_iter 
    iter
    g_mat = zeros(size(X_mat,1),size(Y_mat,1));
    for i= 1:S
        final_term =zeros(size(Y_mat,1),1);
        regularization_term_2= zeros(size(Y_mat,1),1);
        W_T = W';
        if(i<=S)
            for k=1:N
                term_1 = (W_T(:,i)'*Y_mat(:,k) - epsilon_i).*eye(size(Y_mat,1));
                term_2 = (W_T(:,i)*Y_mat(:,k)');
                term12 = term_1 + term_2;
                term_jh = zeros(size(Y_mat,1),1);
                for j=1:h
                    term_wTY = W_T(:,j)'*Y_mat(:,k) - epsilon_i;
                    term_jh = term_jh+ (term_wTY)* W_T(:,j);
                end
                term_CHY = C*h*Y_mat(:,k);
                termjh_chy = term_jh - term_CHY;
                term_prod_ab = term12 * termjh_chy;
                final_term = final_term + term_prod_ab;
            end
            final_term = (1/N)*final_term;
        end 
        regularization_term_1  = 2*lambda_1*W_T(:,i);
        for k=1:N
            W_tilda = zeros(size(X_mat,1),size(Y_mat,1)); 
            W_tilda(1:4,:) = W(1:4,:);
            term_wy = norm(W_tilda*Y_mat(:,k),2)^2;
            term_aa = lambda_2 *term_wy*W_T(:,i);
            fnorm = 0;
            for i1 =1:4
                w1 = W_tilda(i1,:);
                rownorm=sum(W1.^2,1);
                fnorm = fnorm+ rownorm;
            end
            term_ab = lambda_2*fnorm*W_T(:,i)'*Y_mat(:,k)*Y_mat(:,k);
            regularization_term_2 = regularization_term_2+term_aa+ term_ab;
        end
        regularization_term_2 = (1/N)*regularization_term_2;
        g_i = final_term + regularization_term_1 + regularization_term_2;
        g_mat(i,:) = g_i';
        colnorm=sqrt(sum(g_i.^2,1));
        if(i == 1)
            gradient_val = [gradient_val colnorm];
        end
        
        if(gradient_val(1,iter)/gradient_val(1,1)<0.01)
            break;
        end        
    end
    if(gradient_val(1,iter)/gradient_val(1,1)<0.01)
            break;
    end  
    gmat_val = [gmat_val sqrt(diag(g_mat*g_mat'))];
    W = W-eta*g_mat;
    W_T = W';
    for i =1:size(X_mat,1)
        W1 = W_T(:,i) - A_star(:,i);
        colnorm=sqrt(sum(W1.^2,1));
        W_diff2(i,iter) = colnorm;
   end
end

W_T_before_norm = W';
diff = W_T_before_norm*X_test(:,1) - A_star*X_test(:,1);
diff_norm1 = norm(diff,2);


W_diff1 = zeros(size(X_mat,1),1);
for i =1:size(X_mat,1)
    W1 = W_T_before_norm(:,i) - A_star(:,i);
    colnorm=sqrt(sum(W1.^2,1));
    W_diff1(i,1) = colnorm;
end

W_T = W';
for i =1:size(X_mat,1)
    colnorm=sqrt(sum(W_T(:,i).^2,1));
    W_T(:,i) = W_T(:,i)./colnorm;
end
W = W_T';
diff = W'*X_test(:,1) - A_star*X_test(:,1);
diff_norm2 = norm(diff,2);


