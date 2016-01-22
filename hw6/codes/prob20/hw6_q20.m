function [best_out, best_g]=hw6_q20()
    train_N=400;
    [train_x, train_y, test_x, test_y] = input(train_N);
    
    best_out=1;
    for gamma=[32,2,0.125]
        kmatrix=kernel_matrix(train_x,gamma);
        for lambda=[0.001,1,1000]
            beta=inv(lambda*eye(train_N)+kmatrix)*train_y; %size(beta)=train_N*1
            err = cal_error(train_x, test_x, test_y, beta, gamma);
            if (err<best_out)
                best_out=err;
                best_g.gamma=gamma;
                best_g.lambda=lambda;
            end
        end
    end
end

function [train_x, train_y, test_x, test_y] = input(train_N)
    input_data = textread('hw2_lssvm_all.dat');
    train_x=input_data(1:train_N, 1:end-1);
    train_y=input_data(1:train_N, end);
    
    test_x=input_data(train_N+1:end, 1:end-1);
    test_y=input_data(train_N+1:end, end);
end

function k=kernel(x1,x2, gamma)
    x=x1-x2;
    k=exp(-gamma*(x*x.'));
end

function kmatrix = kernel_matrix(X,gamma)
    [N,~]=size(X);
    kmatrix=zeros(N,N);
    for i=1:N
        for j=1:i
            kmatrix(i,j)=kernel(X(i,:), X(j,:), gamma);
            kmatrix(j,i)=kmatrix(i,j);
        end
    end
end

function err = cal_error(ori_in, in, out, beta, gamma)
    N=size(out,1);
    ori_N=size(ori_in,1);
    myout=zeros(N,1);
    for n=1:N
        for on=1:ori_N
            myout(n)=myout(n)+beta(on)* kernel(ori_in(on,:),in(n,:), gamma);
        end
        myout(n)=sign(myout(n));
    end
    err=sum(myout~=out)/N;
end