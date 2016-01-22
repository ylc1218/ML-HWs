function [ein, d, theta, s, eout] = hw2_q20()
    clc
    clear 
    
    train_in = textread('hw2_train.dat');
    train_out = train_in(:, end);
    train_in = train_in(:, 1:end-1);

    [N_train, D] = size(train_in);
    
    
    ein_min = ones(1,D);
    s_best = zeros(1,D);
    theta_best = zeros(1,D);
    
    %choose a best theta, s for every d
    for d=1:D %each dim
        ein_min(d)=N_train;
        for n=1:N_train %choose one theta
            theta = train_in(n, d);
            for s=[-1,1] %choose s
                ein = cal_error(train_in(:, d), train_out, theta, s); %cal ein on dim d, with theta, s
                if ein<ein_min(d)
                    ein_min(d)=ein;
                    s_best(d)=s;
                    theta_best(d)=theta;
                end
            end
        end
    end
    
    %choose the best combination of [theta, s, d]
    
    [ein ,d] = min(ein_min);
    theta = theta_best(d);
    s = s_best(d);
    eout = test_eout(theta, s, d);
    
end

function eout = test_eout(theta, s, d)
    test_in = textread('hw2_test.dat');
    test_out = test_in(:, end);
    test_in = test_in(:, 1:end-1);
    eout = cal_error(test_in(:, d), test_out, theta, s);
end

function ein = cal_error(in, out, theta, s)
    [N, ~]= size(in);
    myout = s*sign(in-theta);
    myout(myout==0) = s; % % take sign(0) = s
    diff = myout.*out; %if(same)->1, else->-1
    ein = sum(diff == -1)./N; % (cnt #-1)/N
end