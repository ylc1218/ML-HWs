function [avg_error, tbl]= hw1_q18()
    [train_x, train_y] = train_input();    
    [N_train, D] =  size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    T=2000;
    U=50; % #update
    
    [test_x, test_y] = test_input();
    vec_eout = zeros(1, T);
    
    for t=1:T              
        w=zeros(1, D+1);        

        % PLA (pocket)
        best_err = 1;
        for i=1:U
            while true % find random mistake
                id = randi(N_train);
                x = train_x_app(id, :);
                y = train_y(id) ;              
                s = sign(x*w.');
                
                s(s==0)=-1; %tread sign(0)=-1

                if(s~=y) %mistake
                    w = w +y*x; %update w
                    e = cal_error(test_x, test_y, w);
                    if e<best_err %update best
                        best_err = e;
                    end
                    
                    break;
                end
            end        
        end %PLA
        vec_eout(t) = best_err;
        
    end %T=2000
    avg_error = sum(vec_eout)/T;
    
    %histogram
    hist(vec_eout);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    title('Histogram of Hw1-q18');
    xlabel('Error Rate');
    ylabel('Frequency');
    
    tbl = tabulate(vec_eout);
end

function [train_x, train_y] = train_input()
    train_x = textread('hw1_18_train.dat');
    train_y = train_x(:, end);
    train_x = train_x(:, 1:end-1);
end

function [test_x, test_y] = test_input()
    test_x = textread('hw1_18_test.dat');
    test_y = test_x(:, end);
    test_x = test_x(:, 1:end-1);
end

function err = cal_error(x, y, w)
    [N, D] =  size(x);
    x_app = [ones(N, 1) x];
    
    myY = sign(x_app*w.');
    myY( myY==0 )=-1; % take sign(0) = -1 
    sub = myY.*y;
    err = sum(sub~=1)/N;
    
end