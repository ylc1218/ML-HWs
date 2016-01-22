function [avg_update, tbl]= hw1_q17()
    [train_x, train_y] = train_input();
    [N_train, D] =  size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    T=2000;
    
    vec_update = zeros(1, T);
    for t=1:T
        cycle_id = randperm(N_train); %random cycle
        update=0;
        stop=0;
        w=zeros(1, D+1);
        now = 0;

        % PLA
        while stop==0
            stop=1;
            for i=now:now+N_train-1 %cycle
                id = cycle_id(mod(i, N_train)+1);
                x = train_x_app(id, :);
                y = train_y(id);                     
                s = sign(x*w.');

                s(s==0)=-1; %treat sign(0)=-1

                if(s~=y) %mistake
                    stop=0;
                    update=update+1; %count #mistake

                    w = w +0.5*y*x; %update w
                    now = mod(i+1, N_train); %update next
                    break;
                end
            end        
        end %PLA
        vec_update(t) = update;
    end %T=2000
    avg_update = sum(vec_update)/T;
    hist(vec_update);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    title('Histogram of Hw1-q17');
    xlabel('Updates');
    ylabel('Frequency');
    
    tbl = tabulate(vec_update);
end

function [train_x, train_y] = train_input()
    train_x = textread('hw1_15_train.dat');
    train_y = train_x(:, end);
    train_x = train_x(:, 1:end-1);
end