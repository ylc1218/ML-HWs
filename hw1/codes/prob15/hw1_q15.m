function [update, last_mistake]= hw1_q15()
    [train_x, train_y] = train_input();
    [N_train, D] =  size(train_x);
    train_x_app = [ones(N_train, 1) train_x];
    
    update=0;
    stop=0;
    w=zeros(1, D+1);
    now = 0;
    last_mistake=0;
    
    % PLA
    while stop==0
        stop=1;
        for i=now:now+N_train-1 %cycle
            id = mod(i, N_train)+1;
            x = train_x_app(id, :);
            y = train_y(id);                     
            s = sign(x*w.');
                                    
            s(s==0)=-1; %treat sign(0)=-1
            if(s~=y) %mistake
                stop=0;
                update=update+1; %count #mistake
                last_mistake = id;
                
                w = w +y*x; %update w
                now = mod(i+1, N_train); %update next
                break;
            end
        end        
    end        
end

function [train_x, train_y] = train_input()
    train_x = textread('hw1_15_train.dat');
    train_y = train_x(:, end);
    train_x = train_x(:, 1:end-1);
end