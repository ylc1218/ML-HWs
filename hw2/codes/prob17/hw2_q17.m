function [avg_ein] = hw2_q17()
    clc
    clear

    minval = -1;
    maxval = 1;
    N = 20; %datasize
    noise_prob = 0.2;
    T=5000;
    
    vec_ein = zeros(1, T);
    
    for t=1:T
        in = (maxval-minval).*rand(N,1) + minval; %input
        out = sign(in); %y

        %add noise
        err_id = randperm(N,N*noise_prob);
        out(err_id) = out(err_id)*-1;

        ein_min=1;
        for i = 1:N
            theta = in(i);

            %s=1
            s=1;
            ein = cal_error(in, out, theta, s);
            if ein<ein_min
                ein_min=ein;
            end

            %s=-1
            s=-1;
            ein = cal_error(in, out, theta, s);
            if ein<ein_min
                ein_min=ein;
            end
        end
        vec_ein(t) = ein_min;
    end %end of loop T
    avg_ein = sum(vec_ein)/T;
    
    %histogram
    hist(vec_ein);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    
    title('Histogram of Hw2-q17');
    xlabel('Error Rate');
    ylabel('Frequency');
end


function ein = cal_error(in, out, theta, s)
    [N, ~]= size(in);
    myout = s*sign(in-theta);
    myout(myout==0) = s; % % take sign(0) = s
    diff = myout.*out; %if(same)->1, else->-1
    ein = sum(diff == -1)./N; % (cnt #-1)/N
end


