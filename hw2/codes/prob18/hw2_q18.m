function [avg_eout] = hw2_q18()
    clc
    clear

    minval = -1;
    maxval = 1;
    N = 20; %datasize
    noise_prob = 0.2;
    T=5000;
    
    vec_eout = zeros(1, T);
    
    for t=1:T
        in = (maxval-minval).*rand(N,1) + minval; %input
        out = sign(in); %y

        %add noise
        err_id = randperm(N,N*noise_prob);
        out(err_id) = out(err_id)*-1;

        ein_min=1;
        s_best=0;
        theta_best=0;
        for i = 1:N
            theta = in(i);

            %s=1
            s=1;
            ein = cal_error(in, out, theta, s);
            if ein<ein_min
                ein_min=ein;
                s_best=s;
                theta_best=theta;
            end

            %s=-1
            s=-1;
            ein = cal_error(in, out, theta, s);
            if ein<ein_min
                ein_min=ein;
                s_best=s;
                theta_best=theta;
            end
        end
        vec_eout(t) = 0.5+0.3*s_best*(abs(theta_best)-1);
    end %end of loop T
    avg_eout = sum(vec_eout)/T;
    
    %histogram
    hist(vec_eout);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    
    title('Histogram of Hw2-q18');
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


