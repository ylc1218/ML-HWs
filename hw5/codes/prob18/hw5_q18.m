function arr_dist=hw5_q18()
    global gamma;
    gamma=100;
    digit=0;
    
    [train_x, train_y] = train_input();
    train_y(train_y~=digit)=-1;
    train_y(train_y==digit)=1;
    
    arr_log_c=[-3,-2,-1,0,1];
    cnt=size(arr_log_c,2);
    
    arr_dist=zeros(1,cnt);
    for i=1:cnt
        c=10^arr_log_c(i);
        para = sprintf('-c %g -t 2 -g %g', c,gamma);
        model = svmtrain(train_y,train_x, para);

        %w = model.SVs' * model.sv_coef;
        b = -model.rho;

        sv_idx=model.sv_indices;
        alpha_y=model.sv_coef;

        K=zeros(model.totalSV,model.totalSV);

        for n=1:model.totalSV
            for m=n:model.totalSV
                K(n,m)=kernel(train_x(sv_idx(n),:).',train_x(sv_idx(m),:).');
                K(m,n)=K(n,m);
            end
        end
        tmp=(alpha_y*alpha_y.').*K;
        norm_w=sqrt(sum(sum(tmp)));

        free_sv_idxs=sv_idx(alpha_y<c & alpha_y>-c);
        fsvid=free_sv_idxs(1);
        fsvx=train_x(fsvid,:);
        val=b;
        for n=1:model.totalSV
            val=val+alpha_y(n)*kernel(train_x(sv_idx(n),:).', fsvx.');
        end
        %{
            svX=train_x(sv_idx,:).';
            fsvx=repmat(train_x(fsvid,:).',1,model.totalSV);

            tmp=svX-fsvx;
            val=sum(exp(-gamma*(sum(tmp.^2))))+b;
        %}
        arr_dist(i)=abs(val)/norm_w;
    end
    
    xmarkers = arr_log_c; % place markers at these x-values
    ymarkers = arr_dist;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k*','LineWidth',2,'MarkerSize',10);
    title('Hw5-q18');
    xlabel('log_{10}c');
    ylabel('Distance');


end

function [train_x, train_y] = train_input()
    train_x = textread('features.train');
    train_y = train_x(:, 1);
    train_x = train_x(:, 2:end);
end

function val=kernel(x1,x2)
    global gamma;
    x=x1-x2;
    val = exp(-gamma* (x.'*x));
end