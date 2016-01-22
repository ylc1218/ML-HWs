function ein_arr=hw7_q16()
    [train_x,train_y]=train_input();
    T=30000;
    N=size(train_x,1);
    
    ein_arr = zeros(1,T);
    for t=1:T
        [bag_train_x, bag_train_y] = bagging(train_x, train_y, N);
        root = hw7_buid_tree(bag_train_x, bag_train_y);
        ein_arr(t)=cal_error(root, train_x, train_y)/N;
    end
%{    
    hist(ein_arr);
    h = findobj(gca,'Type','patch');
    grey=[0.4,0.4,0.4];
    set(h,'FaceColor',grey)
    title('Histogram of Hw7-q16');
    xlabel('E_{in}');
    ylabel('Frequency');
%}    
end

function [train_x,train_y]=train_input()
    train_data = textread('hw7_train.dat');
    train_x=train_data(:,1:end-1);
    train_y=train_data(:,end);
end

function error = cal_error(node, in, out)
    if(node.isLeaf)
        error=sum(out~=node.val);
    else
        stump_out = node.s*sign(in(:,node.d)-node.theta);
        error=cal_error(node.lch, in(stump_out==-1,:), out(stump_out==-1));
        error=error+cal_error(node.rch, in(stump_out==1,:), out(stump_out==1));
    end
end

function [bag_train_x, bag_train_y]=bagging(x, y, N)
    idx = randi([1 size(x,1)],1,N);
    [bag_train_x, bag_train_y] = deal(x(idx,:), y(idx));  
end