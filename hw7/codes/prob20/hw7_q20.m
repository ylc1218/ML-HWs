function eout_arr=hw7_q20()
    [train_x,train_y]=train_input();
    [test_x,test_y]=test_input();
    T=30000;
    train_N=size(train_x,1);
    test_N=size(test_x,1);
    
    eout_arr = zeros(1,T);
    label_cnt=zeros(test_N,1);
    for t=1:T
        [bag_train_x, bag_train_y] = bagging(train_x, train_y, train_N);
        root = hw7_buid_tree_prune(bag_train_x, bag_train_y);
        label = predict_label(root, test_x);
        label_cnt=label_cnt+label;
        eout_arr(t)=cal_error(label_cnt, test_y)/test_N;
    end
    
    %plot
    xmarkers = 1:T; % place markers at these x-values
    ymarkers = eout_arr;
    plot(xmarkers,ymarkers,'k',xmarkers,ymarkers,'k','LineWidth',2);
      
    title('Hw7-q20');
    xlabel('t');
    ylabel('E_{out}(G_t)');
    
end

function error = cal_error(label_cnt, y)
    myout=sign(label_cnt);
    error=sum(myout~=y);
end

function [train_x,train_y]=train_input()
    train_data = textread('hw7_train.dat');
    train_x=train_data(:,1:end-1);
    train_y=train_data(:,end);
end

function [test_x,test_y]=test_input()
    train_data = textread('hw7_test.dat');
    test_x=train_data(:,1:end-1);
    test_y=train_data(:,end);
end

function label = predict_label(node, in)
    label = zeros(size(in,1),1);
    if(node.isLeaf)
        label = ones(size(in,1),1).*node.val;
    else
        stump_out = node.s*sign(in(:,node.d)-node.theta);
        label(stump_out==-1) = predict_label(node.lch, in(stump_out==-1,:));
        label(stump_out==1) = predict_label(node.rch, in(stump_out==1,:));
    end
end

function [bag_train_x, bag_train_y]=bagging(x, y, N)
    idx = randi([1 size(x,1)],1,N);
    [bag_train_x, bag_train_y] = deal(x(idx,:), y(idx));  
end