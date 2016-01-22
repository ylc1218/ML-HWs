function root=hw7_q13()
    [train_x,train_y]=train_input();
    
    root = hw7_buid_tree(train_x, train_y);
    print_node(root, 1);
end

function [train_x,train_y]=train_input()
    train_data = textread('hw7_train.dat');
    train_x=train_data(:,1:end-1);
    train_y=train_data(:,end);
end

function print_node(node, layer)
    for i= 1:layer-1
        fprintf('\t');
    end
    
    if(node.isLeaf~=true)
        fprintf('(%d %d %f)\n', node.d, node.s, node.theta);
        print_node(node.lch, layer+1);
        print_node(node.rch, layer+1);
    else
        fprintf('leaf: %d\n', node.val);
    end
end

