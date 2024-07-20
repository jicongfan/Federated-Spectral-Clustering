function Xmask = splitData2Clients(N, P, min_member)
% Content: randomly partition N data points into P clients and guarantee
%          that each client has at least one data point
% Method: plate insertion
% Input:
%       N: scalar, number of data points
%       P: scalar, number of clients
%       min_member: scalar, minimum number of data points of each clients
%
% Output:
%       Xmask: P by 1, data assignments
%
%

if isnumeric(min_member)
    shuffle = randperm(N);
    Xmask = cell(P, 1);
    if min_member > 1
        for i = 1: P
            Xmask{i, 1} = shuffle((min_member - 1)*(i - 1) +1: (min_member - 1)*i);
        end
    
        shuffle = shuffle((min_member - 1)*P + 1: end);
        N = length(shuffle);
    end
    
    pacesetter = sort(randperm(N, P - 1));
    for i = 1: P
        if i == 1
            Xmask{i, 1} = [Xmask{i, 1}, shuffle(1: pacesetter(i))];
        elseif i == P
            Xmask{i, 1} = [Xmask{i, 1}, shuffle(pacesetter(i - 1) + 1: N)];
        else
            Xmask{i, 1} = [Xmask{i, 1}, shuffle(pacesetter(i - 1) + 1: pacesetter(i))];
        end
    end
elseif ischar(min_member) && strcmp(min_member, 'equal')
    shuffle = randperm(N);
    Xmask = cell(P, 1);
    pacesetter = round(linspace(1, N, P + 1));

    for i = 1: P
        if i < P
            Xmask{i, 1} = shuffle(pacesetter(i): pacesetter(i + 1) - 1);
        else
            Xmask{i, 1} = shuffle(pacesetter(i): pacesetter(i + 1));
        end
    end
else
    fprintf('Warning: Invalid parameter for splitting data.');
    return
end

end