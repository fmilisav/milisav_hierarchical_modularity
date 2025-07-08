% Load binary directed networks
load('pruned_nets.mat'); % level1, level2, level3, levelMS

levels = {'level1', 'level2', 'level3', 'MS'};
nets = {level1, level2, level3, levelMS};

n_levels = length(levels);
n_motifs = 13;
n_nets_per_level = 100;
n_total_nets = n_levels * n_nets_per_level;

% Preallocate arrays
F_struct = zeros(n_total_nets, n_motifs);   % [400 x 13]
F_func   = zeros(n_total_nets, n_motifs);   % [400 x 13]
level_names = cell(n_total_nets, 1);        % [400 x 1] cell array of strings

net_idx = 1;

for l = 1:n_levels
    level_nets = nets{l};       % [100 x 400 x 400]
    level_name = levels{l};     % e.g., 'level1'

    for i = 1:n_nets_per_level
        A = squeeze(level_nets(i, :, :)); % [400 x 400]

        [f_s, ~] = motif3struct_bin(A);   % [1 x 13]
        [f_f, ~] = motif3funct_bin(A);    % [1 x 13]

        F_struct(net_idx, :) = f_s;
        F_func(net_idx, :)   = f_f;
        level_names{net_idx} = level_name;

        net_idx = net_idx + 1;
    end
end

% Save all outputs
save('pruned_nets_motifs.mat', 'F_struct', 'F_func', 'level_names');