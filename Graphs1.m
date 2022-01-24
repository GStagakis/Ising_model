clear;
clc;
clf;

v2 = importdata("./cuda_ising21.out");

x = 4:11;

figure(1);
plot(x, v2, "mo-");

ylabel("Execution times");
xlabel("log_2(gridDim)");
legend("CUDA V2", 'Location', 'best');
title("N = 2^1^5, BlockDim = 16(max number of threads 1024), K = 32");
saveas(gcf, "Graph1.jpg");
