clear;
clc;
clf;

v1 = importdata("./cuda_ising13.out");
v2 = importdata("./cuda_ising23.out");
v3 = importdata("./cuda_ising33.out");

x = 2:7;

figure(1);
plot(x, v1, "gs--");
hold on;
plot(x, v2, "bo-.");
hold on;
plot(x, v3, "kh:");
ylabel("Execution times");
xlabel("log_2(K)");
legend("CUDA 1 moment per thread", "CUDA 4 moments per thread", "CUDA 4 moments per thread + Shared Memory", 'Location', 'best');
title("N = 2^1^2, BlockDim = 16(max number of threads 1024)");
saveas(gcf, "Graph3.jpg");
