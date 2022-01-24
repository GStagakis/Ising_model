clear;
clc;
clf;

v1 = importdata("./cuda_ising12.out");
v2 = importdata("./cuda_ising22.out");
v3 = importdata("./cuda_ising32.out");

x = 5:14;

figure(1);
plot(x, v1, "gs--");
hold on;
plot(x, v2, "bo-.");
hold on;
plot(x, v3, "kh:");
ylabel("Execution times");
xlabel("log_2(N)");
legend("CUDA 1 moment per thread", "CUDA 4 moments per thread", "CUDA 4 moments per thread + Shared Memory", 'Location', 'best');
title("BlockDim = 16(max number of threads 1024), K = 32");
saveas(gcf, "Graph21.jpg");
