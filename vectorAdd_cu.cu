#include<iostream>
#include<chrono>
#include<vector>
#include<algorithm>
#include<cmath>

#include<omp.h>

__global__ void vec_add(const double * x, const double * y, double * out, std::size_t N){
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        out[i] = x[i] + y[i];
    }
}

__global__ void initVector(double * x, std::size_t N){
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        x[i] = static_cast<double>(i);
    }
}

double mean(std::vector<double> v) {
    double res = 0;

    for(auto e: v){
        res+=e;
    }
    res/=v.size();

    return res;
}

double err(std::vector<double> v) {

    double est = mean(v);

    double res = 0;

    for(auto e: v){
        res+= (e - est)*(e - est) ;
    }
    res/=v.size()-1;

    return std::sqrt(res);
}

int main(){
    // Tensor size
    const std::size_t N = std::pow(2,28);

    // Tensor memory  
    // 1 byte = 9.31×10-10 Gb
    const double mem_GB = N * sizeof(double) * 9.31e-10;
    
    // Statistical power
    const int N_meas = 100;

    // Number of sweeps per measurement
    const int N_sweep = 10;

    // store timing data in here
    std::vector<double> timings(N_meas);

    // Define the tensors
    double * x  = nullptr;
    double * y  = nullptr;
    double * out= nullptr;

    cudaMalloc(&x,N * sizeof(double));
    cudaMalloc(&y,N * sizeof(double));
    cudaMalloc(&out,N * sizeof(double));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1)/threadsPerBlock; 

    initVector<<<numBlocks,threadsPerBlock>>>(x,N);
    initVector<<<numBlocks,threadsPerBlock>>>(y,N);
    initVector<<<numBlocks,threadsPerBlock>>>(out,N);

    double start, end;

    // measure T1+T2 and store into Tout N_meas times
    for(int i = 0; i < N_meas; ++i){
        if (i % 10 == 0){
            std::cout << "Measure ID: " << i << "/" << N_meas << std::endl;
        }
        start = omp_get_wtime();
        for(int j = 0; j < N_sweep; ++j){
            vec_add<<<numBlocks,threadsPerBlock>>>(x,y,out,N);
            cudaDeviceSynchronize();
        }
        end = omp_get_wtime();
        
        timings[i] = (end-start)/N_sweep;
    }

    // compute and print statistics    
    double est_timings = mean(timings);
    double err_timings = err(timings);

    // the factor 3 comes from:
    // * factor 2: 2 vector load
    // * factor 1.5: 1 vector store
    // => 1.5 * 2 = 3
    double est_bw = 3*mem_GB / est_timings;
    // just simple gaussian error propagation
    double err_bw = err_timings * 3*mem_GB / (est_timings*est_timings); 

    // N operations in vector add
    double est_flops = N  / est_timings;
    // just simple gaussian error propagation
    double err_flops = err_timings * N / (est_timings*est_timings);
 
    std::cout << "* Memory Footprint   : " << mem_GB << " Gb" << std::endl;
    std::cout << "* Min Execution Time : " << *std::min_element(timings.begin(), timings.end()) << " s" << std::endl;
    std::cout << "* Max Execution Time : " << *std::max_element(timings.begin(), timings.end()) << " s" << std::endl;
    std::cout << "* Mean Execution Time: " << est_timings << " +/- " << err_timings << " s" << std::endl;
    std::cout << "* Mean Bandwidth     : " << est_bw << " +/- " << err_bw << " Gb/s" << std::endl;
    std::cout << "* Mean Flops         : " << est_flops << " +/- " << err_flops << " flops" << std::endl;

    std::cout << std::endl;
#pragma omp parallel
    {
#pragma omp single
    std::cout << "* Number threads     : " << omp_get_num_threads() << "/" << omp_get_max_threads() << std::endl;
    }
    std::cout << "* use GPU            : " << std::boolalpha << true << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    return EXIT_SUCCESS;
}
