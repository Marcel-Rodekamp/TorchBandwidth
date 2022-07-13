#include<iostream>
#include<chrono>
#include<vector>
#include<algorithm>
#include<torch/torch.h>

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
    torch::init_num_threads();
    const int num_threads = omp_get_max_threads();
    torch::set_num_threads(num_threads);

    // Tensor size
    const std::size_t N = std::pow(2,14);

    // Tensor memory  
    // 1 byte = 9.31×10-10 Gb
    const double mem_GB  = N * sizeof(double) * 9.31e-10;
    
    // Statistical power
    const int N_meas = 100;

    // Number of sweeps per measurement
    const int N_sweep = 100;
    
    // store timing data in here
    std::vector<double> timings(N_meas);

    // Tensors have this specification
    // kFloat32 float
    // kFloat64 double
    auto options = torch::TensorOptions()
                        .dtype(torch::kFloat64); // double

    // Define the tensors
    torch::Tensor M = torch::zeros({N,N},options);
    torch::Tensor x = torch::zeros(N,options);
    torch::Tensor out = torch::zeros(N,options);

    double start,end;

    // measure T1+T2 and store into Tout N_meas times
    for(int i = 0; i < N_meas; ++i){
        if (i % 10 == 0){
            std::cout << "Measure ID: " << i << "/" << N_meas << std::endl;
        }
        start = omp_get_wtime();
        for(int j = 0; j < N_sweep; ++j){
            out = torch::matmul(M,x);
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
    double est_bw = (2*mem_GB+mem_GB*mem_GB) / est_timings;
    // just simple gaussian error propagation
    double err_bw = err_timings * (2*mem_GB+mem_GB*mem_GB) / (est_timings*est_timings); 

    // N operations in vector add
    double est_flops = N*N  / est_timings;
    // just simple gaussian error propagation
    double err_flops = err_timings * N * N / (est_timings*est_timings);
 
    std::cout << "* Memory Footprint   : " << mem_GB * mem_GB << " Gb" << std::endl;
    std::cout << "* Min Execution Time : " << *std::min_element(timings.begin(), timings.end()) << " s" << std::endl;
    std::cout << "* Max Execution Time : " << *std::max_element(timings.begin(), timings.end()) << " s" << std::endl;
    std::cout << "* Mean Execution Time: " << est_timings << " +/- " << err_timings << " s" << std::endl;
    std::cout << "* Mean Bandwidth     : " << est_bw << " +/- " << err_bw << " Gb/s" << std::endl;
    std::cout << "* Mean Flops         : " << est_flops << " +/- " << err_flops << " flops" << std::endl;

    std::cout << std::endl;
    std::cout << "* Number threads     : " << torch::get_num_threads() << std::endl;
    std::cout << "* use GPU            : " << std::boolalpha << false << std::endl;

    return EXIT_SUCCESS;
}
