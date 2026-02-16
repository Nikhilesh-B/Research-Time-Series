/**
 * Hamilton's Markov-Switching Model of Business Fluctuations
 * C++ equivalent of the Python data generating process in data_generating_process.ipynb
 *
 * Output: CSV with columns ,t,dy,S,mu (same format as gnp_data_created.csv).
 * Numeric values differ from Python due to different RNGs; structure and DGP are identical.
 *
 * Generates AR(4) process with state-dependent mean (Eq 4.79-4.82):
 *   (dy_t - mu_{S_t}) = phi_1*(dy_{t-1} - mu_{S_{t-1}}) + ... + phi_4*(dy_{t-4} - mu_{S_{t-4}}) + e_t
 *   e_t ~ N(0, sigma^2)
 *
 * Compile: g++ -O3 -o hamilton_gnp_dgp hamilton_gnp_dgp.cpp
 * Run: ./hamilton_gnp_dgp [output_csv] [seed] [n_samples]
 *      Use output_csv "none" to skip file output (benchmark mode).
 *      n_samples defaults to 10001; use 100000000 for 10^8.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <string>

// Hamilton (1989) parameters - 1952:II to 1984:IV sample
constexpr double P = 0.9008; // Pr[S_t=1 | S_{t-1}=1]
constexpr double Q = 0.7606; // Pr[S_t=0 | S_{t-1}=0]
constexpr double PHI1 = 0.0898;
constexpr double PHI2 = -0.0186;
constexpr double PHI3 = -0.1743;
constexpr double PHI4 = -0.0839;
constexpr double SIGMA = 0.7962;
constexpr double MU0 = -0.2132; // recession mean
constexpr double MU1 = 1.1283;  // expansion mean

constexpr int WINDOW_SIZE = 4; // AR(4)

int main(int argc, char *argv[])
{
    const char *output_path = (argc > 1) ? argv[1] : "./gnp_data_created.csv";
    unsigned int seed = (argc > 2) ? static_cast<unsigned int>(std::stoul(argv[2])) : 42u;
    long long n_samples = (argc > 3) ? std::stoll(argv[3]) : 1000001LL;
    bool write_output = (std::string(output_path) != "none");

    if (n_samples < WINDOW_SIZE + 1)
    {
        std::cerr << "Error: n_samples must be >= " << (WINDOW_SIZE + 1) << "\n";
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, SIGMA);

    // Preallocate (matches Python DataFrame columns: t, dy, S, mu)
    std::vector<int> t(n_samples);
    std::vector<double> dy(n_samples);
    std::vector<int> S(n_samples);
    std::vector<double> mu(n_samples);

    // Initial values (rows 0..3)
    dy[0] = 1.2;
    dy[1] = 1.1;
    dy[2] = 1.1;
    dy[3] = 1.1;
    for (int i = 0; i < WINDOW_SIZE; ++i)
    {
        t[i] = i;
        S[i] = 1;
        mu[i] = MU1;
    }

    // Generate t=4..n_samples-1
    for (long long i = WINDOW_SIZE; i < n_samples; ++i)
    {
        t[i] = i;

        int s_prev = S[i - 1];
        double dy_l1 = dy[i - 1], mu_l1 = mu[i - 1];
        double dy_l2 = dy[i - 2], mu_l2 = mu[i - 2];
        double dy_l3 = dy[i - 3], mu_l3 = mu[i - 3];
        double dy_l4 = dy[i - 4], mu_l4 = mu[i - 4];

        // Markov transition (Eq 4.82)
        double u = unif(rng);
        int s;
        if (s_prev == 1)
        {
            s = (u <= P) ? 1 : 0;
        }
        else
        {
            s = (u <= Q) ? 0 : 1;
        }

        double mu_s = (s == 1) ? MU1 : MU0;
        mu[i] = mu_s;

        // Eq 4.79: dy_t = mu_S_t + sum(phi_i * (dy_{t-i} - mu_{S_{t-i}})) + e_t
        double epsilon = norm(rng);
        dy[i] = mu_s + PHI1 * (dy_l1 - mu_l1) + PHI2 * (dy_l2 - mu_l2) + PHI3 * (dy_l3 - mu_l3) + PHI4 * (dy_l4 - mu_l4) + epsilon;

        S[i] = s;
    }

    auto t_compute_done = std::chrono::high_resolution_clock::now();

    // Write CSV (matches pandas to_csv format: index,t,dy,S,mu)
    if (write_output)
    {
        std::ofstream out(output_path);
        out << ",t,dy,S,mu\n";
        out << std::fixed << std::setprecision(16);
        for (long long i = 0; i < n_samples; ++i)
        {
            out << i << "," << t[i] << "," << dy[i] << "," << S[i] << "," << mu[i] << "\n";
        }
        out.close();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto compute_ms = std::chrono::duration<double, std::milli>(t_compute_done - t_start).count();
    auto total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "Generated " << n_samples << " samples in " << total_ms / 1000.0 << " s total";
    if (write_output)
        std::cout << " (compute: " << compute_ms / 1000.0 << " s, I/O: " << (total_ms - compute_ms) / 1000.0 << " s)";
    std::cout << "\n  -> " << (n_samples / (compute_ms / 1000.0)) / 1e6 << " M samples/sec (compute only)\n";
    if (write_output)
        std::cout << "  -> " << output_path << "\n";
    return 0;
}
