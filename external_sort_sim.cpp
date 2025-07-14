// external_sort_sim.cpp
// A modular external sorting simulator prototype for cloud-like settings
// Simulates I/O, network variability, data skew, chunked access patterns, and compute for various external sorting algorithms

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <numeric>

using namespace std;

// Random engine for variability
typedef std::mt19937_64 RNG;
static RNG rng(42);

// Simulated object store with latency, throughput, variability, and cost characteristics
struct ObjectStore {
    double latency_ms;         // base latency per operation
    double mean_throughput_MBps;    // nominal throughput per stream
    double throughput_jitter;  // fractional jitter (e.g., 0.2 means ±20%)
    double cost_per_GB;        // cost per GB transferred
    double cost_per_request;   // fixed cost per API call
    double chunk_size_MB;      // chunk size for I/O granularity

    // Sample a throughput for this operation
    double sample_throughput() {
        normal_distribution<double> d(mean_throughput_MBps, mean_throughput_MBps * throughput_jitter);
        return max(1.0, d(rng));
    }

    // Simulate read: compute time and cost, but do not sleep
    pair<double,double> read(double size_MB) {
        int num_chunks = ceil(size_MB / chunk_size_MB);
        double total_time = 0.0, total_cost = 0.0;
        double remaining = size_MB;
        for (int i = 0; i < num_chunks; ++i) {
            double this_chunk = min(chunk_size_MB, remaining);
            remaining -= this_chunk;
            double thr = sample_throughput();
            double t = latency_ms/1000.0 + this_chunk/thr;
            double c = this_chunk * cost_per_GB / 1024.0 + cost_per_request;
            total_time += t;
            total_cost += c;
        }
        return {total_time, total_cost};
    }

    // Simulate write: compute time and cost, but do not sleep
    pair<double,double> write(double size_MB) {
        int num_chunks = ceil(size_MB / chunk_size_MB);
        double total_time = 0.0, total_cost = 0.0;
        double remaining = size_MB;
        for (int i = 0; i < num_chunks; ++i) {
            double this_chunk = min(chunk_size_MB, remaining);
            remaining -= this_chunk;
            double thr = sample_throughput();
            double t = latency_ms/1000.0 + this_chunk/thr;
            double c = this_chunk * cost_per_GB / 1024.0 + cost_per_request;
            total_time += t;
            total_cost += c;
        }
        return {total_time, total_cost};
    }
};

// Simulated compute node or function with slowdown probability
struct ComputeNode {
    double compute_speed_MBps; // how fast it can sort
    double cost_per_hour;      // compute cost per hour
    double straggler_prob;     // probability a task is slowed
    double straggler_factor;   // slowdown multiplier if straggler

    // Simulate sort: compute time and cost, no sleep
    pair<double,double> sort(double size_MB) {
        bool is_straggler = (uniform_real_distribution<double>(0,1)(rng) < straggler_prob);
        double speed = compute_speed_MBps * (is_straggler ? (1.0/straggler_factor) : 1.0);
        double time_sec = size_MB / speed;
        double cost = time_sec * (cost_per_hour / 3600.0);
        return {time_sec, cost};
    }
};

// Generate run sizes based on data skew distribution
vector<double> generate_run_sizes(double dataset_MB, double avg_run_MB, double skew_alpha) {
    int num_runs = ceil(dataset_MB / avg_run_MB);
    vector<double> weights(num_runs);
    for (int i = 1; i <= num_runs; ++i) weights[i-1] = 1.0 / pow(i, skew_alpha);
    double sum_w = accumulate(weights.begin(), weights.end(), 0.0);
    for (auto &w : weights) w /= sum_w;
    vector<double> sizes(num_runs);
    for (int i = 0; i < num_runs; ++i) sizes[i] = weights[i] * dataset_MB;
    return sizes;
}

// Base class for external sort algorithms
class ExternalSortAlgo {
public:
    virtual string name() = 0;
    // Run simulation on dataset_MB; returns time (sec) and cost ($)
    virtual pair<double,double> run(double dataset_MB, ObjectStore& store, ComputeNode& node) = 0;
    virtual ~ExternalSortAlgo() = default;
};

// 1) Two-Phase Merge Sort (non-skewed)
class TwoPhaseNoSkew : public ExternalSortAlgo {
public:
    string name() override { return "Two-Phase Merge Sort (no skew)"; }
    pair<double,double> run(double dataset_MB, ObjectStore& store, ComputeNode& node) override {
        double chunk = 512;
        int runs = ceil(dataset_MB / chunk);
        double t=0,c=0;
        // initial runs
        for(int i=0;i<runs;++i){ auto rd=store.read(chunk); t+=rd.first; c+=rd.second; auto st=node.sort(chunk); t+=st.first; c+=st.second; auto wt=store.write(chunk); t+=wt.first; c+=wt.second; }
        // merge all
        auto rd_all=store.read(dataset_MB); t+=rd_all.first; c+=rd_all.second;
        auto st_all=node.sort(dataset_MB); t+=st_all.first; c+=st_all.second;
        auto wt_all=store.write(dataset_MB); t+=wt_all.first; c+=wt_all.second;
        return {t,c};
    }
};

// 2) Two-Phase Merge Sort (skewed)
class TwoPhaseSkew : public ExternalSortAlgo {
public:
    string name() override { return "Two-Phase Merge Sort (skewed)"; }
    pair<double,double> run(double dataset_MB, ObjectStore& store, ComputeNode& node) override {
        auto runs = generate_run_sizes(dataset_MB, 512, 1.1);
        double t=0,c=0;
        for(auto sz: runs){ auto rd=store.read(sz); t+=rd.first; c+=rd.second; auto st=node.sort(sz); t+=st.first; c+=st.second; auto wt=store.write(sz); t+=wt.first; c+=wt.second; }
        auto rd_all=store.read(dataset_MB); t+=rd_all.first; c+=rd_all.second;
        auto st_all=node.sort(dataset_MB); t+=st_all.first; c+=st_all.second;
        auto wt_all=store.write(dataset_MB); t+=wt_all.first; c+=wt_all.second;
        return {t,c};
    }
};

// 3) K-Way Merge Sort (non-skewed)
class KWayNoSkew : public ExternalSortAlgo {
    int k;
public:
    KWayNoSkew(int k_):k(k_){}
    string name() override { return string("K-Way Merge Sort (no skew, k=")+to_string(k)+")"; }
    pair<double,double> run(double dataset_MB, ObjectStore& store, ComputeNode& node) override {
        double chunk=512; int runs=ceil(dataset_MB/chunk);
        int passes=ceil(log(runs)/log(k)); double t=0,c=0;
        for(int i=0;i<runs;++i){ auto rd=store.read(chunk); t+=rd.first; c+=rd.second; auto st=node.sort(chunk); t+=st.first; c+=st.second; auto wt=store.write(chunk); t+=wt.first; c+=wt.second; }
        for(int p=0;p<passes;++p){ auto rd=store.read(dataset_MB); t+=rd.first; c+=rd.second; auto st=node.sort(dataset_MB); t+=st.first; c+=st.second; auto wt=store.write(dataset_MB); t+=wt.first; c+=wt.second; }
        return {t,c};
    }
};

// 4) K-Way Merge Sort (skewed)
class KWaySkew : public ExternalSortAlgo {
    int k;
public:
    KWaySkew(int k_):k(k_){}
    string name() override { return string("K-Way Merge Sort (skewed, k=")+to_string(k)+")"; }
    pair<double,double> run(double dataset_MB, ObjectStore& store, ComputeNode& node) override {
        auto runs=generate_run_sizes(dataset_MB,512,1.1);
        int passes=ceil(log(runs.size())/log(k)); double t=0,c=0;
        for(auto sz:runs){ auto rd=store.read(sz); t+=rd.first; c+=rd.second; auto st=node.sort(sz); t+=st.first; c+=st.second; auto wt=store.write(sz); t+=wt.first; c+=wt.second; }
        for(int p=0;p<passes;++p){ auto rd=store.read(dataset_MB); t+=rd.first; c+=rd.second; auto st=node.sort(dataset_MB); t+=st.first; c+=st.second; auto wt=store.write(dataset_MB); t+=wt.first; c+=wt.second; }
        return {t,c};
    }
};

int main(){
    double dataset_MB=10*1024; //10GB
    ObjectStore s3{50,100,0.2,0.023,0.000005,64};
    ComputeNode lambda{100,6,0.1,4};

    vector<ExternalSortAlgo*> algos{
        new TwoPhaseNoSkew(), new TwoPhaseSkew(),
        new KWayNoSkew(4), new KWaySkew(4)
    };

    for(auto* a: algos){
        cout<<"Algorithm: "<<a->name()<<"\n";
        auto r=a->run(dataset_MB,s3,lambda);
        cout<<"  Total time: "<<r.first<<" seconds\n";
        cout<<"  Total cost: $"<<r.second<<"\n";
        cout<<"-----------------------------\n";
    }
    for(auto* a: algos) delete a;
    return 0;
}
