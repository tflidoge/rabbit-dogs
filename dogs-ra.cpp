#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

int n = 1; // number of rabbits
int m = 1; // number of dogs
double dt = 0.001; // time step
int N = (int)(100/dt);
MatrixXd pos_r(n, 2);
MatrixXd pos_d(m, 2);
VectorXd caught(n);
double sigma = 1.0; // standard deviation of Brownian motion
double v = 1.0; // speed of dogs

void update_pos_r(double hatD) {
    for (int j=0; j<n; j++) {
        if (caught(j)) {
            continue;
        }
        Vector2d dW = sqrt(2*dt*hatD) * Vector2d::Random();
        pos_r.row(j) += dW;
    }
}

void update_pos_d() {
    for (int j=0; j<m; j++) {
        if (caught.all()) {
            break;
        }
        if (caught(j%n)) {
            continue;
        }
        Vector2d min_dist = Vector2d::Zero();
        double min_dist_val = numeric_limits<double>::max();
        for (int i=0; i<n; i++) {
            if (caught(i)) {
                continue;
            }
            Vector2d dist = pos_r.row(i) - pos_d.row(j);
            double dist_val = dist.norm();
            if (dist_val < min_dist_val) {
                min_dist = dist;
                min_dist_val = dist_val;
            }
        }
        Vector2d direction = min_dist.normalized();
        pos_d.row(j) += v * direction * dt;
    }
}

bool check_caught() {
    for (int j=0; j<n; j++) {
        if (!caught(j) && (pos_r.row(j) - pos_d.row(j%n)).norm() < 0.1) {
            caught(j) = true;
            return true;
        }
    }
    return false;
}

double capture(int Ns, double hatD) {
    double T = 0.0;
    for (int rnd=0; rnd<Ns; rnd++) {
        pos_r.setZero();
        pos_d.setZero();
        for (int j=0; j<m; j++) {
            pos_d(j, 0) = cos(j*2*M_PI/m);
            pos_d(j, 1) = sin(j*2*M_PI/m);
        }
        caught.setZero();
        for (int k=0; k<N; k++) {
            update_pos_r(hatD);
            update_pos_d();
            if (check_caught()) {
                break;
            }
            if (k == N-1) {
                T += dt*N;
            }
        }
    }
    return T/Ns;
}

int main() {
    int Ns = 20;
    VectorXd hatD_vals = VectorXd::LinSpaced(50, 0.0, 5.0);
    vector<double> avg_times;
    for (int i=0; i<hatD_vals.size(); i++) {
        double hatD = hatD_vals(i);
        vector<future<double>> futures;
        for (int j=0; j<Ns; j++) {
            futures.push_back(async(launch::async, capture, 1, hatD));
        }
        vector<double> times;
        for (auto& fut : futures) {
            times.push_back(fut.get());
        }
        double avg_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
        avg_times.push_back(avg_time);
        cout << "hatD: " << hatD << ", average time: " << avg_time << "s" << endl;
    }
    return 0;
}