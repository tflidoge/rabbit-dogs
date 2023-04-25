#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// Define parameters
const int n = 1;  // number of rabbits
const int m = 1; // number of dogs
const double dt = 0.001;  // time step
const int N = 100000;
const double sigma = 2;  // standard deviation of the Gaussian distribution for rabbits
const double vr = 2;
const double v = 1;  // speed of the dogs

int main() {
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    unsigned long int seed = 1;  // Change this for different random number sequences
    gsl_rng_set(r, seed);

    // Initialize position arrays for the rabbits and the dogs
    double pos_r[n][2] = {0};
    double pos_d[m][2] = {0};
    int caught[n] = {0};  // boolean array indicating whether a rabbit is caught

    for (int k = 0; k < N; k++) {
        // Update rabbit positions with Brownian motion
        for (int j = 0; j < n; j++) {
            if (!caught[j]) {
                double step[2] = {gsl_ran_gaussian_ziggurat(r, sqrt(dt) * sigma), gsl_ran_gaussian_ziggurat(r, sqrt(dt) * sigma)};
                pos_r[j][0] += step[0];
                pos_r[j][1] += step[1];
            }
        }

        // Update dog positions and find nearest rabbit
        for (int j = 0; j < m; j++) {
            double min_dist = INFINITY;
            int nearest = -1;
            for (int i = 0; i < n; i++) {
                if (!caught[i]) {
                    double dist = hypot(pos_r[i][0] - pos_d[j][0], pos_r[i][1] - pos_d[j][1]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = i;
                    }
                }
            }

            // Move dog towards nearest rabbit
            if (nearest != -1) {
                double direction[2] = {pos_r[nearest][0] - pos_d[j][0], pos_r[nearest][1] - pos_d[j][1]};
                double norm = hypot(direction[0], direction[1]);
                direction[0] /= norm;
                direction[1] /= norm;
                pos_d[j][0] += v * direction[0] * dt;
                pos_d[j][1] += v * direction[1] * dt;
            }
        }

        // Check if any rabbits are caught
        for (int j = 0; j < n; j++) {
            if (!caught[j]) {
                for (int i = 0; i < m; i++) {
                    if (hypot(pos_r[j][0] - pos_d[i][0], pos_r[j][1] - pos_d[i][1]) < 0.1) {
                        caught[j] = 1;
                        break;
                    }
                }
            }
        }
    }

    gsl_rng_free(r);
    return 0;
}