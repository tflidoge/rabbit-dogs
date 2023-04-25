#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MOVE_RANGE 3
#define MAX_SPEED 5
#define MAX_DISTANCE 100

double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int main() {
    double dog_x = 0, dog_y = 0;
    double rabbit_x = rand() % MAX_DISTANCE - MAX_DISTANCE / 2.0;
    double rabbit_y = rand() % MAX_DISTANCE - MAX_DISTANCE / 2.0;
    srand(time(NULL));
    while (1) {
        double dx = rabbit_x - dog_x;
        double dy = rabbit_y - dog_y;
        double dist = distance(dog_x, dog_y, rabbit_x, rabbit_y);
        if (dist < 1.0) {
            printf("Dog caught the rabbit!\n");
            break;
        }
        double speed = fmin(MAX_SPEED, dist);
        dx = dx / dist * speed + (rand() % MOVE_RANGE - MOVE_RANGE / 2.0);
        dy = dy / dist * speed + (rand() % MOVE_RANGE - MOVE_RANGE / 2.0);
        rabbit_x += dx;
        rabbit_y += dy;
        dog_x += dx / dist * fmin(MAX_SPEED, dist);
        dog_y += dy / dist * fmin(MAX_SPEED, dist);
        printf("Rabbit moved to (%lf, %lf)\n", rabbit_x, rabbit_y);
        printf("Dog moved to (%lf, %lf)\n", dog_x, dog_y);
    }
    return 0;
}