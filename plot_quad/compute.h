#ifndef C_PROCESS_H
#define C_PROCESS_H

#include <stdio.h>
#include <math.h>

void interceptQuad(float *E, float P1x, float P1y, float P1z, float P2x, float P2y, float P2z, float *pp1x, float *pp1y, float *pp1z);
void compute3DValues(float *E, char *quadValuesInTheWholeCube, float lim, int N);

#endif


