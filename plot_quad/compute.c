//  gcc -fPIC --shared compute.c -o compute.so

#include "compute.h"



float *zeros(int N) {

    float *array;

    array = malloc(sizeof(float)*N);

    array[0] = 1.0;
    array[1] = -1.0;

    return array;
}



float evalQuad(float *E, float x, float y, float z) {

    float a = E[0*4+0];
    float b = E[1*4+1];
    float c = E[2*4+2];
    float d = E[3*4+3];
    float f = E[1*4+2];
    float g = E[0*4+2];
    float h = E[0*4+1];
    float p = E[0*4+3];
    float q = E[1*4+3];
    float r = E[2*4+3];

    return a*x*x + b*y*y + c*z*z + 2.0*f*y*z + 2.0*g*x*z + 2.0*h*x*y + 2.0*p*x + 2.0*q*y + 2.0*r*z + d;
}


void compute3DValues(float *E, char *quadValuesInTheWholeCube, float lim, int N) {

    int i, j, k;
    float x, y, z;

    float dt = 2.0*lim/(float)(N-1);

    for (i=0;i<N;i++)
        for (j=0;j<N;j++)
            for (k=0;k<N;k++) {

                x = -lim+(float)i*dt - dt/2.0;
                y = -lim+(float)j*dt - dt/2.0;
                z = -lim+(float)k*dt - dt/2.0;

                quadValuesInTheWholeCube[i*N*N + j*N + k] = evalQuad(E,x,y,z)>0?1:-1;
            }
}





void interceptQuad(float *E, float P1x, float P1y, float P1z, float P2x, float P2y, float P2z, float *pp1x, float *pp1y, float *pp1z) {

    if (E == NULL) {
        *pp1x = (P1x + P2x)/2.0;
        *pp1y = (P1y + P2y)/2.0;
        *pp1z = (P1z + P2z)/2.0;

        return;
    }
    
    float a = E[4*0 + 0];
    float b = E[4*1 + 1];
    float c = E[4*2 + 2];
    float d = E[4*3 + 3];
    float f = E[4*1 + 2];
    float g = E[4*0 + 2];
    float h = E[4*0 + 1];
    float p = E[4*0 + 3];
    float q = E[4*1 + 3];
    float r = E[4*2 + 3];


    float P1x2_ = P1x*P1x;
    float P1y2_ = P1y*P1y;
    float P1z2_ = P1z*P1z;
    float P2x2_ = P2x*P2x;
    float P2y2_ = P2y*P2y;
    float P2z2_ = P2z*P2z;

    float b_ =  -(P2x*p - P1x*p - P1y*q + P2y*q - P1z*r + P2z*r - P1x2_*a - P1y2_*b - P1z2_*c + P1x*P2x*a + P1y*P2y*b + P1z*P2z*c - 2.0*P1y*P1z*f + P1y*P2z*f + P2y*P1z*f - 2.0*P1x*P1z*g + P1x*P2z*g + P2x*P1z*g - 2.0*P1x*P1y*h + P1x*P2y*h + P2x*P1y*h);
    float D_ = (h*h - a*b)*P1x2_*P2y2_ + (2.0*g*h - 2.0*a*f)*P1x2_*P2y*P2z + (2.0*h*p - 2.0*a*q)*P1x2_*P2y + (g*g - a*c)*P1x2_*P2z2_ + (2.0*g*p - 2.0*a*r)*P1x2_*P2z + (p*p - a*d)*P1x2_ + (- 2.0*h*h + 2.0*a*b)*P1x*P1y*P2x*P2y + (2.0*a*f - 2.0*g*h)*P1x*P1y*P2x*P2z + (2.0*a*q - 2.0*h*p)*P1x*P1y*P2x + (2.0*b*g - 2.0*f*h)*P1x*P1y*P2y*P2z + (2.0*b*p - 2.0*h*q)*P1x*P1y*P2y + (2.0*f*g - 2.0*c*h)*P1x*P1y*P2z2_ + (2.0*f*p + 2.0*g*q - 4*h*r)*P1x*P1y*P2z + (2.0*p*q - 2.0*d*h)*P1x*P1y + (2.0*a*f - 2.0*g*h)*P1x*P1z*P2x*P2y + (- 2.0*g*g + 2.0*a*c)*P1x*P1z*P2x*P2z + (2.0*a*r - 2.0*g*p)*P1x*P1z*P2x + (2.0*f*h - 2.0*b*g)*P1x*P1z*P2y2_ + (2.0*c*h - 2.0*f*g)*P1x*P1z*P2y*P2z + (2.0*f*p - 4*g*q + 2.0*h*r)*P1x*P1z*P2y + (2.0*c*p - 2.0*g*r)*P1x*P1z*P2z + (2.0*p*r - 2.0*d*g)*P1x*P1z + (2.0*a*q - 2.0*h*p)*P1x*P2x*P2y + (2.0*a*r - 2.0*g*p)*P1x*P2x*P2z + (- 2.0*p*p + 2.0*a*d)*P1x*P2x + (2.0*h*q - 2.0*b*p)*P1x*P2y2_ + (2.0*g*q - 4*f*p + 2.0*h*r)*P1x*P2y*P2z + (2.0*d*h - 2.0*p*q)*P1x*P2y + (2.0*g*r - 2.0*c*p)*P1x*P2z2_ + (2.0*d*g - 2.0*p*r)*P1x*P2z + (h*h - a*b)*P1y2_*P2x2_ + (2.0*f*h - 2.0*b*g)*P1y2_*P2x*P2z + (2.0*h*q - 2.0*b*p)*P1y2_*P2x + (f*f - b*c)*P1y2_*P2z2_ + (2.0*f*q - 2.0*b*r)*P1y2_*P2z + (q*q - b*d)*P1y2_ + (2.0*g*h - 2.0*a*f)*P1y*P1z*P2x2_ + (2.0*b*g - 2.0*f*h)*P1y*P1z*P2x*P2y + (2.0*c*h - 2.0*f*g)*P1y*P1z*P2x*P2z + (2.0*g*q - 4*f*p + 2.0*h*r)*P1y*P1z*P2x + (- 2.0*f*f + 2.0*b*c)*P1y*P1z*P2y*P2z + (2.0*b*r - 2.0*f*q)*P1y*P1z*P2y + (2.0*c*q - 2.0*f*r)*P1y*P1z*P2z + (2.0*q*r - 2.0*d*f)*P1y*P1z + (2.0*h*p - 2.0*a*q)*P1y*P2x2_ + (2.0*b*p - 2.0*h*q)*P1y*P2x*P2y + (2.0*f*p - 4*g*q + 2.0*h*r)*P1y*P2x*P2z + (2.0*d*h - 2.0*p*q)*P1y*P2x + (2.0*b*r - 2.0*f*q)*P1y*P2y*P2z + (- 2.0*q*q + 2.0*b*d)*P1y*P2y + (2.0*f*r - 2.0*c*q)*P1y*P2z2_ + (2.0*d*f - 2.0*q*r)*P1y*P2z + (g*g - a*c)*P1z2_*P2x2_ + (2.0*f*g - 2.0*c*h)*P1z2_*P2x*P2y + (2.0*g*r - 2.0*c*p)*P1z2_*P2x + (f*f - b*c)*P1z2_*P2y2_ + (2.0*f*r - 2.0*c*q)*P1z2_*P2y + (r*r - c*d)*P1z2_ + (2.0*g*p - 2.0*a*r)*P1z*P2x2_ + (2.0*f*p + 2.0*g*q - 4*h*r)*P1z*P2x*P2y + (2.0*c*p - 2.0*g*r)*P1z*P2x*P2z + (2.0*d*g - 2.0*p*r)*P1z*P2x + (2.0*f*q - 2.0*b*r)*P1z*P2y2_ + (2.0*c*q - 2.0*f*r)*P1z*P2y*P2z + (2.0*d*f - 2.0*q*r)*P1z*P2y + (- 2.0*r*r + 2.0*c*d)*P1z*P2z + (p*p - a*d)*P2x2_ + (2.0*p*q - 2.0*d*h)*P2x*P2y + (2.0*p*r - 2.0*d*g)*P2x*P2z + (q*q - b*d)*P2y2_ + (2.0*q*r - 2.0*d*f)*P2y*P2z + (r*r - c*d)*P2z2_;
    float a_ = (f*(2.0*P1y*P1z - 2.0*P1y*P2z - 2.0*P2y*P1z + 2.0*P2y*P2z) + g*(2.0*P1x*P1z - 2.0*P1x*P2z - 2.0*P2x*P1z + 2.0*P2x*P2z) + h*(2.0*P1x*P1y - 2.0*P1x*P2y - 2.0*P2x*P1y + 2.0*P2x*P2y) + P1x2_*a + P2x2_*a + P1y2_*b + P2y2_*b + P1z2_*c + P2z2_*c - 2.0*P1x*P2x*a - 2.0*P1y*P2y*b - 2.0*P1z*P2z*c);

 
    float t1 = (b_ + sqrt(D_)+0.000001)/(a_+0.000001);
    float t2 = (b_ - sqrt(D_)+0.000001)/(a_+0.000001);

    float t;

    if (t1<1.0 && t1>0.0)
        t = t1;
    else
        t = t2;

    t = t>=0.0&&t<=1.0?t:0.5;


    *pp1x = P1x + t*(P2x-P1x);
    *pp1y = P1y + t*(P2y-P1y);
    *pp1z = P1z + t*(P2z-P1z); 

    //printf("%.4f, %.4f, %.4f\n",*pp1x, *pp1y, *pp1z);

    return;
}








