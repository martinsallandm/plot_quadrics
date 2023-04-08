import plotly.graph_objects as go
import numpy as np
import time

Vs = {
    1: (0.0,0.0,0.0),
    2: (1.0,0.0,0.0),
    5: (0.0,1.0,0.0),
    6: (1.0,1.0,0.0),
    3: (0.0,0.0,1.0),
    4: (1.0,0.0,1.0),
    7: (0.0,1.0,1.0),
    8: (1.0,1.0,1.0)
}

def evalQuad(E,x,y,z):

    a = E[0,0]
    b = E[1,1]
    c = E[2,2]
    d = E[3,3]
    f = E[1,2]
    g = E[0,2]
    h = E[0,1]
    p = E[0,3]
    q = E[1,3]
    r = E[2,3]

    p = a*x**2 + b*y**2 + c*z**2 + 2.0*f*y*z + 2.0*g*x*z + 2.0*h*x*y + 2.0*p*x + 2.0*q*y + 2.0*r*z + d

    return p

def rotateQuadric(E):
    ## rotate some random amount
    R = np.random.randn(4,4)
    R[:,-1:] = 0
    R[-1:,:] = 0
    R[-1,-1] = 1
    R = R.T@R
    D,R = np.linalg.eig(R)

    E2 = R.T@E@R    

    return E2

def interceptQuad(E, P1x, P1y, P1z, P2x, P2y, P2z):

    if E is None:
        pp1x = (P1x + P2x)/2.0
        pp1y = (P1y + P2y)/2.0
        pp1z = (P1z + P2z)/2.0

        return (pp1x, pp1y, pp1z)
    
    a = E[0,0]
    b = E[1,1]
    c = E[2,2]
    d = E[3,3]
    f = E[1,2]
    g = E[0,2]
    h = E[0,1]
    p = E[0,3]
    q = E[1,3]
    r = E[2,3]


    P1x2 = P1x**2 
    P1y2 = P1y**2 
    P1z2 = P1z**2 
    P2x2 = P2x**2 
    P2y2 = P2y**2 
    P2z2 = P2z**2 


    b_ =  -(P2x*p - P1x*p - P1y*q + P2y*q - P1z*r + P2z*r - P1x2*a - P1y2*b - P1z2*c + P1x*P2x*a + P1y*P2y*b + P1z*P2z*c - 2*P1y*P1z*f + P1y*P2z*f + P2y*P1z*f - 2*P1x*P1z*g + P1x*P2z*g + P2x*P1z*g - 2*P1x*P1y*h + P1x*P2y*h + P2x*P1y*h)
    D_ = (h**2 - a*b)*P1x2*P2y2 + (2*g*h - 2*a*f)*P1x2*P2y*P2z + (2*h*p - 2*a*q)*P1x2*P2y + (g**2 - a*c)*P1x2*P2z2 + (2*g*p - 2*a*r)*P1x2*P2z + (p**2 - a*d)*P1x2 + (- 2*h**2 + 2*a*b)*P1x*P1y*P2x*P2y + (2*a*f - 2*g*h)*P1x*P1y*P2x*P2z + (2*a*q - 2*h*p)*P1x*P1y*P2x + (2*b*g - 2*f*h)*P1x*P1y*P2y*P2z + (2*b*p - 2*h*q)*P1x*P1y*P2y + (2*f*g - 2*c*h)*P1x*P1y*P2z2 + (2*f*p + 2*g*q - 4*h*r)*P1x*P1y*P2z + (2*p*q - 2*d*h)*P1x*P1y + (2*a*f - 2*g*h)*P1x*P1z*P2x*P2y + (- 2*g**2 + 2*a*c)*P1x*P1z*P2x*P2z + (2*a*r - 2*g*p)*P1x*P1z*P2x + (2*f*h - 2*b*g)*P1x*P1z*P2y2 + (2*c*h - 2*f*g)*P1x*P1z*P2y*P2z + (2*f*p - 4*g*q + 2*h*r)*P1x*P1z*P2y + (2*c*p - 2*g*r)*P1x*P1z*P2z + (2*p*r - 2*d*g)*P1x*P1z + (2*a*q - 2*h*p)*P1x*P2x*P2y + (2*a*r - 2*g*p)*P1x*P2x*P2z + (- 2*p**2 + 2*a*d)*P1x*P2x + (2*h*q - 2*b*p)*P1x*P2y2 + (2*g*q - 4*f*p + 2*h*r)*P1x*P2y*P2z + (2*d*h - 2*p*q)*P1x*P2y + (2*g*r - 2*c*p)*P1x*P2z2 + (2*d*g - 2*p*r)*P1x*P2z + (h**2 - a*b)*P1y2*P2x2 + (2*f*h - 2*b*g)*P1y2*P2x*P2z + (2*h*q - 2*b*p)*P1y2*P2x + (f**2 - b*c)*P1y2*P2z2 + (2*f*q - 2*b*r)*P1y2*P2z + (q**2 - b*d)*P1y2 + (2*g*h - 2*a*f)*P1y*P1z*P2x2 + (2*b*g - 2*f*h)*P1y*P1z*P2x*P2y + (2*c*h - 2*f*g)*P1y*P1z*P2x*P2z + (2*g*q - 4*f*p + 2*h*r)*P1y*P1z*P2x + (- 2*f**2 + 2*b*c)*P1y*P1z*P2y*P2z + (2*b*r - 2*f*q)*P1y*P1z*P2y + (2*c*q - 2*f*r)*P1y*P1z*P2z + (2*q*r - 2*d*f)*P1y*P1z + (2*h*p - 2*a*q)*P1y*P2x2 + (2*b*p - 2*h*q)*P1y*P2x*P2y + (2*f*p - 4*g*q + 2*h*r)*P1y*P2x*P2z + (2*d*h - 2*p*q)*P1y*P2x + (2*b*r - 2*f*q)*P1y*P2y*P2z + (- 2*q**2 + 2*b*d)*P1y*P2y + (2*f*r - 2*c*q)*P1y*P2z2 + (2*d*f - 2*q*r)*P1y*P2z + (g**2 - a*c)*P1z2*P2x2 + (2*f*g - 2*c*h)*P1z2*P2x*P2y + (2*g*r - 2*c*p)*P1z2*P2x + (f**2 - b*c)*P1z2*P2y2 + (2*f*r - 2*c*q)*P1z2*P2y + (r**2 - c*d)*P1z2 + (2*g*p - 2*a*r)*P1z*P2x2 + (2*f*p + 2*g*q - 4*h*r)*P1z*P2x*P2y + (2*c*p - 2*g*r)*P1z*P2x*P2z + (2*d*g - 2*p*r)*P1z*P2x + (2*f*q - 2*b*r)*P1z*P2y2 + (2*c*q - 2*f*r)*P1z*P2y*P2z + (2*d*f - 2*q*r)*P1z*P2y + (- 2*r**2 + 2*c*d)*P1z*P2z + (p**2 - a*d)*P2x2 + (2*p*q - 2*d*h)*P2x*P2y + (2*p*r - 2*d*g)*P2x*P2z + (q**2 - b*d)*P2y2 + (2*q*r - 2*d*f)*P2y*P2z + (r**2 - c*d)*P2z2
    a_ = (f*(2*P1y*P1z - 2*P1y*P2z - 2*P2y*P1z + 2*P2y*P2z) + g*(2*P1x*P1z - 2*P1x*P2z - 2*P2x*P1z + 2*P2x*P2z) + h*(2*P1x*P1y - 2*P1x*P2y - 2*P2x*P1y + 2*P2x*P2y) + P1x2*a + P2x2*a + P1y2*b + P2y2*b + P1z2*c + P2z2*c - 2*P1x*P2x*a - 2*P1y*P2y*b - 2*P1z*P2z*c)

 
    t1 = (b_ + np.sqrt(D_)+0.000001)/(a_+0.000001)
    t2 = (b_ - np.sqrt(D_)+0.000001)/(a_+0.000001)

    #print(t1,t2)

    if t1<1.0 and t1>0.0:
        t = t1
    else:
        t = t2

    t = t if t<=1.0 else 1.0
    t = t if t>=0.0 else 0.0



    #t = 0.5

    pp1x = P1x + t*(P2x-P1x)
    pp1y = P1y + t*(P2y-P1y)
    pp1z = P1z + t*(P2z-P1z)        

    return (pp1x, pp1y, pp1z)


def addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,    vertex1, vertex2,   vertex3, vertex4,    vertex5, vertex6,   vertexIndex, ox, oy, oz,  dt, E):

    x,y,z = interceptQuad(E, 
                          ox+dt*Vs[vertex1][0], oy+dt*Vs[vertex1][1], oz+dt*Vs[vertex1][2],
                          ox+dt*Vs[vertex2][0], oy+dt*Vs[vertex2][1], oz+dt*Vs[vertex2][2])

    X.append(x)
    Y.append(y)
    Z.append(z)

    x,y,z = interceptQuad(E, 
                          ox+dt*Vs[vertex3][0], oy+dt*Vs[vertex3][1], oz+dt*Vs[vertex3][2],
                          ox+dt*Vs[vertex4][0], oy+dt*Vs[vertex4][1], oz+dt*Vs[vertex4][2])

    X.append(x)
    Y.append(y)
    Z.append(z)

    x,y,z = interceptQuad(E, 
                          ox+dt*Vs[vertex5][0], oy+dt*Vs[vertex5][1], oz+dt*Vs[vertex5][2],
                          ox+dt*Vs[vertex6][0], oy+dt*Vs[vertex6][1], oz+dt*Vs[vertex6][2])

    X.append(x)
    Y.append(y)
    Z.append(z)


    i.append(vertexIndex[0]+0)
    j.append(vertexIndex[0]+1)
    k.append(vertexIndex[0]+2)

    vertexIndex[0] += 3


def addFacesBasedOnCase(vertexIndex, X,Y,Z, i,j,k, v1,v2,v3,v4,v5,v6,v7,v8, ox, oy, oz, dt, E):

    megaV = (v1==1) + (v2==1)*2 + (v3==1)*4 + (v4==1)*8 + (v5==1)*16 + (v6==1)*32 + (v7==1)*64 + (v8==1)*128

    # CASE 1:


    #  CASE 2:

    if megaV == 0b00000001 or \
       megaV == 0b11111110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 5, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00000100 or \
       megaV == 0b11111011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 7, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001000 or \
       megaV == 0b11110111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00000010 or \
       megaV == 0b11111101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 6, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100000 or \
       megaV == 0b11011111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 2, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010000 or \
       megaV == 0b11101111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 1, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000000 or \
       megaV == 0b10111111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 3, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10000000 or \
       megaV == 0b01111111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 4, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 3:

    if megaV == 0b00000011 or \
       megaV == 0b11111100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 6, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00000101 or \
       megaV == 0b11111010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 5, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001100 or \
       megaV == 0b11110011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 8, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 7, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001010 or \
       megaV == 0b11110101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 6, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 8, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100010 or \
       megaV == 0b11011101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 5, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010001 or \
       megaV == 0b11101110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 7, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000100 or \
       megaV == 0b10111011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 8, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10001000 or \
       megaV == 0b01110111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 6, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00110000 or \
       megaV == 0b11001111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 1, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010000 or \
       megaV == 0b10101111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 3, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000000 or \
       megaV == 0b00111111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 4, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10100000 or \
       megaV == 0b01011111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 2, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 4:

    if megaV == 0b00001001 or \
       megaV == 0b11110110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00000110 or \
       megaV == 0b11111001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 6, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10000010 or \
       megaV == 0b01111101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 7, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100001 or \
       megaV == 0b11011110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010100 or \
       megaV == 0b11101011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01001000 or \
       megaV == 0b10110111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 5, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01100000 or \
       megaV == 0b10011111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 3, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10010000 or \
       megaV == 0b01101111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 4, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010010 or \
       megaV == 0b11101101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 4, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000001 or \
       megaV == 0b10111110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 2, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10000100 or \
       megaV == 0b01111011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 1, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00101000 or \
       megaV == 0b11010111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 3, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 5:

    if megaV == 0b00110010 or \
       megaV == 0b11001101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 8, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 1, 5, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010001 or \
       megaV == 0b10101110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 6, 5, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 3, 7, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000100 or \
       megaV == 0b00111011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 5, 7, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 4, 8, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10101000 or \
       megaV == 0b01010111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 7, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 2, 6, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00110001 or \
       megaV == 0b11001110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 7, 5, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 2, 1, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 6, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010100 or \
       megaV == 0b10101011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 8, 7, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 1, 3, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11001000 or \
       megaV == 0b00110111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 6, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 3, 4, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10100010 or \
       megaV == 0b01011101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 5, 6, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 4, 2, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 8, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010011 or \
       megaV == 0b11101100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 3, 1, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 6, 2, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000101 or \
       megaV == 0b10111010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 4, 3, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 5, 1, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10001100 or \
       megaV == 0b01110011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 2, 4, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 7, 3, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00101010 or \
       megaV == 0b11010101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 1, 2, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 8, 4, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100011 or \
       megaV == 0b11011100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 4, 2, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 5, 6, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010101 or \
       megaV == 0b11101010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 2, 1, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 7, 5, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01001100 or \
       megaV == 0b10110011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 1, 3, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 8, 7, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10001010 or \
       megaV == 0b01110101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 3, 4, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 6, 8, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11100000 or \
       megaV == 0b00011111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 4, 8, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 5, 7, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10110000 or \
       megaV == 0b01001111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 2, 6, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 7, 8, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01110000 or \
       megaV == 0b10001111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 1, 5, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 8, 6, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11010000 or \
       megaV == 0b00101111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 3, 7, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 6, 5, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001101 or \
       megaV == 0b11110010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 7, 3, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 2, 4, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001110 or \
       megaV == 0b11110001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 8, 4, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 1, 2, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00001011 or \
       megaV == 0b11110100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 6, 2, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 3, 1, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00000111 or \
       megaV == 0b11111000:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 5, 1, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 4, 3, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 6:

    if megaV == 0b00110011 or \
       megaV == 0b11001100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 2, 4, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 8, 6, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010101 or \
       megaV == 0b10101010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 1, 2, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 6, 5, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11110000 or \
       megaV == 0b00001111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 6, 2, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 4, 8, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 7:

    if megaV == 0b00110110 or \
       megaV == 0b11001001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 8, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 1, 5, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01011001 or \
       megaV == 0b10100110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 6, 5, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 3, 7, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000110 or \
       megaV == 0b00111001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 5, 7, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 4, 8, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10101001 or \
       megaV == 0b01010110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 7, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 2, 6, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10010011 or \
       megaV == 0b01101100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 3, 1, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 6, 2, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01100101 or \
       megaV == 0b10011010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 4, 3, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 5, 1, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10011100 or \
       megaV == 0b01100011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 2, 4, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 7, 3, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01101010 or \
       megaV == 0b10010101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 1, 2, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 8, 4, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11100001 or \
       megaV == 0b00011110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 4, 8, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 5, 7, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10110100 or \
       megaV == 0b01001011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 2, 6, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 7, 8, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01111000 or \
       megaV == 0b10000111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 1, 5, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 8, 6, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11010010 or \
       megaV == 0b00101101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 3, 7, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 6, 5, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 8:

    if megaV == 0b01101001 or \
       megaV == 0b10010110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 9:

    if megaV == 0b01110001 or \
       megaV == 0b10001110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 3, 7, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 7, 8, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 8, 6, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11010100 or \
       megaV == 0b00101011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 4, 8, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 8, 6, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 6, 5, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11101000 or \
       megaV == 0b00010111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 2, 6, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 6, 5, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 5, 7, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10110010 or \
       megaV == 0b01001101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 1, 5, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 5, 7, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 7, 8, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 10:

    if megaV == 0b01110010 or \
       megaV == 0b10001101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 1, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 7, 8, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 1, 2, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 8, 6, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11010001 or \
       megaV == 0b00101110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 3, 7, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 8, 6, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 3, 1, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 6, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11100100 or \
       megaV == 0b00011011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 4, 8, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 6, 5, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 4, 3, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 5, 7, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10111000 or \
       megaV == 0b01000111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 2, 6, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 5, 7, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 2, 4, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 7, 8, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00110101 or \
       megaV == 0b11001010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 2, 1, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 3, 7, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 2, 6, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 7, 5, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01011100 or \
       megaV == 0b10100011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 1, 3, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 4, 8, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 1, 5, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 8, 7, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 11:

    if megaV == 0b10000001 or \
       megaV == 0b01111110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100100 or \
       megaV == 0b11011011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00011000 or \
       megaV == 0b11100111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 8, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000010 or \
       megaV == 0b10111101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 6, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 12:

    if megaV == 0b10000011 or \
       megaV == 0b01111100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 2, 4, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100101 or \
       megaV == 0b11011010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 1, 2, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00011100 or \
       megaV == 0b11100011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 8, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 3, 1, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01001010 or \
       megaV == 0b10110101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 6, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 4, 3, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01100010 or \
       megaV == 0b10011101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 6, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10010001 or \
       megaV == 0b01101110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 5, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01100100 or \
       megaV == 0b10011011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 7, 5, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10011000 or \
       megaV == 0b01100111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 8, 7, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00110100 or \
       megaV == 0b11001011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 5, 7, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 7, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01011000 or \
       megaV == 0b10100111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 7, 8, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 8, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000010 or \
       megaV == 0b00111101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 8, 6, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 6, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10100001 or \
       megaV == 0b01011110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 6, 5, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 5, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00011001 or \
       megaV == 0b11100110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 1, 3, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000110 or \
       megaV == 0b10111001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 3, 4, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10001001 or \
       megaV == 0b01110110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 4, 2, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00100110 or \
       megaV == 0b11011001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 2, 1, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00111000 or \
       megaV == 0b11000111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 6, 2, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010010 or \
       megaV == 0b10101101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 5, 1, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000001 or \
       megaV == 0b00111110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 7, 3, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10100100 or \
       megaV == 0b01011011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 8, 4, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01000011 or \
       megaV == 0b10111100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 1, 5, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10000101 or \
       megaV == 0b01111010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 3, 7, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00101100 or \
       megaV == 0b11010011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 4, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00011010 or \
       megaV == 0b11100101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 2, 6, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 13:

    if megaV == 0b10000110 or \
       megaV == 0b01111001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 1, 2, 4, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 4, 3, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 7, 8, 4, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00101001 or \
       megaV == 0b11010110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 3, 1, 2, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 2, 4, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 6, 2, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00010110 or \
       megaV == 0b11101001:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 4, 3, 1, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 1, 2, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 5, 1, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01001001 or \
       megaV == 0b10110110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 2, 4, 3, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 3, 1, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 5, 7, 3, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01101000 or \
       megaV == 0b10010111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 8, 4, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 8, 7, 5),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10010010 or \
       megaV == 0b01101101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 6, 2, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 6, 8, 7),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01100001 or \
       megaV == 0b10011110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 3, 7, 5, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 5, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 2, 6, 5, 6, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10010100 or \
       megaV == 0b01101011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 4, 8, 7, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 7, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 1, 5, 7, 5, 6),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 14:

    if megaV == 0b10100101 or \
       megaV == 0b01011010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 5, 6, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 3, 4, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 4, 8, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b00111100 or \
       megaV == 0b11000011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 7, 5, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 4, 2, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 2, 6, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b10011001 or \
       megaV == 0b01100110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 7, 8, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 6, 1, 2, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 8, 2, 4, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        return



    #  CASE 15:

    if megaV == 0b10110001 or \
       megaV == 0b01001110:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 1, 3, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 5, 7, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 2, 8, 4, 2, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(7, 8, 7, 5, 8, 4),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01110100 or \
       megaV == 0b10001011:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 3, 4, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 7, 8, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 1, 6, 2, 1, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(8, 6, 8, 7, 6, 2),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11011000 or \
       megaV == 0b00100111:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 4, 2, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 8, 6, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 3, 5, 1, 3, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(6, 5, 6, 8, 5, 1),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11100010 or \
       megaV == 0b00011101:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 2, 1, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 6, 5, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 4, 7, 3, 4, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(5, 7, 5, 6, 7, 3),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b01010011 or \
       megaV == 0b10101100:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 2, 4, 1, 3),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 1, 3, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(2, 6, 7, 8, 6, 5),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(3, 7, 3, 1, 7, 8),   vertexIndex, ox, oy, oz, dt, E)
        return
    if megaV == 0b11000101 or \
       megaV == 0b00111010:
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 1, 2, 3, 4),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 3, 4, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(1, 5, 8, 6, 5, 7),   vertexIndex, ox, oy, oz, dt, E)
        addFaceBasedOnMiddlePoints(X,Y,Z, i,j,k,  *(4, 8, 4, 3, 8, 6),   vertexIndex, ox, oy, oz, dt, E)
        return
        
def marchingCubes(lim, N, E):
    """
    returns a set of vertex X,Y,Z and faces i,j,k for the plot of implicit equation eqn

         7 +----------+ 8
          /|         /|
         / |        / |
      3 +----------+ 4|
        |  |       |  |
        |5 +-------|--+ 6
        | /        | /
        |/         |/ 
      1 +----------+ 2
       
        
        y   z
        ^  ^
        |/
        . -> x

    """

    Vs_ = {
        1: (0,0,0),
        2: (1,0,0),
        5: (0,1,0),
        6: (1,1,0),
        3: (0,0,1),
        4: (1,0,1),
        7: (0,1,1),
        8: (1,1,1)
    }       

    vertexIndex = [0]
        
    dt = 2.0*lim/(N-1)
    
    
    X = []
    Y = []
    Z = []

    i = []
    j = []
    k = []    


    cubes = []    

    quadValuesInTheWholeCube = np.zeros((N,N,N)).astype('int8')

    tic = time.time()
    for i_,x_ in enumerate( np.linspace(-lim, lim, N) ):
        for j_,y_ in enumerate( np.linspace(-lim, lim, N) ):
            for k_,z_ in enumerate( np.linspace(-lim, lim, N) ):
                x = x_ - dt/2.01
                y = y_ - dt/2.01
                z = z_ - dt/2.01

                quadValuesInTheWholeCube[i_,j_,k_] = int(np.sign(evalQuad(E,x,y,z)))

    toc = time.time()
    print(f'Time to evaluate the cube {toc-tic}s')

    tic = time.time()
    for i_,x_ in enumerate( np.linspace(-lim, lim, N) ):
        for j_,y_ in enumerate( np.linspace(-lim, lim, N) ):
            for k_,z_ in enumerate( np.linspace(-lim, lim, N) ):

                if i_ < N-1 and j_ < N-1 and k_ < N-1:

                    x = x_ - dt/2.01
                    y = y_ - dt/2.01
                    z = z_ - dt/2.01

                    v1 = quadValuesInTheWholeCube[i_+Vs_[1][0],j_+Vs_[1][1],k_+Vs_[1][2]]
                    v2 = quadValuesInTheWholeCube[i_+Vs_[2][0],j_+Vs_[2][1],k_+Vs_[2][2]]
                    v3 = quadValuesInTheWholeCube[i_+Vs_[3][0],j_+Vs_[3][1],k_+Vs_[3][2]]
                    v4 = quadValuesInTheWholeCube[i_+Vs_[4][0],j_+Vs_[4][1],k_+Vs_[4][2]]
                    v5 = quadValuesInTheWholeCube[i_+Vs_[5][0],j_+Vs_[5][1],k_+Vs_[5][2]]
                    v6 = quadValuesInTheWholeCube[i_+Vs_[6][0],j_+Vs_[6][1],k_+Vs_[6][2]]
                    v7 = quadValuesInTheWholeCube[i_+Vs_[7][0],j_+Vs_[7][1],k_+Vs_[7][2]]
                    v8 = quadValuesInTheWholeCube[i_+Vs_[8][0],j_+Vs_[8][1],k_+Vs_[8][2]]
                    
                    vt = v1+v2+v3+v4+v5+v6+v7+v8

                    if not (vt==8 or vt==-8):
                        addFacesBasedOnCase(vertexIndex, X,Y,Z, i,j,k,  v1,v2,v3,v4,v5,v6,v7,v8,  x, y, z,  dt,  E)
                        cubes.append((x,y,z,tuple()))

    toc = time.time()
    #print(f'Time to face up the quad {toc-tic}s')
                
    return X,Y,Z,i,j,k, cubes, dt




def setPatches(X,Y,Z, i,j,k, lim, cubes, dt, color, opacity=1.0):

    Xcm1 = []
    Ycm1 = []
    Zcm1 = []
    Xcm0 = []
    Ycm0 = []
    Zcm0 = []
    
    Xcl = []
    Ycl = []
    Zcl = []
    
    Xe = []
    Ye = []
    Ze = []

    Pc = [[0, dt,  0, dt,  0, dt,  0, dt],
          [0,  0, dt, dt,  0,  0, dt, dt],
          [0,  0,  0,  0, dt, dt, dt, dt]]
    
    
    for p in cubes:
        
        for _ in range(len(Pc[0])):
            if (_+1) in p[3]:
                Xcm1 += [Pc[0][_]+p[0]]
                Ycm1 += [Pc[2][_]+p[1]]
                Zcm1 += [Pc[1][_]+p[2]]
            else:
                Xcm0 += [Pc[0][_]+p[0]]
                Ycm0 += [Pc[2][_]+p[1]]
                Zcm0 += [Pc[1][_]+p[2]]
            
        
#         Xcm += [x+p[0] for x in [0, dt, 0, dt, 0, dt, 0, dt]]
#         Ycm += [y+p[1] for y in [0, 0, dt, dt, 0, 0, dt, dt]]
#         Zcm += [z+p[2] for z in [0, 0, 0, 0, dt, dt, dt, dt]]
        
        Xcl += [x+p[0] for x in [0, dt, dt, 0, 0, np.nan, 0, dt, dt, 0, 0, np.nan, 0, 0, np.nan, dt, dt, np.nan, 0, 0, np.nan, dt, dt, np.nan]]
        Ycl += [y+p[1] for y in [0, 0, 0, 0, 0, np.nan, dt, dt, dt, dt, dt, np.nan, 0, dt, np.nan, 0, dt, np.nan, 0, dt, np.nan, 0, dt, np.nan]]
        Zcl += [z+p[2] for z in [0, 0, dt, dt, 0, np.nan, 0, 0, dt, dt, 0, np.nan, 0, 0, np.nan, 0, 0, np.nan, dt, dt, np.nan, dt, dt, np.nan]]

        
        
    for xe, ye, ze, _ in zip(X,Y,Z, range(len(X))):
        if _%3 == 0:
            Xe.append(np.nan)
            Ye.append(np.nan)
            Ze.append(np.nan)
        Xe.append(xe)
        Ye.append(ye)
        Ze.append(ze)
            
    traces = [
        go.Mesh3d(
            x=X,
            y=Y,
            z=Z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
        ),
        
        go.Scatter3d(
            x=Xcm1,
            y=Ycm1,
            z=Zcm1,
            mode='markers',
            marker={'color':'red'}
        ),

        go.Scatter3d(
            x=Xcm0,
            y=Ycm0,
            z=Zcm0,
            mode='markers',
            marker={'color':'white'}
        ),
        
        go.Scatter3d(
            x=Xcl,
            y=Ycl,
            z=Zcl,
            mode='lines'
        ),
        
        go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode='lines',
            line={'color': '#202020', 'width':3}
        )
    ]
    
    return traces


def plotTraces(traces, lim):

    fig = go.Figure(data=traces)
    

    fig.update_scenes(camera = {
        'up':{'x':0, 'y':0, 'z':1},
        'center':{'x':0, 'y':0, 'z':0},
        'eye':{'x':lim/6, 'y':-lim/6, 'z':lim/6}
        },
        aspectmode = 'cube', 
        xaxis={'range':[-lim*1.6,lim*1.6]}, 
        yaxis={'range':[-lim*1.6,lim*1.6]}, 
        zaxis={'range':[-lim*1.6,lim*1.6]})

    
    fig.update_layout(
        title='Quadric',
        width=1024, height=1024,
        margin={'t':0, 'r':0, 'l':0, 'b':0},
        showlegend=False
    )

    fig.show()



def plot_quadrics(Es, lim, N, colors, withCubes = False):

    traces = []

    for E,color in zip(Es, colors):
        X,Y,Z,i,j,k, cubes, dt = marchingCubes(lim, N, E)
        traces += setPatches(X,Y,Z,i,j,k, lim, cubes if withCubes else [], dt, color)

    plotTraces(traces, lim)


