import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm

flag_P1 = False
flag_P2 = True
flag_P3 = False
flag_P4 = False

if flag_P4:
    # black curve integral approximation
    black_nopeak =  2*(0.3/2+1) + 1*(0.1/2+2.5) + 1*(0.1/2+3.0) + 2*(0.1/2+0.9) + 4*(0.4/2+0.9) + 4*(0.9/2+1.3) + 1*(0.3/2+2.2) + 1*(0.1/2+2.9) + 2*(0.4/2+2.5) + 1*(0.4/2+2.1)
    black_peak =   2*(0.3/2+2.6) + 2*(0.2/2+2.9) 
    black = black_peak + black_nopeak
    # red curve integral approximation
    red_nopeak = 1*(1.0/2+1.7) + 1*(1.1/2+2.1) +2*(0.3/2+1.1) + 2*(0.1/2+1.0) + 4*(0.4/2+1.0) + 5*(0.7/2+1.4) + 1*(0.1/2+3.1) + 2*(0.4/2+2.7) + 1*(0.6/2 +2.1)
    red_peak =  4*(0.4/2+1.7) 
    red = red_peak + red_nopeak

    print("black peak {}, no peak {}, y total {}".format(black_peak,black_nopeak, black))
    print("red peak {}, no peak {}, y total {}".format(red_peak,red_nopeak,red))

    elasticity_own = ((red_peak-black_peak)/(1.3-0.15)) * ((1.3+0.15)/(black_peak+red_peak))
    print("elasticity_own", elasticity_own)


    elasticity_cross = ((red_nopeak-black_nopeak)/(1.3-0.15)) * ((1.3+0.15)/(red_nopeak+black_nopeak))
    print("elasticity_cross", elasticity_cross)

if flag_P3:

    def objective_function(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2):
        ct_p13 = 2; ct_p14 = 3; ct_p23 = 4; ct_p24 = 2;  ct_l37 = 1; ct_l38 = 1.2; ct_l47 = 1; ct_l48 = 1.5; ct_h37 = 1
        ct_h38 = 1.2; ct_h47 = 1; ct_h48 = 1.5; ct_c57 = 1; ct_c58 = 2.5; ct_c67 = 0.75; ct_c68 = 2.75; cr_q3 = 6.5; cr_q4 = 5
        cx_p1_1 = 1; cx_p1_2 = 1.5; cx_p2_1 = 1.25; cx_p2_2 = 1.5; cx_c5_1 = 5; cx_c5_2 = 6; cx_c5_3 = 8; cx_c6_1 = 4; cx_c6_2 = 5; cx_c6_3 = 7
        
        value_1 = 68*ql7 + 65*qh7 + 47.2*qc7 + (1/2)*((-0.03)*ql7*ql7 + (-0.01)*qh7*ql7 + (-0.006)*qc7*ql7+(-0.01)*ql7*qh7 + (-0.03)*qh7*qh7 + (-0.009)*qc7*qh7+(-0.006)*ql7*qc7 + (-0.009)*qh7*qc7 + (-0.019)*qc7*qc7)
        value_2 = 68*ql8 + 65*qh8 + 47.2*qc8 + (1/2)*((-0.03)*ql8*ql8 + (-0.01)*qh8*ql8 + (-0.006)*qc8*ql8+(-0.01)*ql8*qh8 + (-0.03)*qh8*qh8 + (-0.009)*qc8*qh8+(-0.006)*ql8*qc8 + (-0.009)*qh8*qc8 + (-0.019)*qc8*qc8)
        value_3 = -(cx_p1_1*x_p1_1+cx_p1_2*x_p1_2+cx_p2_1*x_p2_1+cx_p2_2*x_p2_2+cx_c5_1*x_c5_1+cx_c5_2*x_c5_2+cx_c5_3*x_c5_3+cx_c6_1*x_c6_1+cx_c6_2*x_c6_2+cx_c6_3*x_c6_3)
        value_4 = -(ct_p13*t_p13+ct_p14*t_p14+ct_p23*t_p23+ct_p24*t_p24+ct_l37*t_l37+ct_l38*t_l38+ct_l47*t_l47+ct_l48*t_l48+ct_h37*t_h37+ct_h38*t_h38+ct_h47*t_h47+ct_h48*t_h48+ct_c57*t_c57+ct_c58*t_c58+ct_c67*t_c67+ct_c68*t_c68)
        value_5 = -(cr_q3*qp3+cr_q4*qp4)
        return value_1 + value_2 + value_3 + value_4 + value_5
    
    def coal_production_5(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2):
        ct_p13 = 2; ct_p14 = 3; ct_p23 = 4; ct_p24 = 2;  ct_l37 = 1; ct_l38 = 1.2; ct_l47 = 1; ct_l48 = 1.5; ct_h37 = 1
        ct_h38 = 1.2; ct_h47 = 1; ct_h48 = 1.5; ct_c57 = 1; ct_c58 = 2.5; ct_c67 = 0.75; ct_c68 = 2.75; cr_q3 = 6.5; cr_q4 = 5
        cx_p1_1 = 1; cx_p1_2 = 1.5; cx_p2_1 = 1.25; cx_p2_2 = 1.5; cx_c5_1 = 5; cx_c5_2 = 6; cx_c5_3 = 8; cx_c6_1 = 4; cx_c6_2 = 5; cx_c6_3 = 7
        
        value_1 = p_c5*(x_c5_1+x_c5_2+x_c5_3)
        value_2 = (cx_c5_1*x_c5_1+cx_c5_2*x_c5_2+cx_c5_3*x_c5_3)
        value_3 = 2.2*p_c02*(x_c5_1+x_c5_2+x_c5_3)
        return (value_1-value_2-value_3), value_1, value_2, value_3

    def coal_production_6(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2):
        ct_p13 = 2; ct_p14 = 3; ct_p23 = 4; ct_p24 = 2;  ct_l37 = 1; ct_l38 = 1.2; ct_l47 = 1; ct_l48 = 1.5; ct_h37 = 1
        ct_h38 = 1.2; ct_h47 = 1; ct_h48 = 1.5; ct_c57 = 1; ct_c58 = 2.5; ct_c67 = 0.75; ct_c68 = 2.75; cr_q3 = 6.5; cr_q4 = 5
        cx_p1_1 = 1; cx_p1_2 = 1.5; cx_p2_1 = 1.25; cx_p2_2 = 1.5; cx_c5_1 = 5; cx_c5_2 = 6; cx_c5_3 = 8; cx_c6_1 = 4; cx_c6_2 = 5; cx_c6_3 = 7
        
        value_1 = p_c6*(x_c6_1+x_c6_2+x_c6_3)
        value_2 = (cx_c6_1*x_c6_1+cx_c6_2*x_c6_2+cx_c6_3*x_c6_3)
        value_3 = 2.2*p_c02*(x_c6_1+x_c6_2+x_c6_3)
        return (value_1-value_2-value_3), value_1, value_2, value_3

    def oil_refinery_3(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2):
        ct_p13 = 2; ct_p14 = 3; ct_p23 = 4; ct_p24 = 2;  ct_l37 = 1; ct_l38 = 1.2; ct_l47 = 1; ct_l48 = 1.5; ct_h37 = 1
        ct_h38 = 1.2; ct_h47 = 1; ct_h48 = 1.5; ct_c57 = 1; ct_c58 = 2.5; ct_c67 = 0.75; ct_c68 = 2.75; cr_q3 = 6.5; cr_q4 = 5
        cx_p1_1 = 1; cx_p1_2 = 1.5; cx_p2_1 = 1.25; cx_p2_2 = 1.5; cx_c5_1 = 5; cx_c5_2 = 6; cx_c5_3 = 8; cx_c6_1 = 4; cx_c6_2 = 5; cx_c6_3 = 7
        
        value_1 = p_l3*x_l3
        value_2 = p_h3*x_h3
        value_3 = (6.5+p_p3)*qp3
        return (value_1+value_2-value_3), value_1+value_2, value_3

    def oil_refinery_4(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2):
        ct_p13 = 2; ct_p14 = 3; ct_p23 = 4; ct_p24 = 2;  ct_l37 = 1; ct_l38 = 1.2; ct_l47 = 1; ct_l48 = 1.5; ct_h37 = 1
        ct_h38 = 1.2; ct_h47 = 1; ct_h48 = 1.5; ct_c57 = 1; ct_c58 = 2.5; ct_c67 = 0.75; ct_c68 = 2.75; cr_q3 = 6.5; cr_q4 = 5
        cx_p1_1 = 1; cx_p1_2 = 1.5; cx_p2_1 = 1.25; cx_p2_2 = 1.5; cx_c5_1 = 5; cx_c5_2 = 6; cx_c5_3 = 8; cx_c6_1 = 4; cx_c6_2 = 5; cx_c6_3 = 7
        
        value_1 = p_l4*x_l4
        value_2 = p_h4*x_h4
        value_3 = (5+p_p4)*qp4
        return (value_1+value_2-value_3), value_1+value_2, value_3

    # Case CO2 with cross-price elasticities
    x_p1_1 = 1100;    x_p1_2 = 1200;    x_p2_1 = 1300;    x_p2_2 = 1100;    x_c5_1 = 300;    x_c5_2 = 300;    x_c5_3 = 0;    x_c6_1 = 200;    x_c6_2 = 300;    x_c6_3 = 125.91
    t_p13 = 2136.99;    t_p14 = 163.02;    t_p23 = 0;    t_p24 = 2400;    t_l37 = 0;    t_l38 = 1282.19;    t_l47 = 1282.51;    t_l48 = 0
    qp3 = 2136.99;    qp4 = 2563.02
    t_h37 = 0;    t_h38 = 854.79;    t_h47 = 1064.16;    t_h48 = 217.35;    t_c57 = 28.52;    t_c58 = 571.48;    t_c67 = 625.91;    t_c68 = 0
    ql7 = 1281.51;    qh7 = 1064.16;    qc7 = 654.43;    ql8 = 1282.198;    qh8 = 1072.15;    qc8 = 571.48
    x_l3 = 1282.19;    x_l4 = 1281.51;    x_h3 = 854.80;    x_h4 = 1281.5
    p_p1 = 4.678;    p_p2 = 5.678;    p_p3 = 6.678;    p_p4 = 7.678;    p_l3 = 14.184;    p_l4 = 13.987;    p_l7 = 14.987;    p_l8 = 15.384;    p_h3 = 11.67;    p_h4 = 11.37;    p_h7 = 12.37;    p_h8 = 12.87;    p_c5 = 16.5;    p_c6 = 16.75;    p_c7 = 17.45;    p_c8 = 19;    p_c02 = 4.432
    e_co2 = 5000 

    fo_wcarbon = objective_function(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_carbon_5_wcarbon = coal_production_5(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_carbon_6_wcarbon = coal_production_6(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_oil_refinery_3_wcarbon = oil_refinery_3(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_oil_refinery_4_wcarbon = oil_refinery_4(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    
    # Case CO2 without cross-price elasticities
    x_p1_1 = 1100;    x_p1_2 = 931.73;    x_p2_1 = 1300;    x_p2_2 = 1100;    x_c5_1 = 300;    x_c5_2 = 300;    x_c5_3 = 0;    x_c6_1 = 200;    x_c6_2 = 3000;    x_c6_3 = 185.66
    t_p13 = 2031.73;    t_p14 = 0;    t_p23 = 0;    t_p24 = 2400;    t_l37 = 4.53;    t_l38 = 1214.51;    t_l47 = 1200;    t_l48 = 0
    qp3 = 2031.73;    qp4 = 2400
    t_h37 = 0;    t_h38 = 812.69;    t_h47 = 1001.35;    t_h48 = 198.65;    t_c57 = 0;    t_c58 = 600;    t_c67 = 685.66;    t_c68 = 0
    ql7 = 1204.52;    qh7 = 1001.34;    qc7 = 685.66;    ql8 = 1214.51;    qh8 = 1011.34;    qc8 = 600
    x_l3 = 1219.04;    x_l4 = 1200;    x_h3 = 812.69;    x_h4 = 1200
    p_p1 = 3.274;    p_p2 = 4.218;    p_p3 = 5.274;    p_p4 = 6.218;    p_l3 = 13.401;    p_l4 = 13.401;    p_l7 = 14.401;    p_l8 = 14.601;    p_h3 = 9.334;    p_h4 = 9.034;    p_h7 = 10.034;    p_h8 = 10.534;    p_c5 = 14.74;    p_c6 = 14.968;    p_c7 = 15.713;    p_c8 = 17.24;    p_c02 = 3.622
    e_co2 = 5000

    fo_wocarbon = objective_function(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_carbon_5_wocarbon = coal_production_5(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_carbon_6_wocarbon = coal_production_6(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_oil_refinery_3_wocarbon = oil_refinery_3(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    max_oil_refinery_4_wocarbon = oil_refinery_4(x_p1_1,x_p1_2,x_p2_1,x_p2_2,x_c5_1,x_c5_2,x_c5_3,x_c6_1,x_c6_2,x_c6_3,t_p13,t_p14,t_p23,t_p24,t_l37,t_l38,t_l47,t_l48,qp3,qp4,t_h37,t_h38,t_h47,t_h48,t_c57,t_c58,t_c67,t_c68,ql7,qh7,qc7,ql8,qh8,qc8,x_l3,x_l4,x_h3,x_h4,p_p1,p_p2,p_p3,p_p4,p_l3,p_l4,p_l7,p_l8,p_h3,p_h4,p_h7,p_h8,p_c5,p_c6,p_c7,p_c8,p_c02, e_co2)
    

    print("FO Case with cross-price elasticities {}".format(fo_wcarbon))
    print("C5 FO is {}, revenue is {}, cost is {}, and taxes {} (Case with cross-price elasticities)".format(max_carbon_5_wcarbon[0], max_carbon_5_wcarbon[1], max_carbon_5_wcarbon[2], max_carbon_5_wcarbon[3]))
    print("C6 FO is {}, revenue is {}, cost is {}, and taxes {} (Case with cross-price elasticities)".format(max_carbon_6_wcarbon[0], max_carbon_6_wcarbon[1], max_carbon_6_wcarbon[2], max_carbon_6_wcarbon[3]))
    print("R3 FO is {}, sales is {}, cost is {} (Case with cross-price elasticities)".format(max_oil_refinery_3_wcarbon[0],max_oil_refinery_3_wcarbon[1],max_oil_refinery_3_wcarbon[2]))
    print("R4 FO is {}, sales is {}, cost is {} (Case with cross-price elasticities)".format(max_oil_refinery_4_wcarbon[0],max_oil_refinery_4_wcarbon[1],max_oil_refinery_4_wcarbon[2]))

    print("FO Case without cross-price elasticities {}".format(fo_wocarbon))
    print("C5 FO is {}, revenue is {}, cost is {}, and taxes {} (Case without cross-price elasticities)".format(max_carbon_5_wocarbon[0], max_carbon_5_wocarbon[1], max_carbon_5_wocarbon[2], max_carbon_5_wocarbon[3]))
    print("C6 FO is {}, revenue is {}, cost is {}, and taxes {} (Case without cross-price elasticities)".format(max_carbon_6_wocarbon[0], max_carbon_6_wocarbon[1], max_carbon_6_wocarbon[2], max_carbon_6_wocarbon[3]))
    print("R3 FO is {}, sales is {}, cost is {} (Case without cross-price elasticities)".format(max_oil_refinery_3_wocarbon[0],max_oil_refinery_3_wocarbon[1],max_oil_refinery_3_wocarbon[2]))
    print("R4 FO is {}, sales is {}, cost is {} (Case without cross-price elasticities)".format(max_oil_refinery_4_wocarbon[0],max_oil_refinery_4_wocarbon[1],max_oil_refinery_4_wocarbon[2]))

if flag_P1:
    # parameters
    d1 = 2693.2
    d2 = 5848.6
    s1 =  2668.2
    s2  = 5873.6
    p1 = 109.60
    p2 = 102.52
    nox1 = 6.67
    nox2 = 8.33
    pnox = 22236
    t12 = 25

    def supplier_1(p1,s1,pnox,nox1):
        return p1*s1 -(30*s1 +0.5*0.009*s1*s1) - pnox*nox1

    def supplier_2(p2,s2,pnox,nox2):
        return p2*s2 -(50*s2 +0.5*0.0025*s2*s2) - pnox*nox2

    def consumer_1(d1,p1):
        return (150*d1 -0.5*0.015*d1*d1)-p1*d1

    def consumer_2(d2,p2):
        return (200*d2 -0.5*0.016667*d2*d2)-p2*d2

    def EPA(pnox,nox1,nox2):
        return pnox*(nox1+nox2)
    
    def trasporter(p1,p2,t12):
        return (p1-p2)*t12 - 5*t12

    
    v_supplier_1=supplier_1(p1,s1,pnox,nox1)
    v_supplier_2=supplier_2(p2,s2,pnox,nox2)
    v_consumer_1=consumer_1(d1,p1)
    v_consumer_2=consumer_2(d2,p2)
    v_EPA=EPA(pnox,nox1,nox2)
    v_trasporter=trasporter(p1,p2,t12)

    print("supplier 1 {}, supplier 2 {}, consumer 1 {}, consumer 2 {}, Tx {}, and EPA {}".format(v_supplier_1,v_supplier_2,v_consumer_1, v_consumer_2,v_trasporter,v_EPA))
    print("the total value is", v_supplier_1+v_supplier_2+v_consumer_1+ v_consumer_2+v_EPA+v_trasporter)

if flag_P2:



    def problem2():
        N = 2
        aux = 1/265920

        # Model 
        model = gb.Model()

        # Variables
        s_n = model.addMVar(2, vtype=GRB.CONTINUOUS, name="s_n", lb=0) 
        d_n = model.addMVar(2, vtype=GRB.CONTINUOUS, name="d_n", lb=0)
        
        # Constraints
        cons_demand = model.addConstrs(s_n[n] == d_n[n] for n in range(N))
        cons_max = model.addConstr(s_n[1] <= 130000)

        # Objective function
        obj = model.setObjective(aux*(20*198040+13*171200)*d_n[0] + aux*(13*198040+50*171200)*d_n[1]+
            0.5*aux*( -20*d_n[0]*d_n[0] -13*d_n[0]*d_n[1] - 13*d_n[0]*d_n[1] -50 *d_n[1]*d_n[1])-
            3.75*s_n[0] -  (5.625*s_n[1] + 0.000015625*s_n[1]*s_n[1] ),GRB.MAXIMIZE)

        # Solve
        model.optimize()

        # Print results
        obj = model.getObjective()

        print("\nThe optimal value is", obj.getValue())
        print("A solution s_n is")
        print("{}: {}".format(s_n.varName, s_n.X))
        print("A solution d_n is")
        print("{}: {}".format(d_n.varName, d_n.X))

    problem2()