// Copyright (c) 2013, David Hirvonen
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.

#include "SteerableFilters.h"
#include <opencv2/imgproc/imgproc.hpp>

typedef float (*KernelType)(float x);

static cv::Mat_<float> create(int width, float spacing, KernelType f)
{
    cv::Mat_<float> kernel(1, width * 2 + 1);
    for(int i = -width; i <= width; i++)
        kernel(i+width) = f(float(i)*spacing);
    
    return kernel;
}

// Utility routine to map the opencv atan2 output from [0 2*pi] to [-pi/2 pi/2]
// This isn't truly necessary but is should be helpful to provide compatibility with conventions in the paper
static void wrap(const cv::Mat_<float> &angle, cv::Mat_<float> &output)
{
    output = angle.clone();
    cv::Mat_<float> tmp = (-M_PI - (M_PI - angle));
    tmp.copyTo(output, (angle > M_PI));
}

SteerableFilters::SteerableFilters(const cv::Mat_<float> &image, int width, float spacing)
{
    // Create separable filters for G2
    m_g1 = create(width, spacing, G21);
    m_g2 = create(width, spacing, G22);
    m_g3 = create(width, spacing, G23);

    // Create separable filters for H2
    m_h1 = create(width, spacing, H21);
    m_h2 = create(width, spacing, H22);
    m_h3 = create(width, spacing, H23);
    m_h4 = create(width, spacing, H24);
    
    setup(image);
}

//  phase = arg(G2, H2) where arg(x + iy) = atan(y,x), (opencv return angles in [0..2pi])
//  0      = dark line
//  pi     = bright line
// +pi/2   = edge
// -pi/2   = edge
void SteerableFilters::computeMagnitudeAndPhase(const cv::Mat_<float> &g2, const cv::Mat_<float> &h2, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase)
{
    cv::cartToPolar(g2, h2, magnitude, phase); // [0..2*piI]
    wrap(phase, phase); // [-pi/2 pi/2]
    cv::patchNaNs(phase);
}

void SteerableFilters::steer(float theta, cv::Mat_<float> &g2, cv::Mat_<float> &h2, cv::Mat_<float> &e, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    float ct(cos(theta)), ct2(ct*ct), ct3(ct2*ct), st(sin(theta)), st2(st*st), st3(st2*st);
    g2 = ct2 * m_g2a + (-2.0 * ct * st * m_g2b) + (st2 * m_g2c);
    h2 = ct3 * m_h2a + (-3.0 * ct2 * st * m_h2b) + (3.0 * ct * st2 * m_h2c) + (-st3 * m_h2d);
    computeMagnitudeAndPhase(g2, h2, magnitude, phase);
    phase.setTo(0, (magnitude < 1e-6f));
    
    // Compute oriented energy as a function of angle theta
    float c2t(cos(theta*2.0)), s2t(sin(theta*2.0));
    e = m_c1 + (c2t * m_c2) + (s2t * m_c3);
}

void SteerableFilters::steer(const cv::Mat_<float> &theta, cv::Mat_<float> &g2, cv::Mat_<float> &h2, cv::Mat_<float> &e, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    cv::Mat_<float> ct, ct2, ct3, st, st2, st3;
    cv::polarToCart(cv::Mat(), theta, ct, st);
    ct2 = ct.mul(ct), ct3 = ct2.mul(ct);
    st2 = st.mul(st), st3 = st2.mul(st);
    g2 = ct2.mul(m_g2a) + (-2.0 * ct.mul(st).mul(m_g2b)) + (st2.mul(m_g2c));
    h2 = ct3.mul(m_h2a) + (-3.0 * ct2.mul(st).mul(m_h2b)) + (3.0 * ct.mul(st2).mul(m_h2c)) + (-st3.mul(m_h2d));
    computeMagnitudeAndPhase(g2, h2, magnitude, phase);
    
    // Compute oriented energy as a function of angle theta
    cv::Mat_<float> c2t, s2t;
    cv::polarToCart(cv::Mat(), theta * 2.0, c2t, s2t);
    e = m_c1 + m_c2.mul(c2t) + m_c3.mul(s2t);
}

void SteerableFilters::phaseLine1(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &lines, float k)
{
    cv::Mat_<float> phaseOffset = phase, cp, sp, lambda;
    cv::polarToCart(cv::Mat(), phaseOffset, cp, sp); // implicitly (phase - 0)
    cv::pow(cp, k, lambda);
    lambda.setTo(0, cv::abs(phaseOffset) > M_PI_2 );
    lines = e.mul(lambda);
}

void SteerableFilters::phaseLine0(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &lines, float k)
{
    cv::Mat_<float> phaseOffset = phase - M_PI, cp, sp, lambda;
    cv::polarToCart(cv::Mat(), phaseOffset, cp, sp); // implicitly (phase - 0)
    cv::pow(cp, k, lambda);
    lambda.setTo(0, cv::abs(phaseOffset) > M_PI_2 );
    lines = e.mul(lambda);
}

void SteerableFilters::phaseEdge(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &edges, float k)
{
    cv::Mat_<float> phaseOffset = cv::abs(phase) - M_PI_2, cp, sp, lambda;
    cv::polarToCart(cv::Mat(), phaseOffset, cp, sp);
    cv::pow(cp, k, lambda);
    lambda.setTo(0, cv::abs(phaseOffset) > M_PI_2 );
    edges = e.mul(lambda);
}

void SteerableFilters::phaseEdge01(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &edges, float k)
{
    cv::Mat_<float> phaseOffset = phase - M_PI_2, cp, sp, lambda;
    cv::polarToCart(cv::Mat(), phaseOffset, cp, sp);
    cv::pow(cp, k, lambda);
    lambda.setTo(0, cv::abs(phaseOffset) > M_PI_2 );
    edges = e.mul(lambda);
}

void SteerableFilters::phaseEdge10(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &edges, float k)
{
    cv::Mat_<float> phaseOffset = phase + M_PI_2, cp, sp, lambda;
    cv::polarToCart(cv::Mat(), phaseOffset, cp, sp);
    cv::pow(cp, k, lambda);
    lambda.setTo(0, cv::abs(phaseOffset) > M_PI_2 );
    edges = e.mul(lambda);
}

void SteerableFilters::setup(const cv::Mat_<float> &image)
{
    cv::sepFilter2D(image, m_g2a, CV_32FC1, m_g1, m_g2.t());
    cv::sepFilter2D(image, m_g2b, CV_32FC1, m_g3, m_g3.t());
    cv::sepFilter2D(image, m_g2c, CV_32FC1, m_g2, m_g1.t());
    cv::sepFilter2D(image, m_h2a, CV_32FC1, m_h1, m_h2.t());
    cv::sepFilter2D(image, m_h2b, CV_32FC1, m_h4, m_h3.t());
    cv::sepFilter2D(image, m_h2c, CV_32FC1, m_h3, m_h4.t());
    cv::sepFilter2D(image, m_h2d, CV_32FC1, m_h2, m_h1.t());
    
    cv::Mat_<float> g2aa = m_g2a.mul(m_g2a); // g2a*
    cv::Mat_<float> g2ab = m_g2a.mul(m_g2b);
    cv::Mat_<float> g2ac = m_g2a.mul(m_g2c);
    cv::Mat_<float> g2bb = m_g2b.mul(m_g2b); // g2b*
    cv::Mat_<float> g2bc = m_g2b.mul(m_g2c);
    cv::Mat_<float> g2cc = m_g2c.mul(m_g2c); // g2c*
    
    cv::Mat_<float> h2aa = m_h2a.mul(m_h2a); // h2a*
    cv::Mat_<float> h2ab = m_h2a.mul(m_h2b);
    cv::Mat_<float> h2ac = m_h2a.mul(m_h2c);
    cv::Mat_<float> h2ad = m_h2a.mul(m_h2d);
    cv::Mat_<float> h2bb = m_h2b.mul(m_h2b); // h2b*
    cv::Mat_<float> h2bc = m_h2b.mul(m_h2c);
    cv::Mat_<float> h2bd = m_h2b.mul(m_h2d);
    cv::Mat_<float> h2cc = m_h2c.mul(m_h2c); // h2c*
    cv::Mat_<float> h2cd = m_h2c.mul(m_h2d);
    cv::Mat_<float> h2dd = m_h2d.mul(m_h2d); // h2d*
    
    // The phase angle of a complex number is the angle the theoretical vector to (real,imag) forms with the real axis (i.e., its arc tangent).
    // It returns the same as: atan2(x.imag(),x.real());
    // OpenCV: cartToPolar(x, y) return atan2(y, x);
    // In the paper theta = 0 is the vertical orientation, and theta increases counterclockwise
    
    m_c1 = 0.5*(g2bb) + 0.25*(g2ac) + 0.375*(g2aa + g2cc) + 0.3125*(h2aa + h2dd) + 0.5625*(h2bb + h2cc) + 0.375*(h2ac + h2bd);
    m_c2 = 0.5*(g2aa - g2cc) + 0.46875*(h2aa - h2dd) + 0.28125*(h2bb - h2cc) + 0.1875*(h2ac - h2bd);
    m_c3 = (-g2ab) - g2bc - (0.9375*(h2cd + h2ab)) - (1.6875*(h2bc)) - (0.1875*(h2ad));
    
    cv::cartToPolar(m_c2, m_c3, m_orientationStrength, m_theta); // dominant orientation angle
    wrap(m_theta, m_theta);
    m_theta *= 0.5;
}

