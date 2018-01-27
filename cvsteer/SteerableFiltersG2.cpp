// Copyright (c) 2013-2018, David Hirvonen
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

#include <cvsteer/SteerableFiltersG2.h>
#include <opencv2/imgproc/imgproc.hpp>

_STEER_BEGIN

typedef cv::Mat1f Matf;

static float G21(float x) { return 0.9213 * (2.0 * x * x - 1.0) * exp(-x * x); }
static float G22(float x) { return exp(-x * x); }
static float G23(float x) { return sqrt(1.8430) * x * exp(-x * x); }

static float H21(float x) { return 0.9780 * (-2.254 * x + x * x * x) * exp(-x * x); }
static float H22(float x) { return exp(-x * x); }
static float H23(float x) { return x * exp(-x * x); }
static float H24(float x) { return 0.9780 * (-0.7515 + x * x) * exp(-x * x); }

SteerableFiltersG2::SteerableFiltersG2(const Matf& image, int width, float spacing)
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

void SteerableFiltersG2::setup(const Matf& image)
{
    cv::sepFilter2D(image, m_g2a, CV_32FC1, m_g1, m_g2.t());
    cv::sepFilter2D(image, m_g2b, CV_32FC1, m_g3, m_g3.t());
    cv::sepFilter2D(image, m_g2c, CV_32FC1, m_g2, m_g1.t());
    cv::sepFilter2D(image, m_h2a, CV_32FC1, m_h1, m_h2.t());
    cv::sepFilter2D(image, m_h2b, CV_32FC1, m_h4, m_h3.t());
    cv::sepFilter2D(image, m_h2c, CV_32FC1, m_h3, m_h4.t());
    cv::sepFilter2D(image, m_h2d, CV_32FC1, m_h2, m_h1.t());

    Matf g2aa = m_g2a.mul(m_g2a); // g2a*
    Matf g2ab = m_g2a.mul(m_g2b);
    Matf g2ac = m_g2a.mul(m_g2c);
    Matf g2bb = m_g2b.mul(m_g2b); // g2b*
    Matf g2bc = m_g2b.mul(m_g2c);
    Matf g2cc = m_g2c.mul(m_g2c); // g2c*

    Matf h2aa = m_h2a.mul(m_h2a); // h2a*
    Matf h2ab = m_h2a.mul(m_h2b);
    Matf h2ac = m_h2a.mul(m_h2c);
    Matf h2ad = m_h2a.mul(m_h2d);
    Matf h2bb = m_h2b.mul(m_h2b); // h2b*
    Matf h2bc = m_h2b.mul(m_h2c);
    Matf h2bd = m_h2b.mul(m_h2d);
    Matf h2cc = m_h2c.mul(m_h2c); // h2c*
    Matf h2cd = m_h2c.mul(m_h2d);
    Matf h2dd = m_h2d.mul(m_h2d); // h2d*

    // The phase angle of a complex number is the angle the theoretical vector to (real,imag) forms with the real axis (i.e., its arc tangent).
    // It returns the same as: atan2(x.imag(),x.real());
    // OpenCV: cartToPolar(x, y) return atan2(y, x);
    // In the paper theta = 0 is the vertical orientation, and theta increases counterclockwise

    m_c1 = 0.5 * (g2bb) + 0.25 * (g2ac) + 0.375 * (g2aa + g2cc) + 0.3125 * (h2aa + h2dd) + 0.5625 * (h2bb + h2cc) + 0.375 * (h2ac + h2bd);
    m_c2 = 0.5 * (g2aa - g2cc) + 0.46875 * (h2aa - h2dd) + 0.28125 * (h2bb - h2cc) + 0.1875 * (h2ac - h2bd);
    m_c3 = (-g2ab) - g2bc - (0.9375 * (h2cd + h2ab)) - (1.6875 * (h2bc)) - (0.1875 * (h2ad));

    cv::cartToPolar(m_c2, m_c3, m_orientationStrength, m_theta); // dominant orientation angle
    wrap(m_theta, m_theta);
    m_theta *= 0.5;
}

//  phase = arg(G2, H2) where arg(x + iy) = atan(y,x), (opencv return angles in [0..2pi])
//  0      = dark line
//  pi     = bright line
// +pi/2   = edge
// -pi/2   = edge
void SteerableFiltersG2::computeMagnitudeAndPhase(const Matf& g2, const Matf& h2, Matf& magnitude, Matf& phase)
{
    cv::cartToPolar(g2, h2, magnitude, phase); // [0..2*piI]
    wrap(phase, phase);                        // [-pi/2 pi/2]
    cv::patchNaNs(phase);
}

// Steer filters at a single pixel:
void SteerableFiltersG2::steer(const cv::Point& p, float theta, float& g2, float& h2)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    float ct(std::cos(theta)), ct2(ct * ct), ct3(ct2 * ct), st(std::sin(theta)), st2(st * st), st3(st2 * st);
    float ga(ct2), gb(-2.0 * ct * st), gc(st2);
    float ha(ct3), hb(-3.0 * ct2 * st), hc(3.0 * ct * st2), hd(-st3);
    g2 = ga * m_g2a(p) + gb * m_g2b(p) + gc * m_g2c(p);
    h2 = ha * m_h2a(p) + hb * m_h2b(p) + hc * m_h2c(p) + hd * m_h2d(p);
}

void SteerableFiltersG2::steer(const cv::Point& p, float theta, float& g2, float& h2, float& e, float& magnitude, float& phase)
{
    steer(p, theta, g2, h2);
    phase = atan2(h2, g2);
    magnitude = sqrt(h2 * h2 + g2 * g2);

    // Compute oriented energy as a function of angle theta
    float c2t(std::cos(theta * 2.0)), s2t(std::sin(theta * 2.0));
    e = m_c1(p) + (c2t * m_c2(p)) + (s2t * m_c3(p));
}

// Steer filters across the entire image:
void SteerableFiltersG2::steer(float theta, Matf& g2, Matf& h2)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    float ct(std::cos(theta)), ct2(ct * ct), ct3(ct2 * ct), st(std::sin(theta)), st2(st * st), st3(st2 * st);
    float ga(ct2), gb(-2.0 * ct * st), gc(st2);
    float ha(ct3), hb(-3.0 * ct2 * st), hc(3.0 * ct * st2), hd(-st3);
    g2 = ga * m_g2a + gb * m_g2b + gc * m_g2c;
    h2 = ha * m_h2a + hb * m_h2b + hc * m_h2c + hd * m_h2d;
}

void SteerableFiltersG2::steer(const Matf& theta, Matf& g2, Matf& h2)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    Matf ct, ct2, ct3, st, st2, st3;
    cv::polarToCart(cv::Mat(), theta, ct, st);
    ct2 = ct.mul(ct), ct3 = ct2.mul(ct), st2 = st.mul(st), st3 = st2.mul(st);
    g2 = ct2.mul(m_g2a) + (-2.0 * ct.mul(st).mul(m_g2b)) + (st2.mul(m_g2c));
    h2 = ct3.mul(m_h2a) + (-3.0 * ct2.mul(st).mul(m_h2b)) + (3.0 * ct.mul(st2).mul(m_h2c)) + (-st3.mul(m_h2d));
}

void SteerableFiltersG2::steer(float theta, Matf& g2, Matf& h2, Matf& e, Matf& magnitude, Matf& phase)
{
    steer(theta, g2, h2);
    computeMagnitudeAndPhase(g2, h2, magnitude, phase);

    // Compute oriented energy as a function of angle theta
    float c2t(std::cos(theta * 2.0)), s2t(std::sin(theta * 2.0));
    e = m_c1 + (c2t * m_c2) + (s2t * m_c3);
}

void SteerableFiltersG2::steer(const Matf& theta, Matf& g2, Matf& h2, Matf& e, Matf& magnitude, Matf& phase)
{
    // Create the steering coefficients, then compute G2 and H2 at orientation theta:
    steer(theta, g2, h2);
    computeMagnitudeAndPhase(g2, h2, magnitude, phase);

    // Compute oriented energy as a function of angle theta
    Matf c2t, s2t;
    cv::polarToCart(cv::Mat(), theta * 2.0, c2t, s2t);
    e = m_c1 + m_c2.mul(c2t) + m_c3.mul(s2t);
}

void SteerableFiltersG2::phaseWeights(const Matf& phase, Matf& lambda, float phi, bool signum, float k)
{
    Matf ct, st, error = cv::abs(signum ? (phase - phi) : (cv::abs(phase) - std::abs(phi)));
    error = cv::min(error, 2.0 * M_PI - error);
    cv::polarToCart(cv::Mat(), error, ct, st);
    lambda = ct.mul(ct);
    lambda.setTo(0, cv::abs(error) > M_PI_2);
}

//  phase = arg(G2, H2) where arg(x + iy) = atan(y,x), (opencv return angles in [0..2pi])
//  0      = dark line
//  pi     = bright line
// +pi/2   = edge
// -pi/2   = edge

static void phaseEdge(const Matf& e, const Matf& phase, Matf& edges, float phi, bool signum, float k)
{
    Matf lambda;
    SteerableFiltersG2::phaseWeights(phase, lambda, phi, signum, k);
    edges = e.mul(lambda);
}

void SteerableFiltersG2::findEdges(const Matf& e, const Matf& phase, Matf& output, float k)
{
    phaseEdge(e, phase, output, M_PI_2, false, k);
}
void SteerableFiltersG2::findDarkLines(const Matf& e, const Matf& phase, Matf& output, float k)
{
    phaseEdge(e, phase, output, 0, true, k);
}
void SteerableFiltersG2::findBrightLines(const Matf& e, const Matf& phase, Matf& output, float k)
{
    phaseEdge(e, phase, output, M_PI, true, k);
}

_STEER_END
