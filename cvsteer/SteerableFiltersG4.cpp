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

#include <cvsteer/SteerableFiltersG4.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

_STEER_BEGIN

static float G41(float x) { return 1.246 * (0.75 - 3.0f * x * x + x * x * x * x) * exp(-x * x); } // f1
static float G42(float x) { return exp(-x * x); }                                                 // f2
static float G43(float x) { return (-1.5 * x + x * x * x) * exp(-x * x); }                        // f3
static float G44(float x) { return 1.246 * x * exp(-x * x); }                                     // f4
static float G45(float x) { return sqrt(1.246) * (x * x - 0.5) * exp(-x * x); }                   // f5

static float H41(float x) { return 0.3975 * (7.189 * x - 7.501 * x * x * x + x * x * x * x * x) * exp(-x * x); } // 1
static float H42(float x) { return exp(-x * x); }                                                                // 2
static float H43(float x) { return 0.3975 * (1.438 - 4.501 * x * x + x * x * x * x) * exp(-x * x); }             // 3
static float H44(float x) { return x * exp(-x * x); }                                                            // 4
static float H45(float x) { return 0.3975 * (x * x * x - 2.225 * x) * exp(-x * x); }                             // 5
static float H46(float x) { return (x * x - 0.6638) * exp(-x * x); }                                             // 6

SteerableFiltersG4::SteerableFiltersG4(const cv::Mat1f& image, int width, float spacing)
{
    // Create separable filters for G4
    m_g1 = create(width, spacing, G41);
    m_g2 = create(width, spacing, G42);
    m_g3 = create(width, spacing, G43);
    m_g4 = create(width, spacing, G44);
    m_g5 = create(width, spacing, G45);

    // Create separable filters for H4
    m_h1 = create(width, spacing, H41);
    m_h2 = create(width, spacing, H42);
    m_h3 = create(width, spacing, H43);
    m_h4 = create(width, spacing, H44);
    m_h5 = create(width, spacing, H45);
    m_h6 = create(width, spacing, H46);

    setup(image);
}

void SteerableFiltersG4::setup(const cv::Mat1f& image)
{
    cv::sepFilter2D(image, m_g4a, CV_32FC1, m_g1, m_g2.t());
    cv::sepFilter2D(image, m_g4b, CV_32FC1, m_g3, m_g4.t());
    cv::sepFilter2D(image, m_g4c, CV_32FC1, m_g5, m_g5.t());
    cv::sepFilter2D(image, m_g4d, CV_32FC1, m_g4, m_g3.t());
    cv::sepFilter2D(image, m_g4e, CV_32FC1, m_g2, m_g1.t());

    cv::sepFilter2D(image, m_h4a, CV_32FC1, m_h1, m_h2.t());
    cv::sepFilter2D(image, m_h4b, CV_32FC1, m_h3, m_h4.t());
    cv::sepFilter2D(image, m_h4c, CV_32FC1, m_h5, m_h6.t());
    cv::sepFilter2D(image, m_h4d, CV_32FC1, m_h6, m_h5.t());
    cv::sepFilter2D(image, m_h4e, CV_32FC1, m_h4, m_h3.t());
    cv::sepFilter2D(image, m_h4f, CV_32FC1, m_h2, m_h1.t());
}

//  phase = arg(G2, H2) where arg(x + iy) = atan(y,x), (opencv return angles in [0..2pi])
//  0      = dark line
//  pi     = bright line
// +pi/2   = edge
// -pi/2   = edge
void SteerableFiltersG4::computeMagnitudeAndPhase(const cv::Mat1f& g2, const cv::Mat1f& h2, cv::Mat1f& magnitude, cv::Mat1f& phase)
{
}

void SteerableFiltersG4::steer(const cv::Mat1f& theta, cv::Mat1f& g4, cv::Mat1f& h4)
{
    cv::Mat1f ct, ct2, ct3, ct4, ct5;
    cv::Mat1f st, st2, st3, st4, st5;

    cv::polarToCart(cv::Mat(), theta, ct, st);
    ct2 = ct.mul(ct);
    ct3 = ct2.mul(ct);
    ct4 = ct3.mul(ct);
    ct5 = ct4.mul(ct);

    st2 = st.mul(st);
    st3 = st2.mul(st);
    st4 = st3.mul(st);
    st5 = st4.mul(st);

    cv::Mat1f ga(ct4), gb(-4.0 * ct3.mul(st)), gc(6.0 * ct2.mul(st2)), gd(-4.0 * ct.mul(st3)), ge(st4);
    cv::Mat1f ha(ct5), hb(-5.0f * ct4.mul(st)), hc(10.0 * ct3.mul(st2)), hd(-10.0 * ct2.mul(st3)), he(5.0 * ct.mul(st4)), hf(-st5);
    g4 = ga.mul(m_g4a) + gb.mul(m_g4b) + gc.mul(m_g4c) + gd.mul(m_g4d) + ge.mul(m_g4e);
    h4 = ha.mul(m_h4a) + hb.mul(m_h4b) + hc.mul(m_h4c) + hd.mul(m_h4d) + he.mul(m_h4e) + hf.mul(m_h4f);
}

void SteerableFiltersG4::steer(float theta, cv::Mat1f& g4, cv::Mat1f& h4)
{
    float ct(std::cos(theta)), ct2(ct * ct), ct3(ct2 * ct), ct4(ct3 * ct), ct5(ct4 * ct);
    float st(std::sin(theta)), st2(st * st), st3(st2 * st), st4(st3 * st), st5(st4 * st);
    float ga(ct4), gb(-4.0 * ct3 * st), gc(6.0 * ct2 * st2), gd(-4.0 * ct * st3), ge(st4);
    float ha(ct5), hb(-5.0f * ct4 * st), hc(10.0 * ct3 * st2), hd(-10.0 * ct2 * st3), he(5.0 * ct * st4), hf(-st5);
    g4 = ga * m_g4a + gb * m_g4b + gc * m_g4c + gd * m_g4d + ge * m_g4e;
    h4 = ha * m_h4a + hb * m_h4b + hc * m_h4c + hd * m_h4d + he * m_h4e + hf * m_h4f;
}

_STEER_END
