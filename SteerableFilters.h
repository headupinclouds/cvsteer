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

#ifndef __steerable__SteerableFilters__
#define __steerable__SteerableFilters__

#include <opencv2/core/core.hpp>

class SteerableFilters
{
public:
    
    static float G21(float x) { return 0.9213 * (2.0*x*x - 1.0) * exp(-x*x); }
    static float G22(float x) { return exp(-x*x); }
    static float G23(float x) { return sqrt(1.8430) * x * exp(-x*x); }
    
    static float H21(float x) { return 0.9780 * (-2.254 * x + x*x*x) * exp(-x*x);}
    static float H22(float x) { return exp(-x*x);}
    static float H23(float x) { return x * exp(-x*x); }
    static float H24(float x) { return 0.9780 * (-0.7515 + x*x) * exp(-x*x); }
    
    SteerableFilters(const cv::Mat_<float> &image, int width = 4, float spacing = 0.67);

    const cv::Mat_<float> getDominantOrientationAngle() const { return m_theta; }
    const cv::Mat_<float> getDominantOrientationStrength() const { return m_orientationStrength; }
    
    void computeMagnitudeAndPhase(const cv::Mat_<float> &g2, const cv::Mat_<float> &h2, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase);
    void steer(float theta, cv::Mat_<float> &g2, cv::Mat_<float> &h2, cv::Mat_<float> &e, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase);    
    void steer(const cv::Mat_<float> &theta, cv::Mat_<float> &g2, cv::Mat_<float> &h2, cv::Mat_<float> &e, cv::Mat_<float> &magnitude, cv::Mat_<float> &phase);
    void phaseLine(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &lines, float k=2.0);
    void phaseEdge(const cv::Mat_<float> &e, const cv::Mat_<float> &phase, cv::Mat_<float> &edges, float k=2.0);
    void setup(const cv::Mat_<float> &image);
    
protected:
        
    cv::Mat_<float> m_g1, m_g2, m_g3, m_h1, m_h2, m_h3, m_h4;
    cv::Mat_<float> m_g2a, m_g2b, m_g2c, m_h2a, m_h2b, m_h2c, m_h2d;
    cv::Mat_<float> m_c1, m_c2, m_c3, m_theta, m_orientationStrength;
};



#endif /* defined(__steerable__SteerableFilters__) */
