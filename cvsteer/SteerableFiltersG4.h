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

#ifndef __cvsteer_cvsteer_SteerableFiltersG4_h__
#define __cvsteer_cvsteer_SteerableFiltersG4_h__

#include <cvsteer/SteerableFilters.h>

_STEER_BEGIN

class SteerableFiltersG4 : public SteerableFilters
{
public:
    SteerableFiltersG4(const cv::Mat1f& image, int width = 6, float spacing = 0.5);

    const cv::Mat1f& getDominantOrientationAngle() const { return m_theta; }
    const cv::Mat1f& getDominantOrientationStrength() const { return m_orientationStrength; }

    void setup(const cv::Mat1f& image);

    // Processing on entire images:
    void steer(const cv::Mat1f& theta, cv::Mat1f& g4, cv::Mat1f& h4);
    void steer(float theta, cv::Mat1f& g4, cv::Mat1f& h4);
    void computeMagnitudeAndPhase(const cv::Mat1f& g4, const cv::Mat1f& h4, cv::Mat1f& magnitude, cv::Mat1f& phase);

protected:
    cv::Mat1f m_g1, m_g2, m_g3, m_g4, m_g5;
    cv::Mat1f m_h1, m_h2, m_h3, m_h4, m_h5, m_h6;
    cv::Mat1f m_g4a, m_g4b, m_g4c, m_g4d, m_g4e;
    cv::Mat1f m_h4a, m_h4b, m_h4c, m_h4d, m_h4e, m_h4f;
    cv::Mat1f m_c1, m_c2, m_c3, m_theta, m_orientationStrength;
};

_STEER_END

#endif
