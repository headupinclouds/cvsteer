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

#include <gtest/gtest.h>

#include <cvsteer/SteerableFiltersG2.h>
#include <cvsteer/SteerableFiltersG4.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iostream>
#include <cstdint>

#include "Pterois_volitans_Manado-e_edit_smallest.h" // unsigned char Pterois_volitans_Manado_e_edit_smallest_gray_jpg[]
#include "edges.h"                                   // unsigned char edges_jpeg[] 
#include "linesDark.h"                               // unsigned char linesDark_jpeg[]
#include "linesBright.h"                             // unsigned char linesBright_jpeg[]

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// NOTE: xxd -i MyFile.bin > MyFile.h
cv::Mat readJpeg(unsigned char *bytes, unsigned int length)
{
    return cv::imdecode(std::vector<std::uint8_t>(bytes, bytes + length), cv::IMREAD_GRAYSCALE);
}

// Reproduce compression errors for binary comparisons.
// This is admittedly a little wonky, but will serve as
// a workable place holder to support "comiled" image
// byte arrays of reasonable size using JPEG compression
// until some workable alternative can be provided using
// HDF5 or similar.
cv::Mat recode(const cv::Mat &input)
{
    std::vector<std::uint8_t> buffer;
    cv::imencode(".jpg", input, buffer);
    return cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
}
TEST(cvsteer, basic)
{

    cv::Mat fish = readJpeg(Pterois_volitans_Manado_e_edit_smallest_gray_jpg, Pterois_volitans_Manado_e_edit_smallest_gray_jpg_len); 
    cv::Mat edgesGT = readJpeg(edges_jpeg, edges_jpeg_len);
    cv::Mat linesDarkGT = readJpeg(linesDark_jpeg, linesDark_jpeg_len);
    cv::Mat linesBrightGT = readJpeg(linesBright_jpeg, linesBright_jpeg_len);
    
    ASSERT_EQ(fish.channels(), 1);
    ASSERT_EQ(edgesGT.channels(), 1);
    ASSERT_EQ(linesDarkGT.channels(), 1);
    ASSERT_EQ(linesBrightGT.channels(), 1);
    
    // steerable filters
    cv::Mat1f g2, h2, e, magnitude, phase, edges, linesDark, linesBright;
    fa::SteerableFiltersG2 filters2(fish, 4, 0.67f);
    filters2.steer(filters2.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);

    filters2.findEdges(magnitude, phase, edges);
    filters2.findDarkLines(magnitude, phase, linesDark);
    filters2.findBrightLines(magnitude, phase, linesBright);
    
    cv::Mat1b edges8u(fish.size()), linesDark8u(fish.size()), linesBright8u(fish.size());
    cv::normalize(edges, edges8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(linesDark, linesDark8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(linesBright, linesBright8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    auto edgesError = cv::norm(recode(edges8u), edgesGT, cv::NORM_L1) / static_cast<float>(fish.total());
    auto linesDarkError = cv::norm(recode(linesDark8u), linesDarkGT, cv::NORM_L1) / static_cast<float>(fish.total());
    auto linesBrightError = cv::norm(recode(linesBright8u), linesBrightGT, cv::NORM_L1) / static_cast<float>(fish.total());
    
    ASSERT_LE(edgesError, 1.0);
    ASSERT_LE(linesDarkError, 1.0);
    ASSERT_LE(linesBrightError, 1.0);
    
    //cv::imwrite("/tmp/edges.jpeg", edges8u);
    //cv::imwrite("/tmp/linesDark.jpeg", linesDark8u);
    //cv::imwrite("/tmp/linesBright.jpeg", linesBright8u);
}

