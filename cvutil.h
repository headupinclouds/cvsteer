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

#ifndef __steerable__cvutil_
#define __steerable__cvutil_

#include <opencv2/core/core.hpp>

// include iota (in our own namespace) so it works with old and new compilers
template<class ForwardIterator, class T>
void iota(ForwardIterator first, ForwardIterator last, T value)
{
    while(first != last) {
        *first++ = value;
        ++value;
    }
}

// http://stackoverflow.com/questions/5735720/effcient-way-to-do-fft-shift-in-matlab-without-using-fftshift-function/5740442#5740442
//sz = ceil(size(A)/2)
//A = A([sz(1)+1:end, 1:sz(1)], [sz(2)+1:end, 1:sz(2)])

template <typename T>
cv::Mat_<T> ifftshift(const cv::Mat_<T> &src)
{
    cv::Mat_<T> tmp = src.clone();
    int cx = src.cols/2, cy = src.rows/2; // i.e., floor(x/2)
    std::vector<int> xvals(src.cols, 0), yvals(src.rows, 0);
    iota(yvals.begin(), yvals.begin()+(src.rows-cy), cy);
    iota(yvals.begin()+(src.rows-cy), yvals.end(), 0);
    iota(xvals.begin(), xvals.begin()+(src.cols-cx), cx);
    iota(xvals.begin()+(src.cols-cx), xvals.end(), 0);
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            
            tmp(y,x) = src(yvals[y], xvals[x]);
    return tmp;
}

template <typename T>
cv::Mat_<T> fftshift(const cv::Mat_<T> &src)
{
    cv::Mat_<T> tmp = src.clone();
    int cx = (src.cols+1)/2, cy = (src.rows+1)/2; // i.e., ceil(x/2)
    std::vector<int> xvals(src.cols), yvals(src.rows);
    iota(yvals.begin(), yvals.begin()+(src.rows-cy), cy);
    iota(yvals.begin()+(src.rows-cy), yvals.end(), 0);
    iota(xvals.begin(), xvals.begin()+(src.cols-cx), cx);
    iota(xvals.begin()+(src.cols-cx), xvals.end(), 0);
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            tmp(y,x) = src(yvals[y], xvals[x]);
    
    return tmp;
}

void frankotchellapa(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output);
void frankotchellapa(const cv::Mat_<cv::Vec2f> &gradients, cv::Mat_<float> &output);
void reduce(const cv::Mat &image, int op, cv::Mat &output);
cv::Mat dot(const cv::Mat &src0, const cv::Mat &src1);
void MaxSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, int kernel, float scale);
void ColorSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, cv::Mat &magnitude, int kernel, float scale);

// A poor man's matlab quiver display, via upsampling and anti-aliased line drawing
cv::Mat quiver(const cv::Mat &image, const cv::Mat_<cv::Vec2f> &orientation, int yTic, int xTic, float scale);
cv::Mat quiver(const cv::Mat &image, const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, int yTic, int xTic, float scale);
void show(const std::string &name, const cv::Mat_<float> &image, bool norm);
float findPercentile(const cv::Mat_<float> &image, float rank = 99.0f, int bins = 100);
void CannyRawInput( cv::Mat &dst, cv::Mat_<short> &dx, cv::Mat_<short> &dy, double low_thresh, double high_thresh, bool L2gradient );

#endif