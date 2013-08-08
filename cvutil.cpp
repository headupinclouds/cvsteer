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

#include "cvutil.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef std::vector<cv::Point2f> PointSetf;

static std::pair<double,double> filterRange(int span)
{
    double cols = span;
    std::pair<double,double> range;
    if(span == 1)
        range = std::make_pair(0, 0);
    else if(span % 2)
        range = std::make_pair( (-(cols-1.0)/2.0)/(cols-1.0), 1.0/(cols-1.0) );
    else
        range = std::make_pair( (-cols/2.0)/cols, 1.0/cols );
    
    return range;
}

void frankotchellapa(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output)
{
    cv::Size size = dx.size();
    cv::Mat_<float> wx(size, 0.f), wy(size, 0.f);
    std::pair<double,double> xrange = filterRange(size.width);
    std::pair<double,double> yrange = filterRange(size.height);
    double x, y = yrange.first;
    for(int i = 0; i < size.height; i++, y+=yrange.second)
    {
        x = xrange.first;
        for(int j = 0; j < size.width; j++, x+=xrange.second)
        {
            wx(i,j) = x;
            wy(i,j) = y;
        }
    }
    
    cv::Mat_<float> h;
    cv::createHanningWindow(h, size, CV_32FC1);
    
    wx = ifftshift(wx);
    wy = ifftshift(wy);
    cv::Mat d = wx.mul(wx) + wy.mul(wy);
    
    cv::Mat dx_, DX, dy_, DY, wx_, wy_, d_, z;
    {
        cv::Mat planes[2] = { dx.mul(h), cv::Mat_<float>(size, 0.f) };
        cv::merge(planes, 2, dx_);
        cv::dft(dx_, dx_, cv::DFT_COMPLEX_OUTPUT);
    }
    {
        cv::Mat planes[2] = { dy.mul(h), cv::Mat_<float>(size, 0.f) };
        cv::merge(planes, 2, dy_);
        cv::dft(dy_, dy_, cv::DFT_COMPLEX_OUTPUT);
    }
    {
        cv::Mat planes[2] = { cv::Mat_<float>(size, 0.f), -wx };
        cv::merge(planes, 2, wx_);
    }
    {
        cv::Mat planes[2] = { cv::Mat_<float>(size, 0.f), -wy };
        cv::merge(planes, 2, wy_);
    }
    {
        cv::Mat planes[2] = {d, d};
        cv::merge(planes, 2, d_);
    }
    
    cv::mulSpectrums(wx_, dx_, dx_, cv::DFT_COMPLEX_OUTPUT, false);
    cv::mulSpectrums(wy_, dy_, dy_, cv::DFT_COMPLEX_OUTPUT, false);
    
    cv::Mat response = (dx_ + dy_);
    cv::divide(response, (d_ + 1e-10f), z);
    
    cv::idft(z, output, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE);
}

void frankotchellapa(const cv::Mat_<cv::Vec2f> &gradients, cv::Mat_<float> &output)
{
    cv::Mat_<float> dx, dy;
    std::vector<cv::Mat_<float> > channels;
    cv::split(gradients, channels);
    dx = channels[0].clone();
    dy = channels[1].clone();
    frankotchellapa(dx, dy, output);
}

void reduce(const cv::Mat &image, int op, cv::Mat &output)
{
    output = image.reshape(1, image.size().area());
    cv::reduce(output, output, 1, op);
    output = output.reshape(1, image.rows);
}

cv::Mat dot(const cv::Mat &src0, const cv::Mat &src1)
{
    cv::Mat result = src0.mul(src1);
    result = result.reshape(1, src0.size().area());
    cv::reduce(result, result, 1, CV_REDUCE_SUM, CV_32FC1);
    return result.reshape(1, src0.rows);
}

void MaxSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, int kernel, float scale)
{
    cv::Mat dx_, dy_;
    cv::Sobel(image, dx_, CV_32F, 1, 0, kernel, scale);
    cv::Sobel(image, dy_, CV_32F, 0, 1, kernel, scale);
    reduce(dx_, CV_REDUCE_MAX, dx);
    reduce(dy_, CV_REDUCE_MAX, dy);
}

void ColorSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, cv::Mat &magnitude, int kernel, float scale)
{
    cv::Mat luv;
    cv::cvtColor(image, luv, cv::COLOR_BGR2Luv);
    
    cv::Mat dx_, dy_, dxx, dxy, dyy, lambda1, lambda2, D;
    cv::Sobel(luv, dx_, CV_32F, 1, 0, kernel, scale);
    cv::Sobel(luv, dy_, CV_32F, 0, 1, kernel, scale);
    dxx = dot(dx_, dx_);
    dxy = dot(dx_, dy_);
    dyy = dot(dy_, dy_);
    
    cv::GaussianBlur(dxx, dxx, cv::Size(3, 3), 1.0);
    cv::GaussianBlur(dxy, dxy, cv::Size(3, 3), 1.0);
    cv::GaussianBlur(dyy, dyy, cv::Size(3, 3), 1.0);
    
    // Compute the eigenvalues:
    cv::Mat a = (dxx-dyy), b = (2.0 * dxy);
    cv::sqrt(a.mul(a) + b.mul(b), D);
    cv::Mat tmp = dxx + dyy;
    lambda1 = 0.5 * (tmp + D);
    lambda2 = 0.5 * (tmp - D);
    
    // Compute eigenvectors:
    dy = (dyy - dxx + D);
    dx = (2.0 * dxy);
    cv::magnitude(dx, dy, magnitude);
    dx /= (magnitude + 1e-6f);
    dy /= (magnitude + 1e-6f);
    
    //cv::sqrt(lambda1.mul(lambda1) - lambda2.mul(lambda2), magnitude);
    cv::magnitude(lambda1, lambda2, magnitude);
    
    // Now recover sign by comparing with response in each channel
    cv::Sobel(image, dx_, CV_32F, 1, 0, kernel, scale);
    cv::Sobel(image, dy_, CV_32F, 0, 1, kernel, scale);
    
    cv::Mat error[3], dxs[3], dys[3], s;
    cv::split(dx_, dxs);
    cv::split(dy_, dys);
    for(int i = 0; i < 3; i++)
    {
        cv::magnitude(dxs[i], dys[i], s);
        dxs[i] /= (s + 1e-6f);
        dys[i] /= (s + 1e-6f);
        error[i] = 1.0 - (dxs[i].mul(dx) + dys[i].mul(dy));
    }
    cv::Mat mu = (error[0] + error[1] + error[2]) * 0.3333;
    s = 1.0;
    s.setTo(-1.0, (mu > M_PI_2));
    dx = dx.mul(s);
    dy = dy.mul(s);
    
};

// A poor man's matlab quiver display, via upsampling and anti-aliased line drawing
cv::Mat quiver(const cv::Mat &image, const cv::Mat_<cv::Vec2f> &orientation, int yTic, int xTic, float scale)
{
    cv::Mat canvas = image.clone();
    if(image.channels() == 1)
        cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
    
    cv::resize(canvas, canvas, cv::Size(canvas.cols*scale, canvas.rows*scale));
    
    for(int y = 0; y < orientation.rows; y+= yTic)
    {
        for(int x = 0; x < orientation.cols; x+= xTic)
        {
            cv::Point2f p(x, y);
            cv::Vec2f v = orientation(y, x);
            double d = cv::norm(v);
            if(d > 1e-6f)
            {
                p *= scale;
                v *= scale;
                cv::circle(canvas, p, 1, CV_RGB(0,255,0), -1);
                cv::line(canvas, p, p+cv::Point2f(v[0], v[1]), CV_RGB(255,0,0), 1, CV_AA);
            }
        }
    }
    
    return canvas;
}

cv::Mat quiver(const cv::Mat &image, const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, int yTic, int xTic, float scale)
{
    cv::Mat_<cv::Vec2f> v;
    cv::Mat tmp[2] = { dx, dy };
    cv::merge(tmp, 2, v);
    return quiver(image, v, yTic, xTic, scale);
}

void show(const std::string &name, const cv::Mat_<float> &image, bool norm)
{
    cv::Mat_<float> tmp = image;
    if(norm == true)
        cv::normalize(image, tmp, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, tmp);
    cv::waitKey(0);
}

// Return the percentile corresponding to the specified percentile rank
float findPercentile(const cv::Mat_<float> &image, float rank, int bins)
{
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    float probability = std::max(std::min(rank, 0.f), 100.0f) / 100.0f; // normalize percentile
    
    // build a histogram:
    const float min = minVal, max = maxVal;
    cv::Mat hist;
    float range[] = { min, max };
    float bin = (range[1] - range[0]) / float(bins);
    const float * ranges[] = { range };
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &bins, ranges, true, false);
    
    float threshold = max;
    float count = cv::sum(hist)[0], cutoff = (probability * count), tally = 0;
    for(int i = 0; i < bins; i++)
    {
        tally += hist.at<float>(i);
        if(tally > cutoff)
        {
            threshold = minVal + float(i) * bin;
            break;
        }
    }
    return threshold;
}

void CannyRawInput( cv::Mat &dst, cv::Mat_<short> &dx, cv::Mat_<short> &dy, double low_thresh, double high_thresh, bool L2gradient )
{
    dst.create(dx.size(), CV_8U);
    
    cv::Mat src = dst; // used for dimensions, fix code that broke w/ copy and paste.
    const int cn = 1;
    
    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);
    
    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);
        
        if (low_thresh > 0) low_thresh *= low_thresh;
        if (high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);
    
    ptrdiff_t mapstep = src.cols + 2;
    cv::AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2) + cn * mapstep * 3 * sizeof(int));
    
    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));
    
    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);
    
    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];
    
    /* sector numbers
     (Top-Left Origin)
     
     1   2   3
     *  *  *
     * * *
     0*******0
     * * *
     *  *  *
     3   2   1
     */
    
#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top
    
    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);
            
            if (!L2gradient)
            {
                for (int j = 0; j < src.cols*cn; j++)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                for (int j = 0; j < src.cols*cn; j++)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }
            
            if (cn > 1)
            {
                for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));
        
        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;
        
        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;
        
        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];
        
        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);
        
        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }
        
        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
#define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);
            
            int m = _mag[j];
            
            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;
                
                int tg22x = x * TG22;
                
                if (y < tg22x)
                {
                    if (m > _mag[j-1] && m >= _mag[j+1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
        __ocv_canny_push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }
        
        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }
    
    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }
        
        CANNY_POP(m);
        
        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }
    
    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}


