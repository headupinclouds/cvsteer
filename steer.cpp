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

#include "SteerableFiltersG2.h"
#include "SteerableFiltersG4.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iostream>

void CannyRawInput( cv::Mat &dst, cv::Mat_<short> &dx, cv::Mat_<short> &dy, double low_thresh, double high_thresh, bool L2gradient );

typedef std::vector<cv::Point2f> PointSetf;

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

//nd = ndims (x);
//sz = size (x);
//sz2 = floor (sz ./ 2);
//idx = cell ();
//for i = 1:nd
//idx{i} = [sz2(i)+1:sz(i), 1:sz2(i)];
//endfor
//retval = x(idx{:});

template <typename T>
void ifftshift(cv::Mat_<T> &out)
{
    cv::Mat_<T> tmp = out.clone();
    int cx = out.cols/2; // i.e., floor(x/2)
    int cy = out.rows/2;
    
    std::vector<int> xvals(out.cols, 0), yvals(out.rows, 0);
    
    iota(yvals.begin(), yvals.begin()+(out.rows-cy), cy);
    iota(yvals.begin()+(out.rows-cy), yvals.end(), 0);
    
    iota(xvals.begin(), xvals.begin()+(out.cols-cx), cx);
    iota(xvals.begin()+(out.cols-cx), xvals.end(), 0);
    
    //for(int i = 0; i < xvals.size(); i++) std::cout << xvals[i] << " "; std::cout << std::endl;
    //for(int i = 0; i < yvals.size(); i++) std::cout << yvals[i] << " "; std::cout << std::endl;
    
    for(int y = 0; y < out.rows; y++)
        for(int x = 0; x < out.cols; x++)
            out(y,x) = tmp(yvals[y], xvals[x]);
}

//nd = ndims (x);
//sz = size (x);
//sz2 = ceil (sz ./ 2);
//idx = cell ();
//for i = 1:nd
//idx{i} = [sz2(i)+1:sz(i), 1:sz2(i)];
//endfor
//retval = x(idx{:});

template <typename T>
void fftshift(cv::Mat_<T> &out)
{
    cv::Mat_<T> tmp = out.clone();
    int cx = (out.cols+1)/2;  // i.e., ceil(x/2)
    int cy = (out.rows+1)/2;
    std::vector<int> xvals(out.cols), yvals(out.rows);
    
    iota(yvals.begin(), yvals.begin()+(out.rows-cy), cy);
    iota(yvals.begin()+(out.rows-cy), yvals.end(), 0);
    
    iota(xvals.begin(), xvals.begin()+(out.cols-cx), cx);
    iota(xvals.begin()+(out.cols-cx), xvals.end(), 0);
    
    //for(int i = 0; i < xvals.size(); i++) std::cout << xvals[i] << " "; std::cout << std::endl;
    //for(int i = 0; i < yvals.size(); i++) std::cout << yvals[i] << " "; std::cout << std::endl;
    
    for(int y = 0; y < out.rows; y++)
        for(int x = 0; x < out.cols; x++)
            out(y,x) = tmp(yvals[y], xvals[x]);
    
}

inline std::pair<double,double> filterRange(int span)
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

#include <opencv2/imgproc/imgproc.hpp>


static void frankotchellapa(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output)
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
    
    ifftshift(wx);
    ifftshift(wy);
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


    
    //{ std::ofstream ox("/tmp/wx.txt"); for(int y=0;y<wx.rows;y++){for(int x=0;x<wx.cols;x++) ox << wx.at<float>(y,x) << " "; ox << "\n";} }
    //{ std::ofstream ox("/tmp/wy.txt"); for(int y=0;y<wx.rows;y++){for(int x=0;x<wx.cols;x++) ox << wy.at<float>(y,x) << " "; ox << "\n";} }
    //{ std::ofstream ox("/tmp/z.txt"); for(int y=0;y<z.rows;y++){for(int x=0;x<z.cols;x++) ox << z.at<float>(y,x) << " "; ox << "\n";} }
    
    cv::idft(z, output, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE);
}

static void frankotchellapa(const cv::Mat_<cv::Vec2f> &gradients, cv::Mat_<float> &output)
{
    cv::Mat_<float> dx, dy;
    std::vector<cv::Mat_<float> > channels;
    cv::split(gradients, channels);
    dx = channels[0].clone();
    dy = channels[1].clone();
    frankotchellapa(dx, dy, output);
}

static void reduce(const cv::Mat &image, int op, cv::Mat &output)
{
    output = image.reshape(1, image.size().area());
    cv::reduce(output, output, 1, op);
    output = output.reshape(1, image.rows);
}

static cv::Mat dot(const cv::Mat &src0, const cv::Mat &src1)
{
    cv::Mat result = src0.mul(src1);
    result = result.reshape(1, src0.size().area());
    cv::reduce(result, result, 1, CV_REDUCE_SUM, CV_32FC1);
    return result.reshape(1, src0.rows);
}

static void MaxSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, int kernel, float scale)
{
    cv::Mat dx_, dy_;
    cv::Sobel(image, dx_, CV_32F, 1, 0, kernel, scale);
    cv::Sobel(image, dy_, CV_32F, 0, 1, kernel, scale);
    reduce(dx_, CV_REDUCE_MAX, dx);
    reduce(dy_, CV_REDUCE_MAX, dy);
}


static void ColorSobel(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy, cv::Mat &magnitude, int kernel, float scale)
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
static cv::Mat quiver(const cv::Mat &image, const cv::Mat_<cv::Vec2f> &orientation, int yTic, int xTic, float scale)
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
static cv::Mat quiver(const cv::Mat &image, const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, int yTic, int xTic, float scale)
{
    cv::Mat_<cv::Vec2f> v;
    cv::Mat tmp[2] = { dx, dy };
    cv::merge(tmp, 2, v);
    return quiver(image, v, yTic, xTic, scale);
}

static void show(const std::string &name, const cv::Mat_<float> &image, bool norm)
{
    cv::Mat_<float> tmp = image;
    if(norm == true)
        cv::normalize(image, tmp, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, tmp);
    cv::waitKey(0);
}

// Return the percentile corresponding to the specified percentile rank
float findPercentile(const cv::Mat_<float> &image, float rank = 99.0f, int bins = 100)
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

// BASH style basename function with STL string
static std::string basename(const std::string &name)
{
	size_t pos = name.rfind("/") + 1;
	std::string base = name.substr((pos < name.size()) * pos);
	return base.substr(0, std::min(base.size()-1, base.rfind(".")));
};

static cv::Mat squash(const cv::Mat &image, float n, float percent)
{
    cv::Scalar mu, sigma;
    cv::meanStdDev(image, mu, sigma);
    cv::Mat result = cv::min(image, mu[0] + sigma[0]);
    return cv::min(result, findPercentile(result, 95, 100));
}

#include <numeric>
#include "SteerableFiltersG4.h"

class ParallelSteerable : public cv::ParallelLoopBody
{
public:
    ParallelSteerable(const std::vector<std::string> &filenames, std::vector<cv::Mat> &images, const std::string &directory )
    : m_filenames(filenames)
    , m_images(images)
    , m_directory(directory)
    , m_percentileRank(95.0)
    , m_doLogging(true) {}
    
    void setDoLogging(bool flag) { m_doLogging = flag; }
    void setPercentileRank(float rank) { m_percentileRank = rank; }
    
    virtual void operator()( const cv::Range &r ) const
    {
        for (int i = r.start; i != r.end; i++)
        {
            cv::Mat image = cv::imread(m_filenames[i]), gray;
            if(image.empty())
                continue;
            
            // cv::pyrDown(image, image);
            cv::resize(image, image, cv::Size(100, 100*image.rows/image.cols));
            
            cv::medianBlur(image, image, 3);
            
            if(image.channels() != 1)
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
           
            cv::Mat dx, dy, m, m2, o;
            cv::Mat_<float> z;
            //MaxSobel(image, dx, dy, 3, 1.0);
            ColorSobel(image, dx, dy, m, 7, 1.0);
            m2 = squash(m, 3.0, 75);
            dx = dx.mul(m2);
            dy = dy.mul(m2);
            frankotchellapa(dx, dy, z);
            
           

            cv::Mat_<float> g2, h2, e, magnitude, phase, edges, lines0, lines1;
            SteerableFilters filters(z, 4, 0.67);
            filters.steer(filters.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
            filters.phaseEdge(e, phase, edges, 2.0);
            filters.phaseLine0(e, phase, lines0, 2.0);
            filters.phaseLine1(e, phase, lines1, 2.0);
            
#if 0
            SteerableFiltersG4 filters4(z, 6, 0.5);
            for(int t = 0; ; t++)
            {
                cv::Mat_<float> g4, h4, m4, m2;
                filters4.steer(t * M_PI/180.0, g4, h4);
                cv::magnitude(g4, h4, m4);
                cv::normalize(m4, m4, 0, 1, cv::NORM_MINMAX, CV_32FC1);
                
                filters.steer(t * M_PI/180.0, g2, h2);
                cv::magnitude(g2, h2, m2);
                cv::normalize(m2, m2, 0, 1, cv::NORM_MINMAX, CV_32FC1);
                
                cv::hconcat(m4, m2, m4);
                cv::imshow("m4", m4), cv::waitKey(40);
            }
#endif
            


#if 0
            cv::Mat_<short> dx_, dy_, output;
            cv::polarToCart(cv::Mat(), filters.getDominantOrientationAngle(), dx, dy);
            dx = edges.mul(dx);
            dy = edges.mul(dy);

            
            double maxVal = 0;
            cv::minMaxLoc(cv::abs(edges), 0, &maxVal);
            float top = float(std::numeric_limits<short>::max());
            dx.convertTo(dx_, CV_16S, top/maxVal);
            dy.convertTo(dy_, CV_16S, top/maxVal);
            CannyRawInput( output, dx_, dy_, top/96.0, top/32.0, true ); // top/64=511.98
            cv::imshow("out", output), cv::waitKey(0);
#endif
            
            // cv::Mat p2 = cv::abs(phase) * (1.0/M_PI); cv::imshow("phase", p2), cv::waitKey(0);
            
            // Now display the computed results:
            std::vector<cv::Mat> images;
            cv::Mat canvas;;
            
            cv::normalize(gray, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            cv::normalize(z, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            //cv::equalizeHist(canvas, canvas);
            images.push_back(canvas.clone());
            
            // Normalize the g2 h2 images together:
            cv::hconcat(g2, h2, g2);
            g2 = squash(g2, 2.0, m_percentileRank);
            cv::normalize(g2, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            // Add the dominant orientation energy:
            e = squash(e, 2.0, m_percentileRank);
            cv::normalize(e, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            // Normalize the phase-edge and phase-line images together to compare magnitude:
            cv::Mat channels[3] = { edges, lines0, lines1 };
            cv::hconcat(channels, 3, edges);
            edges = squash(edges, 2.0, m_percentileRank);
            cv::normalize(edges, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            
            images.push_back(canvas.clone());
        
     
            //cv::Mat arrows = quiver(image, -dx, dy, 3, 3, 10.0);
            //cv::imshow("a", arrows), cv::waitKey(0);
            
            cv::normalize(m, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            
            cv::hconcat(images, canvas);  // cv::imshow("g2h2", canvas), cv::waitKey(0);
        
            m_images[i] = canvas;
        
            if(!m_images[i].empty())
            {
                std::string filename = m_directory + "/" + basename(m_filenames[i]) + "_g2h2.png";
                cv::imwrite(filename, m_images[i]);
            }
        }
    }
    
protected:
    
    bool m_doLogging;
    float m_percentileRank;
    const std::string &m_directory;
    const std::vector<std::string> &m_filenames;
    std::vector<cv::Mat> &m_images;
};


// For use with istream_iterator to read complete lines (new line delimiter)
// std::vector<std::string> lines;
// std::copy(std::istream_iterator<line>(std::cin), std::istream_iterator<line>(), std::back_inserter(lines));
//http://stackoverflow.com/questions/1567082/how-do-i-iterate-over-cin-line-by-line-in-c/1567703#1567703
struct Line
{
    std::string data;
    friend std::istream &operator>>(std::istream &is, Line &l)
    {
        std::getline(is, l.data);
        return is;
    }
    operator std::string() const { return data; }
};


int main(int argc, const char * argv[])
{
    std::string filename, directory;
    if(argc < 2)
    {
        std::cout << "usage: " << argv[0] << "<filenames> <output>" << std::endl;
    }
    else
    {
        filename = argv[1];
        directory = (argc >= 3) ? argv[2] : "/tmp/";
    }
    
    std::vector<std::string> filenames;
    if( (filename.rfind(".txt") != std::string::npos) || (filename.rfind(".") == std::string::npos) )
    {
        std::ifstream file(filename.c_str());
        std::copy(std::istream_iterator<Line>(file), std::istream_iterator<Line>(), std::back_inserter(filenames));
    }
    else
    {
        filenames.push_back(filename);
    }
    
    std::vector<cv::Mat> images(filenames.size());
    ParallelSteerable body(filenames, images, directory);
    
    body( cv::Range(0, filenames.size() ) );
    //cv::parallel_for_(cv::Range(0, static_cast<int>(filenames.size())), body);
    
    //cv::Mat canvas;
    //cv::vconcat(images, canvas);
    //cv::namedWindow("g2h2", CV_WINDOW_NORMAL), cv::imshow("g2h2", canvas), cv::waitKey(0);
    
    return 0;
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

