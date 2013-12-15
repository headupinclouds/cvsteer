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
#define DO_DRISHTI_TIME 1

#include "SteerableFiltersG2.h"
#include "SteerableFiltersG4.h"
#include "cvutil.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iostream>
#include "unwrap.h"
#include "timing.h"

/////////////////

#if 0
function [ theta_x, theta_y, phi_x, phi_y, r_x, r_y, intensityL2] = spherical_der(input_im, sigma)

%split color channels
R=double(input_im(:,:,1));
G=double(input_im(:,:,2));
B=double(input_im(:,:,3));

% computation of spatial derivatives
Rx=gDer(R,sigma,1,0);
Ry=gDer(R,sigma,0,1);
R =gDer(R,sigma,0,0);

Gx=gDer(G,sigma,1,0);
Gy=gDer(G,sigma,0,1);
G =gDer(G,sigma,0,0);

Bx=gDer(B,sigma,1,0);
By=gDer(B,sigma,0,1);
B =gDer(B,sigma,0,0);

intensityL2 = sqrt(R.*R+G.*G+B.*B+eps);
I2 = sqrt(R.*R+G.*G+eps);

theta_x=(R.*Gx-G.*Rx)./I2;
phi_x=(G.*(B.*Gx-G.*Bx)+R.*(B.*Rx-R.*Bx))./(intensityL2.*I2);
r_x=(R.*Rx+G.*Gx+B.*Bx)./intensityL2;

theta_y=(R.*Gy-G.*Ry)./I2;
phi_y=(G.*(B.*Gy-G.*By)+R.*(B.*Ry-R.*By))./(intensityL2.*I2);
r_y=(R.*Ry+G.*Gy+B.*By)./intensityL2;
#else

void sphericalDerivative(const cv::Mat &image, float sigma)
{
    std::vector< cv::Mat > channels;
    cv::split(image, channels);
    R = channels[2];
    G = channels[1];
    B = channels[0];
    
    cv::Mat_<float> g0, g1;
    gder(sigma, 0, g0);
    gder(sigma, 1, g1);
    
    cv::Mat dx, dy, g0;
    cv::Mat_<float> Rx, Ry, Gx, Gy, Bx, By, intensityL2, I2, theta_x, theta_y, phi_x, phi_y, r_x, r_y;
    
    cv::sepFilter2D(R, Rx, CV_32FC1, g1, g0.t());
    cv::sepFilter2D(R, Ry, CV_32FC1, g0, g1.t());
    cv::sepFilter2D(R, R, CV_32FC1, g0, g0.t())
    
    cv::sepFilter2D(G, Gx, CV_32FC1, g1, g0.t());
    cv::sepFilter2D(G, Gy, CV_32FC1, g0, g1.t());
    cv::sepFilter2D(G, G, CV_32FC1, g0, g1.t());
    
    cv::sepFilter2D(B, Bx, CV_32FC1, g1, g0.t());
    cv::sepFilter2D(B, By, CV_32FC1, g0, g1.t());
    cv::sepFilter2D(B, B, CV_32FC1, g0, g1.t());
    
    cv::sqrt(R.mul(R) + G.mul(G) + B.mul(B) + 1e-6f, intensityL2);
    cv::sqrt(R.mul(R) + G.mul(G) + 1e-6f, I2);
    
    theta_x=(R.mul(Gx)-G.mul(Rx)) / I2;
    phi_x=(G.mul(B.mul(Gx)-G.mul(Bx)) + R.mul(B.mul(Rx)-R.mul(Bx)) / (intensityL2.mul(I2));
    r_x=(R.mul(Rx)+G.mul(Gx)+B.mul(Bx)) / intensityL2;
    
    theta_y=(R.mul(Gy)-G.mul(Ry)) / I2;
    phi_y=(G.*(B.*Gy-G.*By)+R.*(B.*Ry-R.*By))./(intensityL2.*I2);
    r_y=(R.*Ry+G.*Gy+B.*By)./intensityL2;


}

#endif

//////////

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
    cv::Mat result = cv::min(image, mu[0] + sigma[0]*n);
    return cv::min(result, findPercentile(result, percent, 100));
}

// This runs an infinite loop showing quadrature filter magnitude from [0 .. 2*pi] for G2/H2 and G4/H4
static void demo(const cv::Mat_<float> &image)
{
    cv::Mat_<float> g2, h2, g4, h4, m2, m4, e2, p2, pa, lambda2, lambdaa, phi, cp, sp;
    
    float s = 2.0;
    fa::SteerableFiltersG2 filtersa(image, int(4*s), 0.67/s);
    fa::SteerableFiltersG2 filters2(image, 4, 0.67);
    fa::SteerableFiltersG4 filters4(image, 6, 0.50);
    
    float t = 0.f;
    do
    {
        float theta = t * M_PI/180.0;
        
        filtersa.steer(theta, g2, h2, e2, m2, p2);
        { cv::Mat tmp[] = {g2, h2, m2}; cv::hconcat(tmp, 3, m2); }
        cv::normalize(m2, m2, 0, 1, cv::NORM_MINMAX, CV_32FC1);
        
        phi = cv::abs( p2 + M_PI_2 );
        cv::polarToCart(cv::Mat(), phi, cp, sp);
        lambda2 = cp.mul(cp);
        lambda2.setTo(0, (phi >= M_PI_2));
        cv::imshow("lambda2", lambda2);
        
        e2 = e2.mul(lambda2), cv::normalize(squash(e2, 1.0, 95), e2, 0, 1, cv::NORM_MINMAX);
        cv::imshow("e2", e2);
        cv::waitKey(0);
        
        filters4.steer(theta, g4, h4);
        cv::magnitude(g4, h4, m4);
        { cv::Mat tmp[] = {g4, h4, m4}; cv::hconcat(tmp, 3, m4); }
        cv::normalize(m4, m4, 0, 1, cv::NORM_MINMAX, CV_32FC1);

        cv::hconcat(m2, m4, m4);
        
        cv::Mat canvas;
        m4.convertTo(canvas, CV_8UC1, 255.0);
        cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
        
        cv::Point p(canvas.cols/2, canvas.rows/2), q(10.0*cos(theta), 10.0*sin(theta));
        q.x *= -1.0;
        cv::line(canvas, p, p + q, CV_RGB(0,255,0), 2, CV_AA);
        
        t += 2.0;
        cv::namedWindow("g2h2m2g4h4m4", CV_WINDOW_NORMAL);
        cv::imshow("g2h2m2g4h4m4", canvas);
   
    } while(cv::waitKey(20) != int('q'));
}

static cv::Mat sqr(const cv::Mat &src) { return src.mul(src); }
void ColorQuadEdge(const cv::Mat &src, cv::Mat &dx, cv::Mat &dy, cv::Mat &magnitude, float sigma = 1.0)
{
    
    cv::Mat dx_, dy_, dxx_, dyy_, dxx, dxy, dyy, lambda1, lambda2, D;
    
    cv::Mat_<float> g0, g1, gauss;
    gder(sigma, 0, gauss);
    gder(sigma, 0, g0);
    gder(sigma, 1, g1);

#if 1
    {
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        std::vector<cv::Mat_<float> > dxs(3), dys(3);
        for(int i = 0; i < 3; i++)
        {
            fa::SteerableFiltersG2 filters2(channels[i], 4.0*sigma, 0.67/sigma);
            cv::Mat_<float> g2, h2;
            filters2.steer(atan2(0,-1), g2, h2);
            cv::magnitude(g2, h2, dxs[i]);
            filters2.steer(atan2(1, 0), g2, h2);
            cv::magnitude(g2, h2, dys[i]);
        }
        cv::merge(dxs, dx_);
        cv::merge(dys, dy_);
    }

#else
      cv::sepFilter2D(src, dx_, CV_32FC3, g1, g0.t());
    cv::sepFilter2D(src, dy_, CV_32FC3, g0, g1.t());
#endif
    
    CV_Assert(dx_.channels() == 3);
    
    dx_ = dot(dx_, dx_);
    dy_ = dot(dy_, dy_);
    
    dxx = dx_ * (sigma * sigma); // these are already squared here
    dyy = dy_ * (sigma * sigma);
    cv::sqrt(dx_, dx_);
    cv::sqrt(dy_, dy_);
    dxy = (dx_.mul(dy_)) * (sigma * sigma);
    
#if 0
    cv::sepFilter2D(dxx, dxx, CV_32FC1, gauss, gauss.t());
    cv::sepFilter2D(dxy, dxy, CV_32FC1, gauss, gauss.t());
    cv::sepFilter2D(dyy, dyy, CV_32FC1, gauss, gauss.t());
#endif
    
    // Compute the eigenvalues:
    cv::sqrt(sqr(dxx-dyy) + sqr(2.0 * dxy) + 1e-10f, D);
    cv::Mat tmp = dxx + dyy;
    lambda1 = 0.5 * (tmp + D);
    lambda2 = 0.5 * (tmp - D);
    
    // Compute eigenvectors:
    dy = (dyy - dxx + D);
    dx = (2.0 * dxy);
    cv::magnitude(dx, dy, magnitude);
    dy /= (magnitude + 1e-10f);
    dx /= (magnitude + 1e-10f);
    
    //sqrt(lambda1 + lambda2, magnitude);
    sqrt(lambda1 - lambda2, magnitude); // return line strength
};

#include <numeric>
#include "SteerableFiltersG4.h"
#include "Pyramid.h"



#include <iostream>
using namespace std;
using namespace cv;
#include "SkinProbablilityMaps.h"

#define DO_FRANKOT_CHELLAPA 0

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

    cv::Mat_<cv::Vec3b> createTestImage(const cv::Size &size) const
    {
        cv::Mat_<cv::Vec3b> image(size, cv::Vec3b::all(127));
        cv::Rect box(image.cols/8,image.rows/4,image.cols/4, image.rows/2);
        //cv::rectangle(image, box, CV_RGB(0,0,0), -1);
        cv::circle(image, cv::Point(image.cols/2,image.rows/2), box.size().width/2, CV_RGB(0,0,0), -1);
        cv::line(image, cv::Point(image.cols*3/4, 0), cv::Point(image.cols*3/4, image.rows), CV_RGB(255,255,255), 3);
        cv::line(image, cv::Point(image.cols*1/4, 0), cv::Point(image.cols*1/4, image.rows), CV_RGB(0,0,0), 3);
        return image;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        for (int i = r.start; i != r.end; i++)
        {
            cv::Mat image = cv::imread(m_filenames[i]), gray;
            if(image.empty())
                continue;
        
            //image = createTestImage(cv::Size(400, 200));
        
            if(image.channels() != 1)
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            
            // demo(gray);
#if 0
            { // steerable pyramids
                fa::Pyramid pyramid(gray), edges(gray.size(), CV_32FC1);
                for(int j = 0; j < pyramid.size(); j++)
                {
                    cv::Mat_<float> g2, h2, e, magnitude, phase;
                    fa::SteerableFiltersG2 filters2(pyramid[j], 4, 0.67);
                    filters2.steer(filters2.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
                    cv::Mat_<float> tmp(edges[j]);
                    filters2.findEdges(magnitude, phase, tmp);
                    cv::normalize(tmp, tmp, 0, 1, cv::NORM_MINMAX);
                }
                cv::Mat canvas = fa::draw(edges);
                cv::imshow("pyramid", canvas), cv::waitKey(0);
            }
#endif
            
            { // bilateral filtering 
                cv::Mat tmp, tmp2;
                cv::cvtColor(image, tmp, cv::COLOR_BGR2Lab);
                cv::bilateralFilter(tmp, tmp2, 10, 20, 0);
                cv::cvtColor(tmp2, image, cv::COLOR_Lab2BGR);
            }
            
            cv::Mat_<float> z;
            gray.convertTo(z, CV_32FC1);
            cv::normalize(z, z, 0, 1, cv::NORM_MINMAX, CV_32FC1);

//            cv::imshow("z", z);
//            cv::dilate(z, z, cv::Mat(), cv::Point(-1,-1), 2);
//            cv::erode(z, z, cv::Mat(), cv::Point(-1,-1), 2);
//            cv::imshow("z2", z);

#if DO_FRANKOT_CHELLAPA
            cv::Mat_<float> z2, m;
            { // reconstruct integrable image from local gradients for color => gray projection
                cv::Mat dx, dy, o;
                MaxSobel(image, dx, dy, 3, 1.0), cv::magnitude(dx, dy, m);
                //ColorSobel(image, dx, dy, m, 3, 1.0);
                m = squash(m, 1.0, 90);
                dx = dx.mul(m);
                dy = dy.mul(m);
                frankotchellapa(dx, dy, z2);
                cv::normalize(z2, z2, 0, 1, cv::NORM_MINMAX, CV_32FC1);
            }
#endif
            
            cv::Mat_<float> g2, h2, e, magnitude, phase, edges, lines0, lines1;
            
            float s=2.0;
            fa::SteerableFiltersG2 filters2(z, 4*s, 0.67/s);
            filters2.steer(filters2.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
            filters2.findEdges(magnitude, phase, edges);
            filters2.findDarkLines(magnitude, phase, lines0);
            filters2.findBrightLines(magnitude, phase, lines1);
            
#if 0
            cv::Mat_<float> lambda, phi = cv::abs( cv::abs(phase)-std::abs(M_PI_2) ), ct, st, foo;
            cv::polarToCart(cv::Mat(), phi, ct, st);
            lambda = ct.mul(ct);

            phi = cv::abs( phase ); // dark lines
            cv::polarToCart(cv::Mat(), phi, ct, st);
            cv::Mat_<float> lines = ct.mul(ct);
            
            cv::normalize(e, e, 0, 1, cv::NORM_MINMAX);
            
            cv::Mat_<float> e2 = e.mul(lambda);
            cv::normalize(squash(e2, 2.0, 90), e2, 0, 1, cv::NORM_MINMAX);
            
            cv::namedWindow("lambda", CV_WINDOW_NORMAL);
            while(true)
            {
                cv::imshow("lambda", lambda);
                cv::waitKey(0);

                cv::imshow("lambda", z);
                cv::waitKey(0);
   
                cv::imshow("lambda", e);
                cv::waitKey(0);
                
                cv::imshow("lambda", e2);
                cv::waitKey(0);
            }
            
            // demo(z);
#endif
            
            cv::Mat quad, fx, fy;
            ColorQuadEdge(image, fx, fy, quad, 2.0);
            
            cv::normalize(quad, quad, 0, 1, cv::NORM_MINMAX, CV_32F);
            //cv::imshow("quad", quad);

            
            // ((((((((((((((((( Now display the computed results )))))))))))))))))
            std::vector<cv::Mat> images;
            cv::Mat canvas;;
            
            cv::normalize(z, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());

            
#if DO_FRANKOT_CHELLAPA
            cv::normalize(z2, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
#endif
            
#if 0
            // Normalize the g2 h2 images together:
            cv::hconcat(g2, h2, g2);
            g2 = squash(g2, 2.0, m_percentileRank);
            cv::normalize(g2, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
#endif
            
            
#if 0
            // Add the dominant orientation energy:
            e = squash(e, 2.0, m_percentileRank);
            cv::normalize(e, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());

            
            // Normalize the phase-edge and phase-line images together to compare magnitude:
            //            cv::Mat channels[3] = { edges, lines0, lines1 };
            cv::Mat channels[3] = { edges, lines0, lines1 };
            //for(int i = 0; i < 3; i++)
            //    cv::normalize(channels[i], channels[i], 0, 1, cv::NORM_MINMAX);
            
            cv::hconcat(channels, 3, edges);
            //edges = squash(edges, 6.0, m_percentileRank);
            cv::normalize(edges, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
#else
            cv::normalize(lines0, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back( canvas.clone() );
#endif
        
            //cv::Mat arrows = quiver(image, -dx, dy, 3, 3, 10.0);
            //cv::imshow("a", arrows), cv::waitKey(0);
        
            cv::hconcat(images, canvas);  
            
#if 1
            cv::Mat output;
            {
                cv::Mat mask, pdf;
                cv::Mat_<uchar> border(image.size(), 255);
                int b = 5;
                border( cv::Rect(b, b, border.cols-2*b, border.rows - 2*b) ).setTo(0);
                cv::ellipse(border, cv::RotatedRect(cv::Point2f(image.cols/2,image.rows/2), cv::Size2f(image.cols,image.rows), 0), 0, -1);
                
                cv::Mat rgb;
                cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
                
                SkinProbablilityMaps spm;
                spm.setTheta(2.0);
                spm.verbose = true;
                spm.boostrap(rgb, mask), mask |= border;
                
                for (int i=0; i<10; i++)
                { //predict-train N times for convergence
                    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
                    cv::erode(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);  cv::imshow("mask", mask), cv::waitKey(0);
                    spm.train(rgb, mask);
                    spm.predict(rgb, mask);
                }
                spm.probability(rgb, pdf);
                
                cv::normalize(pdf, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::hconcat(gray, output, output);
                cv::hconcat(output, mask, output);
                cv::imshow("image", rgb),  cv::imshow("pdf", output), cv::waitKey(0);
            }
            cv::hconcat(canvas, output, canvas);
#endif

            cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
            cv::hconcat(image, canvas, canvas);
                
            //cv::imshow("g2h2", canvas), cv::waitKey(0);
            
            m_images[i] = canvas;
        
            if(!m_images[i].empty())
            {
                std::string filename = m_directory + "/" + basename(m_filenames[i]) + "_g2h2.png";
                cv::imwrite(filename, m_images[i]);
                
                filename = m_directory + "/" + basename(m_filenames[i]) + "_quad2.png";
                cv::normalize(quad, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::imwrite(filename, canvas);

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


///////// Frequency domain low pass filter for timing purposes //////

void GaussianFrequency(const cv::Mat_<float> &image, cv::Mat_<float> &output, float sigma) // todo convert sigma to [0 .. 0.5] wavelength
{
    cv::Mat_<float> h;
    cv::createHanningWindow(h, image.size(), CV_32FC1);
    
    cv::Mat_<float> filter(image.size(), 0); // Simple LP idea filter for now, good enough for timing, replace w/ gaussian
    cv::circle(filter, cv::Point(image.cols/2, image.rows/2), cv::min(image.rows, image.cols)/8, 1.0, -1);
    filter *= (1.0 / sum(filter)[0]);
    filter = ifftshift(filter);
    //cv::imshow("filter", filter), cv::waitKey(0);

    cv::Mat ifft, ffft;
    {
        cv::Mat planes[2] = { image /* image.mul(h) */, cv::Mat_<float>(image.size(), 0.f) }; /* no hanning window for speed */
        cv::merge(planes, 2, ifft);
        cv::dft(ifft, ifft, cv::DFT_COMPLEX_OUTPUT);
    }
    {
        cv::Mat planes[2] = { filter, cv::Mat_<float>(image.size(), 0.f) };
        cv::merge(planes, 2, ffft);
        //cv::dft(ffft, ffft, cv::DFT_COMPLEX_OUTPUT);
    }

    cv::mulSpectrums(ifft, ffft, ifft, cv::DFT_COMPLEX_OUTPUT, false);
    cv::idft(ifft, output, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE);
}

#include <opencv2/imgproc/imgproc.hpp>

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
    

    
#if 0
    // ((((((((( Compare spatial domain and frequency domain filtering )))))))))
    cv::Mat image = cv::imread(filenames.front(), cv::IMREAD_GRAYSCALE);
    cv::Mat_<float> real(image.size()), output;
    image.convertTo(real, CV_32FC1);
    
    int M = cv::getOptimalDFTSize( image.rows );
    int N = cv::getOptimalDFTSize( image.cols );
    cv::copyMakeBorder(real, real, 0, M - image.rows, 0, N - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // run this once to put the image in the cache:
    cv::Mat_<float> gx(1, 65, 1.0f); // sample 33 tap filter
    cv::sepFilter2D(real, output, CV_32FC1, gx, gx.t());
    
    DECLARE_TIMING(thing);

    START_TIMING(thing);
    for(int i = 0; i < 100; i++) GaussianFrequency(real, output, 2.0);
    STOP_TIMING(thing);
    std::cout << "Frequency: " << GET_AVERAGE_TIMING(thing) << std::endl;    
    
    // cv::normalize(output, output, 0, 1, cv::NORM_MINMAX), cv::imshow("output", output), cv::waitKey(0);
    
    START_TIMING(thing);
    for(int i = 0; i < 100; i++) cv::sepFilter2D(real, output, CV_32FC1, gx, gx.t());
    STOP_TIMING(thing);
    std::cout << "Spatial: " << GET_AVERAGE_TIMING(thing) << std::endl;

    // cv::normalize(output, output, 0, 1, cv::NORM_MINMAX), cv::imshow("output", output), cv::waitKey(0);
    ////////////
#endif
    
    std::vector<cv::Mat> images(filenames.size());
    ParallelSteerable body(filenames, images, directory);
    
    body( cv::Range(0, filenames.size() ) );
    //cv::parallel_for_(cv::Range(0, static_cast<int>(filenames.size())), body);
    
    return 0;
}

