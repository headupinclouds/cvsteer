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
#include "cvutil.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iostream>


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
            //cv::resize(image, image, cv::Size(100, 100*image.rows/image.cols));
            
            cv::medianBlur(image, image, 3);
            
            if(image.channels() != 1)
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
           
            cv::Mat dx, dy, m, o;
            cv::Mat_<float> z;
            //MaxSobel(image, dx, dy, 3, 1.0);
            ColorSobel(image, dx, dy, m, 7, 1.0);
            m = squash(m, 3.0, 75);
            dx = dx.mul(m);
            dy = dy.mul(m);
            frankotchellapa(dx, dy, z);
            
            cv::Mat_<float> g2, h2, e, magnitude, phase, edges, lines0, lines1;
            SteerableFilters filters(z, 4, 0.67);
            filters.steer(filters.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
            filters.phaseEdge(e, phase, edges, 2.0);
            filters.phaseLine0(e, phase, lines0, 2.0);
            filters.phaseLine1(e, phase, lines1, 2.0);
            
#if 1
            cv::Mat_<float> g4, h4, m4, m2;
            SteerableFiltersG4 filters4(z, 6, 0.5);
            for(int t = 0; ; t++)
            {
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

