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

#include "cvsteer/Pyramid.h"
#include "cvsteer/SteerableFiltersG2.h"
#include "cvsteer/SteerableFiltersG4.h"

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

cv::Mat3b createTestImage(const cv::Size &size)
{
    cv::Mat_<cv::Vec3b> image(size, cv::Vec3b::all(127));
    cv::Rect box(image.cols/8,image.rows/4,image.cols/4, image.rows/2);
    cv::circle(image, cv::Point(image.cols/2,image.rows/2), box.size().width/2, CV_RGB(0,0,0), -1);
    cv::line(image, cv::Point(image.cols*3/4, 0), cv::Point(image.cols*3/4, image.rows), CV_RGB(255,255,255), 3);
    cv::line(image, cv::Point(image.cols*1/4, 0), cv::Point(image.cols*1/4, image.rows), CV_RGB(0,0,0), 3);
    return image;
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
        
        e2 = e2.mul(lambda2);
        cv::normalize(e2, e2, 0, 1, cv::NORM_MINMAX);
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

using namespace std;
using namespace cv;

class ParallelSteerable : public cv::ParallelLoopBody
{
public:
    ParallelSteerable(const std::vector<std::string> &filenames, std::vector<cv::Mat> &images, const std::string &directory )
    : m_filenames(filenames)
    , m_images(images)
    , m_directory(directory)
    , m_doLogging(true) {}
    
    void setDoLogging(bool flag) { m_doLogging = flag; }

    virtual void operator()( const cv::Range &r ) const
    {
        for (int i = r.start; i != r.end; i++)
        {
            cv::Mat image = cv::imread(m_filenames[i]), gray;
            if(image.empty())
            {
                continue;
            }
        
            if(image.channels() != 1)
            {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            }

            // steerable pyramids
            fa::Pyramid pyramid(gray), edges(gray.size(), CV_32FC1);
            for(int j = 0; j < pyramid.size(); j++)
            {
                cv::Mat1f g2, h2, e, magnitude, phase, tmp(edges[j]);
                fa::SteerableFiltersG2 filters2(pyramid[j], 4, 0.67);
                filters2.steer(filters2.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
                filters2.findEdges(magnitude, phase, tmp);
                cv::normalize(tmp, tmp, 0, 1, cv::NORM_MINMAX);
            }
            cv::Mat canvas = fa::draw(edges);
            cv::imshow("pyramid", canvas), cv::waitKey(0);
            
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

// ((((((((((((((((((((((((((((((( Command line parameters )))))))))))))))))))))))))))))))
const char *version = "0.1";
const char *keys =
{
    "{  input       |       | input training file                        }"
    "{  output      |  /tmp | output directory                           }"
    "{  demo        | false | do steerable filter demo                   }"
    "{  version     | false | display version number                     }"
    "{  build       | false | display opencv build information           }"
    "{  help        | false | help message                               }"
};

int main(int argc, const char * argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    
    if(argc < 2 || parser.get<bool>("help"))
    {
        //parser.printParams();
        return 0;
    }
    else if(parser.get<bool>("build"))
    {
        std::cout << cv::getBuildInformation() << std::endl;
        return 0;
    }
    else if(parser.get<bool>("version"))
    {
        std::cout << argv[0] << " v" << version << std::endl;
        return 0;
    }
    else if(parser.get<bool>("demo"))
    {
        cv::Mat image = createTestImage(cv::Size(320, 240));
        if(image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }
        
        cv::Mat1f image_;
        cv::normalize(image, image_, 0, 1, cv::NORM_MINMAX, CV_32F);
        demo(image_);
    }
    else
    {
        std::string filename = parser.get<std::string>("input");
        std::string output = parser.get<std::string>("output");
        
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
        ParallelSteerable body(filenames, images, output);
        
        body( cv::Range(0, filenames.size() ) );
        //cv::parallel_for_(cv::Range(0, static_cast<int>(filenames.size())), body);
    }
    
    return 0;
}

