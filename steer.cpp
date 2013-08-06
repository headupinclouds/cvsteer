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

#include "SteerableFilters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>

typedef std::vector<cv::Point2f> PointSetf;

// A poor man's matlab quiver display, via upsampling and anti-aliased line drawing
static cv::Mat quiver(const cv::Mat &image, const cv::Mat_<cv::Vec2f> &orientation, int yTic, int xTic, float scale)
{
    cv::Mat canvas;
    if(canvas.channels() == 1)
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
float findPercentile(const cv::Mat_<float> &image, float rank = 99.0f)
{
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    float probability = std::max(std::min(rank, 0.f), 100.0f) / 100.0f; // normalize percentile

    // build a histogram:
    const float min = minVal, max = maxVal;
    cv::Mat hist;
    int bins = 100;
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
                continue;
            
            if(image.channels() != 1)
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            
            cv::Mat_<float> g2, h2, e, magnitude, phase, edges, lines;
            
            SteerableFilters filters(gray);
            filters.steer(filters.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
            filters.phaseEdge(e, phase, edges, 2.0);
            filters.phaseLine(e, phase, lines, 2.0);
            
            // Now display the computed results:
            std::vector<cv::Mat> images;
            cv::Mat canvas;;
            
            cv::normalize(gray, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            // Normalize the g2 h2 images together:
            cv::hconcat(g2, h2, g2);
            g2 = cv::min(g2, findPercentile(g2, 98.0));
            cv::normalize(g2, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            // Add the dominant orientation energy:
            e = cv::min(e, findPercentile(e, 98.0));
            cv::normalize(e, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());
            
            // Normalize the phase-edge and phase-line images together to compare magnitude:
            cv::hconcat(edges, lines, edges);
            edges = cv::min(edges, findPercentile(edges, 98.0));
            cv::normalize(edges, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            images.push_back(canvas.clone());

            cv::hconcat(images, canvas);
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
    cv::parallel_for_(cv::Range(0, static_cast<int>(filenames.size())), body);
    
    //cv::Mat canvas;
    //cv::vconcat(images, canvas);
    //cv::namedWindow("g2h2", CV_WINDOW_NORMAL), cv::imshow("g2h2", canvas), cv::waitKey(0);
    
    return 0;
}

