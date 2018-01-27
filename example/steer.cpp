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

#include <cvsteer/SteerableFiltersG2.h>
#include <cvsteer/SteerableFiltersG4.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iostream>

// For use with istream_iterator to read complete lines (new line delimiter):
// http://stackoverflow.com/questions/1567082/how-do-i-iterate-over-cin-line-by-line-in-c/1567703#1567703
struct Line
{
    std::string data;
    friend std::istream& operator>>(std::istream& is, Line& l)
    {
        std::getline(is, l.data);
        return is;
    }
    operator std::string() const { return data; }
};

// BASH style basename function with STL string
static std::string basename(const std::string& name)
{
    size_t pos = name.rfind("/") + 1;
    std::string base = name.substr((pos < name.size()) * pos);
    return base.substr(0, std::min(base.size() - 1, base.rfind(".")));
};

class ParallelSteerable : public cv::ParallelLoopBody
{
public:
    ParallelSteerable(const std::vector<std::string>& filenames, const std::string& directory, float gain)
        : m_filenames(filenames)
        , m_directory(directory)
        , m_gain(gain)
    {
    }

    virtual void operator()(const cv::Range& r) const
    {
        for (int i = r.start; i != r.end; i++)
        {
            cv::Mat image = cv::imread(m_filenames[i]), gray;
            if (image.empty())
            {
                continue;
            }

            if (image.channels() != 1)
            {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            }

            // steerable filters
            cv::Mat1f g2, h2, e, magnitude, phase, edges, linesDark, linesBright;
            fa::SteerableFiltersG2 filters2(gray, 4, 0.67f);
            filters2.steer(filters2.getDominantOrientationAngle(), g2, h2, e, magnitude, phase);
            filters2.findEdges(magnitude, phase, edges);
            filters2.findDarkLines(magnitude, phase, linesDark);
            filters2.findBrightLines(magnitude, phase, linesBright);

            cv::Mat1b edges8u(edges.size()), linesDark8u(linesDark.size()), linesBright8u(linesBright.size());
            if(m_gain > 0.f)
            {
                edges.convertTo(edges8u, CV_8UC1, m_gain);
                linesDark.convertTo(linesDark8u, CV_8UC1, m_gain);
                linesBright.convertTo(linesBright8u, CV_8UC1, m_gain);
            }
            else
            {
                cv::normalize(edges, edges8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::normalize(linesDark, linesDark8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::normalize(linesBright, linesBright8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);                
            }
            
            if(!m_directory.empty())
            {
                { // write edges
                    std::string filename = m_directory + "/" + basename(m_filenames[i]) + "_edges.png";
                    cv::imwrite(filename, edges8u);
                }
                
                { // write lines (dark)
                    std::string filename = m_directory + "/" + basename(m_filenames[i]) + "_lines_dark.png";
                    cv::imwrite(filename, linesDark8u);
                }

                { // write lines (bright)
                    std::string filename = m_directory + "/" + basename(m_filenames[i]) + "_lines_bright.png";
                    cv::imwrite(filename, linesBright8u);
                }                
            }
        }
    }

protected:
    
    const std::vector<std::string>& m_filenames;    
    const std::string& m_directory;
    float m_gain = 0.f; // default 0 will run cv::normalize()    
};

const char* keys = {

    "{ input    |       | input training file         }"
    "{ output   |       | output directory            }"
    "{ gain     |  0.0  | gain for CV_8UC1 output     }"
    "{ verbose  | false | use verbose display         }"
    "{ help     | false | help message                }"
};

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);

    if (argc < 2 || parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }
    else
    {
        std::string filename = parser.get<std::string>("input");
        std::string output = parser.get<std::string>("output");

        std::vector<std::string> filenames;
        if ((filename.rfind(".txt") != std::string::npos) || (filename.rfind(".") == std::string::npos))
        {
            std::ifstream file(filename.c_str());
            std::copy(std::istream_iterator<Line>(file), std::istream_iterator<Line>(), std::back_inserter(filenames));
        }
        else
        {
            filenames.push_back(filename);
        }

        bool verbose = parser.get<bool>("verbose");
        ParallelSteerable body(filenames, output, verbose);
        cv::parallel_for_(cv::Range(0, static_cast<int>(filenames.size())), body);
    }

    return 0;
}
