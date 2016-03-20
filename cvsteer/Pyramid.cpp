//
//  Pyramid.cpp
//
//  Created by David Hirvonen on 1/11/13.
//  Copyright (c) 2013 David Hirvonen. All rights reserved.
//

#include "Pyramid.h"
#include <opencv2/imgproc/imgproc.hpp>

_STEER_BEGIN

Pyramid::Pyramid(const cv::Size &size, int type)
{
    cv::Mat image(size, type);
    m_level.push_back(image);
    while(m_level.back().size().area() >= 2)
    {
        cv::Mat level;
        cv::pyrDown(m_level.back(), level ); // TODO: replace with allocation, check if (w+1)/2 or w/2
        m_level.push_back( level );
    }
}

Pyramid::Pyramid(const cv::Mat &image)
{
    m_level.push_back(image);
    while(m_level.back().size().area() >= 2)
    {
        cv::Mat level;
        cv::pyrDown(m_level.back(), level );
        m_level.push_back( level );
    }
}

Pyramid::~Pyramid()
{
    
}

Pyramid Pyramid::clone() const
{
    Pyramid pyramid;
    for(int i = 0; i < m_level.size(); i++)
        pyramid.m_level.push_back( m_level[i].clone() );
    return pyramid;
}

void Pyramid::buildLaplacian()
{
    for(int i = 0; i < m_level.size() - 1; i++)
    {
        cv::Mat up;
        cv::pyrUp(m_level[i+1], up, m_level[i].size());
        m_level[i] = m_level[i] - up;
    }
}


void Pyramid::reconstructFromLaplacian()
{
    for(int i = m_level.size() - 2; i >= 0; i--)
    {
        cv::Mat up;
        cv::pyrUp(m_level[i+1], up, m_level[i].size());
        m_level[i] = m_level[i] + up;
    }
}

cv::Mat draw(Pyramid &pyramid)
{
    cv::Point tl(0,0);
    cv::Mat canvas(pyramid[0].rows, pyramid[0].cols+((pyramid[0].cols+2)>>1), pyramid[0].type(), cv::Scalar::all(0));
    for(int i = 0; i < pyramid.size() - 1; i++)
    {
        pyramid[i].copyTo(canvas(cv::Rect(tl, pyramid[i].size())));
        if(!(i % 2))
        {
            tl.x += pyramid[i].cols;
        }
        else
        {
            tl.y += pyramid[i].rows;
        }
    }
    return canvas;
}

_STEER_END