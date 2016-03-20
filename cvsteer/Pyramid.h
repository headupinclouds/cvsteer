/*
 * Pyramid.h
 *
 * Lineage: Pyramid8u.h
 *  Created on: Jul 2, 2012
 *      Author: headupinclouds
 *
 *  Copyright (c) 2013 David Hirvonen. All rights reserved.
 */

#ifndef cvsteer_Pyramid_h
#define cvsteer_Pyramid_h

#include <opencv2/core/core.hpp>

#define _STEER_BEGIN namespace fa { // Freeman and Adelson
#define _STEER_END }

_STEER_BEGIN

class Pyramid
{
public:
    Pyramid() {}
    Pyramid(const cv::Size &size, int type);
    Pyramid(const cv::Mat &image);
    ~Pyramid();
    const cv::Mat& operator[](int i) const { return m_level[i]; }
    cv::Mat& operator[](int i) { return m_level[i]; }
    size_t size() const { return m_level.size(); }
    Pyramid clone() const;
    void resize(size_t size) { m_level.resize(size); }
    void buildLaplacian();
    void reconstructFromLaplacian();
    
protected:
    std::vector<cv::Mat> m_level;
};

inline Pyramid operator *(double value, const Pyramid &pyramid)
{
    Pyramid tmp = pyramid.clone();
    for(int i = 0; i < tmp.size(); i++)
        tmp[i] = (value * tmp[i]);
        return tmp;
}

inline Pyramid operator *(const Pyramid &pyramid, double value)
{
    Pyramid tmp = pyramid.clone();
    for(int i = 0; i < tmp.size(); i++)
        tmp[i] = (value * tmp[i]);
        return tmp;
}

inline Pyramid operator -(const Pyramid &a, const Pyramid &b)
{
    Pyramid tmp = a.clone();
    for(int i = 0; i < tmp.size(); i++)
        tmp[i] -= b[i];
        return tmp;
}

inline Pyramid operator +(const Pyramid &a, const Pyramid &b)
{
    Pyramid tmp = a.clone();
    for(int i = 0; i < tmp.size(); i++)
        tmp[i] += b[i];
        return tmp;
}

inline Pyramid abs(const Pyramid &a)
{
    Pyramid tmp = a.clone();
    for(int i = 0; i < tmp.size(); i++)
        tmp[i] = cv::abs(tmp[i]);
    return tmp;
}

cv::Mat draw(Pyramid &pyramid);

_STEER_END


#endif
