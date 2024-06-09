#pragma once
#include <opencv2/opencv.hpp>

struct Contour
{
    int index;
    double value;
    bool isClosed;
    int depth;
    std::vector<cv::Point> points;
    cv::Rect boundingRect;
};

class ColorScaler
{
public:
    ColorScaler(double min, double max, const cv::Scalar& minColor, const cv::Scalar& maxColor);
    cv::Scalar getColor(double value) const;
protected:
    double m_min;
    double m_max;
    cv::Scalar m_minColor;
    cv::Scalar m_maxColor;
};

enum class Direction
{
    TOP,
    TOP_RIGHT,
    RIGHT,
    BOTTOM_RIGHT,
    DOWN,
    BOTTOM_LEFT,
    LEFT,
    TOP_LEFT,
    NONE
};

struct GenerationParams
{
    int width, height; // image size
    double Xmul, Ymul; // multipliers for X and Y for Perlin noise
    int mul; // general multiplier for Perlin noise
    bool generateWells; // generate wells
    int numOfWells; // number of wells
    bool generateIsolines; // generate isolines
    bool fillContours; // fill contours with color
    bool drawValues; // draw values on isolines
    int textDistance; // minimal distance between texts on isolines
};

namespace ContoursOperations
{
    cv::Mat generateIsolines(const GenerationParams& params);
    void findContours(const cv::Mat& img, std::vector<Contour>& contours);
    void extractContour(int x_start, int y_start, cv::Mat& img, std::vector<cv::Point>& contour);
    Direction getDirection(cv::Point prev, cv::Point next);
    std::vector<cv::Point> getOrder(cv::Point pt, Direction direction);
    // Find depth of each contour
    void findDepth(cv::Mat& img, std::vector<Contour>& contours);
    void fillContours(cv::Mat& contoursMat, const std::vector<Contour>& contours, cv::Mat& drawing);
};

