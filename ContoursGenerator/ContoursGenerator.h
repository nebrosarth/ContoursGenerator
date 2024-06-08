#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ContoursGenerator.h"
#include "opencv2/opencv.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class ContoursGeneratorClass; };
QT_END_NAMESPACE

struct WellParams;

struct GenerationParams
{
    int width, height;
    double Xmul, Ymul;
    int mul;
    bool generateWells;
    int numOfWells;
};

struct GenImg
{
    QPixmap image;
    QPixmap mask;
};

namespace utils
{
    QPixmap cvMat2Pixmap(const cv::Mat& input);
}

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

class ContoursGenerator : public QMainWindow
{
    Q_OBJECT

public:
    ContoursGenerator(QWidget *parent = nullptr);
    ~ContoursGenerator();

protected slots:
    void OnGenerateImage();
    void OnUpdateImage();
    void OnSaveImage();
    void OnSaveBatch();

protected:
    void initConnections();
    cv::Mat generateIsolines(const GenerationParams& params);
    GenImg generateImage();

    void findContours(const cv::Mat& img, std::vector<Contour>& contours);
    void extractContour(int x_start, int y_start, cv::Mat& img, std::vector<cv::Point>& contour);
    Direction getDirection(cv::Point prev, cv::Point next);
    std::vector<cv::Point> getOrder(cv::Point pt, Direction direction);
    // Find depth of each contour
    void findDepth(cv::Mat& img, std::vector<Contour>& contours);
    void fillContours(cv::Mat& contoursMat, const std::vector<Contour>& contours, cv::Mat& drawing);

    void saveImage(const QString& folderPath, const QPixmap& img, const QPixmap& mask);
    GenerationParams getUIParams();
    WellParams getUIWellParams();

    template<int size> 
    void setSize(); // set image size

private:
    std::unique_ptr<Ui::ContoursGeneratorClass> ui;
    QPixmap m_generatedImage;
    QPixmap m_generatedMask;
};
