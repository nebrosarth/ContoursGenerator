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

class ContoursGenerator : public QMainWindow
{
    Q_OBJECT

public:
    ContoursGenerator(QWidget *parent = nullptr);
    ~ContoursGenerator();

protected slots:
    void OnGenerateImage();
    void OnUpdateImage();

protected:
    void initConnections();
    GenImg generateImage();
    GenerationParams getUIParams();
    WellParams getUIWellParams();

    template<int size> 
    void setSize(); // set image size

private:
    std::unique_ptr<Ui::ContoursGeneratorClass> ui;
    QPixmap m_generatedImage;
    QPixmap m_generatedMask;
};
