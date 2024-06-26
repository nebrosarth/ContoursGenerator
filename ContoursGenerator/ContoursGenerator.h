#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ContoursGenerator.h"
#include "opencv2/opencv.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class ContoursGeneratorClass; };
QT_END_NAMESPACE

struct WellParams;
struct GenerationParams;

struct GenImg
{
    QPixmap image;
    QPixmap mask;
};

namespace utils
{
    QPixmap cvMat2Pixmap(const cv::Mat& input);
    cv::Mat QPixmap2cvMat(const QPixmap& in, bool grayscale);
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
    void OnSaveImage();
    void OnSaveBatch();

protected:
    void initConnections();
    GenImg generateImage();

    void saveImageSplit(const QString& folderPath, const GenImg& gen);
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
