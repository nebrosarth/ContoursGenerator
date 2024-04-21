#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ContoursGenerator.h"
#include "opencv.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class ContoursGeneratorClass; };
QT_END_NAMESPACE

struct GenerationParams
{
    int width, height;
    double Xmul, Ymul;
    int mul;
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

protected:
    void initConnections();
    void generateImage();
    GenerationParams getUIParams();

    template<int size> 
    void setSize(); // set image size

private:
    std::unique_ptr<Ui::ContoursGeneratorClass> ui;
};
