#include "ContoursGenerator.h"
#include "opencv.hpp"

ContoursGenerator::ContoursGenerator(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ContoursGeneratorClass())
{
    ui->setupUi(this);

    initConnections();
}

ContoursGenerator::~ContoursGenerator()
{
}

void ContoursGenerator::initConnections()
{
    connect(ui->pushButton_Generate, &QPushButton::pressed, this, &ContoursGenerator::generateImage);
}

void ContoursGenerator::generateImage()
{
    GenerationParams params = getUIParams();

    QPixmap pixmap(params.width, params.height);
    pixmap.fill(Qt::darkBlue);
    ui->label_Image->setPixmap(pixmap);
}

GenerationParams ContoursGenerator::getUIParams()
{
    GenerationParams params{};
    if (ui)
    {
        params.height = ui->spinBox_Width->value();
        params.width = ui->spinBox_Height->value();
    }
    return params;
}
