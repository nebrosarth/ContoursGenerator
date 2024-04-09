#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ContoursGenerator.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ContoursGeneratorClass; };
QT_END_NAMESPACE

struct GenerationParams
{
    int width, height;
};

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

private:
    std::unique_ptr<Ui::ContoursGeneratorClass> ui;
};
