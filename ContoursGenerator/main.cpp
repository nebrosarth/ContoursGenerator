#include "ContoursGenerator.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ContoursGenerator w;
    w.show();
    return a.exec();
}
