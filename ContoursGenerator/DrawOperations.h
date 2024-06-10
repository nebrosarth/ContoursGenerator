#pragma once
#include <qimage.h>

struct Contour;

struct WellParams
{
	int radius;
	int fontSize;
	int offset;
	bool drawText;
	int outline;
	QColor color;
};


namespace DrawOperations
{
	void drawRandomWell(QPixmap& image, const WellParams& params);
	void drawWellTitle(QPainter& painter, const QPoint& wellPt, const WellParams& params);
	void drawContourValues(QPainter& painter, const Contour& contour, QColor textColor, const QFont& font, int minTextDistance);
	void drawContour(QPainter& painter, const Contour& contour, QColor color);
};

