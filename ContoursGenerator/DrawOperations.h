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
};


namespace DrawOperations
{
	void drawRandomWell(QPixmap& image, const WellParams& params);
	void drawWellTitle(QPainter& painter, const QPoint& wellPt, const WellParams& params);
	void drawContourValues(QPixmap& image, const Contour& contour, QColor textColor, const QFont& font);
};

