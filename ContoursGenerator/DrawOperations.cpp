#include "DrawOperations.h"
#include "RandomGenerator.h"
#include <qpainter.h>

void DrawOperations::drawRandomWell(QPixmap& image, const WellParams& params)
{
	int width = image.width();
	int height = image.height();
	int radius = params.radius;

	auto& gen = RandomGenerator::instance();
	QPoint wellPt = gen.getRandomPoint(image.width(), image.height());

	QPainter painter(&image);

	int outline = params.outline;
	if (outline > 0)
	{
		QPen pen = painter.pen();
		pen.setWidth(params.outline);
		painter.setPen(pen);
	}
	else
	{
		painter.setPen(Qt::NoPen);
	}

	painter.setBrush(params.color);
	painter.drawEllipse(wellPt, radius, radius);

	if (params.drawText)
	{
		drawWellTitle(painter, wellPt, params);
	}
}

void DrawOperations::drawWellTitle(QPainter& painter, const QPoint& wellPt, const WellParams& params)
{
	int offset = params.radius + params.offset;
	QPoint textPt(wellPt.x() + offset, wellPt.y() - offset);

	QFont font;
	font.setPointSize(params.fontSize);
	painter.setFont(font);

	painter.setPen(QPen());

	short idWell = RandomGenerator::instance().getRandomInt(999);

	QString idWellStr = QString::number(idWell);

	painter.drawText(textPt, idWellStr);
}
