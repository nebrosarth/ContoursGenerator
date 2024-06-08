#include "DrawOperations.h"
#include "RandomGenerator.h"
#include <qpainter.h>
#include <ContoursOperations.h>
#include <qpainterpath.h>

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

	painter.setBrush(Qt::cyan);
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

void DrawOperations::drawContourValues(QPixmap& image, const Contour& contour, QColor textColor, const QFont& font)
{
	QPainter painter(&image);
	painter.setPen(textColor);
	painter.setFont(font);

	QPainterPath clipPath;

	// Draw text along the contour points
	// Calc the text rotation angle according to the slope of the line
	// Add rotated bounding rect of text to the clip path

	double minDist = 50;
	cv::Point prevPt;
	for (size_t i = 0; i < contour.points.size(); i++)
	{
		cv::Point pt1_cv = contour.points[i];
		cv::Point pt2_cv = contour.points[(i + 1) % contour.points.size()];

		if(i != 0)
		{
			double dist = cv::norm(pt1_cv - prevPt);
			if (dist < minDist)
			{
				continue;
			}
			double distEnd = cv::norm(pt1_cv - contour.points.back());
			if (distEnd < minDist)
			{
				break;
			}
		}
		prevPt = pt1_cv;

		QPoint pt1 = { pt1_cv.x, pt1_cv.y };
		QPoint pt2 = { pt2_cv.x, pt2_cv.y };

		QLineF line(pt1, pt2);
		double angle = line.angle();

		if (angle > 180) 
		{
			angle -= 180;
		}

		if (angle > 90)
		{
			angle -= 180;
		}

		angle = 360 - angle;

		QFont rotatedFont = font;
		rotatedFont.setPointSize(10);
		rotatedFont.setBold(true);

		QRectF textRect = painter.boundingRect(QRect(), Qt::AlignCenter, QString::number(contour.depth + 1));

		QTransform transform;
		transform.translate(pt1.x(), pt1.y());
		transform.rotate(angle);

		QRectF rotatedRect = transform.mapRect(textRect);

		clipPath.addRect(rotatedRect);

		painter.save();
		painter.translate(pt1);
		//painter.translate(-textRect.bottomRight());
		painter.rotate(angle);

		painter.drawText(textRect, Qt::AlignCenter, QString::number(contour.depth + 1));
		//painter.drawRect(textRect);
		painter.restore();
		painter.resetTransform();
	}

	QPainterPath clipInv;
	clipInv.addRect(image.rect());
	clipInv -= clipPath;

	painter.setClipPath(clipInv);
	// Draw contour polyline
	std::vector<QPoint> pts;
	for (auto& pt : contour.points)
	{
		pts.emplace_back(pt.x, pt.y);
	}

	painter.drawPolyline(pts.data(), contour.points.size());
}
