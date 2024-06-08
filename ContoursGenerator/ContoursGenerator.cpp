#include "ContoursGenerator.h"
#include "PerlinNoise.hpp"
#include <omp.h>
#include "DrawOperations.h"
#include "RandomGenerator.h"
#include <qfile.h>
#include <qfiledialog.h>
#include <qtemporaryfile.h>
#include <quuid.h>
#include <opencv2/ximgproc.hpp>
#include <stack>

ContoursGenerator::ContoursGenerator(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::ContoursGeneratorClass())
{
	ui->setupUi(this);

	initConnections();
}

ContoursGenerator::~ContoursGenerator()
{
}

void ContoursGenerator::OnGenerateImage()
{
	GenImg genImg = generateImage();
	m_generatedImage = genImg.image;
	m_generatedMask = genImg.mask;

	OnUpdateImage();
}

void ContoursGenerator::OnUpdateImage()
{
	if (ui->checkBox_ShowMask->isChecked())
	{
		ui->label_Image->setPixmap(m_generatedMask);
	}
	else
	{
		ui->label_Image->setPixmap(m_generatedImage);
	}
}

void ContoursGenerator::OnSaveImage()
{
	if (m_generatedImage.isNull() || m_generatedMask.isNull())
	{
		return;
	}

	QString folderName = QFileDialog::getExistingDirectory(this);
	if (folderName.isEmpty())
	{
		return;
	}

	saveImage(folderName, m_generatedImage, m_generatedMask);
}

void ContoursGenerator::OnSaveBatch()
{
	QString folderName = QFileDialog::getExistingDirectory(this);
	if (folderName.isEmpty())
	{
		return;
	}

	int batchSize = ui->spinBox_BatchSize->value();
	for (int i = 0; i < batchSize; ++i)
	{
		GenImg generation = generateImage();
		saveImage(folderName, generation.image, generation.mask);
	}
}

void ContoursGenerator::initConnections()
{
	connect(ui->pushButton_Generate, &QPushButton::pressed, this, &ContoursGenerator::OnGenerateImage);
	connect(ui->pushButton_256, &QPushButton::pressed, this, &ContoursGenerator::setSize<256>);
	connect(ui->pushButton_512, &QPushButton::pressed, this, &ContoursGenerator::setSize<512>);
	connect(ui->pushButton_1024, &QPushButton::pressed, this, &ContoursGenerator::setSize<1024>);
	connect(ui->pushButton_2048, &QPushButton::pressed, this, &ContoursGenerator::setSize<2048>);
	connect(ui->checkBox_ShowMask, &QCheckBox::stateChanged, this, &ContoursGenerator::OnUpdateImage);
	connect(ui->pushButton_Save, &QPushButton::pressed, this, &ContoursGenerator::OnSaveImage);
	connect(ui->pushButton_GenerateBatch, &QPushButton::pressed, this, &ContoursGenerator::OnSaveBatch);
}

cv::Mat ContoursGenerator::generateIsolines(const GenerationParams& params)
{
	const siv::PerlinNoise::seed_type seed = RandomGenerator::instance().getRandomInt(INT_MAX);
	const siv::PerlinNoise perlin{ seed };
	ui->label_Seed->setText(QString::number(seed));

	cv::Mat grad;
	cv::Mat n(params.width, params.height, CV_64FC1);

	double xMul = params.Xmul; // default: 0.005
	double yMul = params.Ymul; // default: 0.005
	int mul = params.mul; // default: 20

#pragma omp parallel for
	for (int j = 0; j < n.rows; ++j)
	{
		for (int i = 0; i < n.cols; ++i)
		{
			double noise = perlin.noise2D_01(i * xMul, j * yMul) * mul;
			int nnn = (int)ceil(noise);
			noise = noise - floor(noise);
			n.at<double>(j, i) = noise;
		}
	}

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Sobel(n, grad_x, CV_64FC1, 1, 0);
	cv::Sobel(n, grad_y, CV_64FC1, 0, 1);
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	cv::Mat normalized;
	n.convertTo(normalized, CV_8UC1, 255, 0);
	grad.forEach<uchar>([](uchar& u, const int* pos)
		{
			if (u == 1)
				u = 0;
			else if (u > 1)
			{
				u = 255;
			}
		});
	cv::Mat gradInv = cv::Scalar(255) - grad;

	return gradInv;
}

GenImg ContoursGenerator::generateImage()
{
	GenerationParams params = getUIParams();

	cv::Mat isolines = generateIsolines(params);

	cv::Mat mask = cv::Scalar(255) - isolines;

	// apply thinning
	cv::Mat thinned;
	cv::ximgproc::thinning(mask, thinned, cv::ximgproc::THINNING_GUOHALL);

	// crop by 1 pixel

	cv::Rect cropRect(1, 1, thinned.cols - 2, thinned.rows - 2);
	thinned = thinned(cropRect);

	std::vector<Contour> contours;

	findContours(thinned, contours);

	for (size_t i = 0; i < contours.size(); ++i)
	{
		Contour& c = contours[i];
		c.index = i;
		c.value = i + 1;

		bool isClosed = true;

		if (cv::norm(contours[i].points.front() - contours[i].points.back()) > 3)
		{
			isClosed = false;
		}

		c.isClosed = isClosed;

		c.boundingRect = cv::boundingRect(contours[i].points);
	}

	cv::Mat contours_mat = cv::Mat::zeros(thinned.size(), CV_8UC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		const Contour& c = contours[i];
		cv::Scalar color = cv::Scalar(255, 255, 255);
		for (size_t j = 0; j < c.points.size(); ++j)
		{
			contours_mat.at<uchar>(c.points[j]) = c.value;
		}
	}

	// Depth mat
	cv::Mat depthMat = cv::Mat::zeros(thinned.size(), CV_8UC1);

	findDepth(contours_mat, contours);

	for (size_t i = 0; i < contours.size(); i++)
	{
		const Contour& c = contours[i];
		for (size_t j = 0; j < c.points.size(); ++j)
		{
			depthMat.at<uchar>(c.points[j]) = c.depth + 1;
		}
	}

	// Draw contours
	cv::Mat drawing = cv::Mat::zeros(thinned.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 255, 255);
		for (size_t j = 0; j < contours[i].points.size(); ++j)
		{
			cv::Scalar color = contours[i].isClosed ? cv::Scalar(75, 75, 75) : cv::Scalar(150, 100, 150);
			drawing.at<cv::Vec3b>(contours[i].points[j]) = cv::Vec3b(color[0], color[1], color[2]);
		}
	}

	// Fill areas
	fillContours(contours_mat, contours, drawing);

	QPixmap pixIso = utils::cvMat2Pixmap(drawing);

	if (params.generateWells)
	{
		WellParams wellParams = getUIWellParams();
		for (int i = 0; i < params.numOfWells; ++i)
		{
			DrawOperations::drawRandomWell(pixIso, wellParams);
		}
	}

	QPixmap pixMask = utils::cvMat2Pixmap(mask);

	GenImg result{ pixIso, pixMask };
	return result;

}

void ContoursGenerator::findContours(const cv::Mat& img, std::vector<Contour>& contours)
{
	int width = img.cols;
	int height = img.rows;

	cv::Mat mat = img.clone();
	for (int m = 0; m < height; ++m)
	{
		for (int n = 0; n < width; ++n)
		{
			if (mat.at<uchar>(m, n) == 255)
			{
				Contour c;
				extractContour(n, m, mat, c.points);
				contours.push_back(std::move(c));
			}
		}
	}
}

void ContoursGenerator::extractContour(int x_start, int y_start, cv::Mat& img, std::vector<cv::Point>& contour)
{
	int width = img.cols;
	int height = img.rows;

	int x = x_start;
	int y = y_start;

	auto isContour = [&](int x, int y) -> bool
		{
			if (x < 0 || x >= width || y < 0 || y >= height)
			{
				return false;
			}
			return img.at<uchar>(y, x) == 255;
		};

	std::stack<cv::Point> stack;

	stack.push(cv::Point(x, y));

	Direction direction = Direction::NONE;

	cv::Point prev_point = cv::Point(x, y);

	bool reverse = false;

	while (!stack.empty())
	{
		cv::Point p = stack.top();
		stack.pop();

		if (img.at<uchar>(p.y, p.x) == 255)
		{
			contour.push_back(p);
			img.at<uchar>(p.y, p.x) = 0;
		}

		direction = getDirection(prev_point, p);

		std::vector<cv::Point> neighbours = getOrder(p, direction);

		for (const auto& n : neighbours)
		{
			if (isContour(n.x, n.y))
			{
				stack.push(n);
				break;
			}
		}

		prev_point = p;

		if (stack.empty())
		{
			if (!reverse)
			{
				reverse = true;

				cv::Point point(x, y);

				stack.push(point);
				std::reverse(contour.begin(), contour.end());
				prev_point = point;
			}
		}
	}
}

Direction ContoursGenerator::getDirection(cv::Point prev, cv::Point next)
{
	Direction direction = Direction::NONE;
	cv::Point dir = next - prev;
	if (dir.x != 0 && dir.y != 0)
	{
		if (dir.x > 0 && dir.y > 0)
		{
			direction = Direction::BOTTOM_RIGHT;
		}
		else if (dir.x > 0 && dir.y < 0)
		{
			direction = Direction::TOP_RIGHT;
		}
		else if (dir.x < 0 && dir.y > 0)
		{
			direction = Direction::BOTTOM_LEFT;
		}
		else if (dir.x < 0 && dir.y < 0)
		{
			direction = Direction::TOP_LEFT;
		}
	}
	else if (dir.x != 0)
	{
		if (dir.x > 0)
		{
			direction = Direction::RIGHT;
		}
		else
		{
			direction = Direction::LEFT;
		}
	}
	else if (dir.y != 0)
	{
		if (dir.y > 0)
		{
			direction = Direction::DOWN;
		}
		else
		{
			direction = Direction::TOP;
		}
	}
	return direction;
}

std::vector<cv::Point> ContoursGenerator::getOrder(cv::Point pt, Direction direction)
{
	switch (direction)
	{
	case Direction::TOP:
		return { cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1) };
	case Direction::TOP_RIGHT:
		return { cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y) };
	case Direction::RIGHT:
		return { cv::Point(pt.x + 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1) };
	case Direction::BOTTOM_RIGHT:
		return { cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x, pt.y - 1) };
	case Direction::DOWN:
		return { cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1) };
	case Direction::BOTTOM_LEFT:
		return { cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y) };
	case Direction::LEFT:
		return { cv::Point(pt.x - 1, pt.y), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x + 1, pt.y - 1) };
	case Direction::TOP_LEFT:
		return { cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y) };
	default:
		return { cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1) };
	}
}

void ContoursGenerator::findDepth(cv::Mat& img, std::vector<Contour>& contours)
{
	int width = img.cols;
	int height = img.rows;

	for (int k = 0; k < contours.size(); ++k)
	{
		Contour& c = contours[k];
		std::set<int> outers_left;
		std::set<int> outers_right;

		int y = c.points[0].y;
		uchar prev = 0;
		for (int x = 0; x < width; ++x)
		{
			uchar val = img.at<uchar>(y, x);

			if (val == prev)
			{
				continue;
			}

			if (val != 0 && val != c.value)
			{
				if (x < c.points[0].x)
				{
					if (outers_left.find(val) == outers_left.end())
					{
						outers_left.insert(val);
					}
					else
					{
						outers_left.erase(val);
					}
				}
				else
				{
					if (outers_right.find(val) == outers_right.end())
					{
						outers_right.insert(val);
					}
					else
					{
						outers_right.erase(val);
					}
				}
				prev = val;
			}

			std::set<int> outers;
			std::set_union(outers_left.begin(), outers_left.end(), outers_right.begin(), outers_right.end(), std::inserter(outers, outers.begin()));

			for (auto iter = outers.begin(); iter != outers.end();)
			{
				int id = *iter;
				Contour& outer = contours[id - 1];
				if ((outer.boundingRect & c.boundingRect) != c.boundingRect)
				{
					iter = outers.erase(iter);
				}
				else
				{
					++iter;
				}
			}

			c.depth = outers.size();
		}
	}
}

void ContoursGenerator::fillContours(cv::Mat& contoursMat, const std::vector<Contour>& contours, cv::Mat& drawing)
{
	int max_depth = 0;

	for (auto& c : contours)
	{
		if (c.depth > max_depth)
		{
			max_depth = c.depth;
		}
	}

	ColorScaler scaler(-1, max_depth, cv::Scalar(18, 185, 27), cv::Scalar(20, 20, 185));

	int width = contoursMat.cols;
	int height = contoursMat.rows;

	for (int k = 0; k < contours.size(); ++k)
	{
		const Contour& c = contours[k];
		cv::Point seed_point(-1, -1);

		// try point polygon test to find seed point
		for (auto& pt : c.points)
		{
			for (int m = -1; m <= 1; ++m)
			{
				for (int n = -1; n <= 1; ++n)
				{
					if (m == 0 && n == 0)
					{
						continue;
					}
					cv::Point p(pt.x + m, pt.y + n);
					if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height)
					{
						continue;
					}
					if (contoursMat.at<uchar>(p) == 0)
					{
						if (cv::pointPolygonTest(c.points, p, false) > 0)
						{
							seed_point = p;
							goto floodfill;
						}
					}
				}
			}
		}

	floodfill:
		if (seed_point.x != -1)
		{
			cv::Scalar color = scaler.getColor(c.depth);
			cv::floodFill(drawing, seed_point, color);
		}
	}

	// fill the holes
	cv::Scalar hole_color = scaler.getColor(-1);
	for(int i = 0; i < drawing.rows; ++i)
	{
		for(int j = 0; j < drawing.cols; ++j)
		{
			if (drawing.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0))
			{
				cv::floodFill(drawing, cv::Point(j, i), hole_color);
			}
		}
	}
}

void ContoursGenerator::saveImage(const QString& folderPath, const QPixmap& img, const QPixmap& mask)
{
	QDir().mkpath(folderPath + "/images");
	QDir().mkpath(folderPath + "/masks");

	QString baseName;
	int index = 0;
	QString imageFileName, maskFileName;
	do
	{
		baseName = QString::number(index);
		imageFileName = folderPath + "/images/" + baseName + ".png";
		maskFileName = folderPath + "/masks/" + baseName + ".png";
		index++;
	} while (QFile::exists(imageFileName) || QFile::exists(maskFileName));

	if (!img.save(imageFileName, "PNG"))
	{
		return;
	}
	if (!mask.save(maskFileName, "PNG"))
	{
		QFile::remove(imageFileName);
		return;
	}
}

GenerationParams ContoursGenerator::getUIParams()
{
	GenerationParams params{};
	if (ui)
	{
		params.height = ui->spinBox_Width->value();
		params.width = ui->spinBox_Height->value();
		params.Xmul = ui->doubleSpinBox_Xmul->value();
		params.Ymul = ui->doubleSpinBox_Ymul->value();
		params.mul = ui->spinBox_mul->value();
		params.generateWells = ui->groupBox_Wells->isChecked();
		params.numOfWells = ui->spinBox_Wells->value();
	}
	return params;
}

WellParams ContoursGenerator::getUIWellParams()
{
	WellParams params{};
	if (ui)
	{
		params.drawText = ui->groupBox_Wellname->isChecked();
		params.fontSize = ui->spinBox_wellFontSize->value();
		params.radius = ui->spinBox_WellRadius->value();
		params.offset = ui->spinBox_WellnameOffset->value();
		params.outline = ui->spinBox_WellOutline->value();
	}
	return params;
}

QPixmap utils::cvMat2Pixmap(const cv::Mat& input)
{
	QImage image;
	if (input.channels() == 3)
	{
		image = QImage((uchar*)input.data, input.cols, input.rows, input.step, QImage::Format_BGR888);
	}
	else if (input.channels() == 1)
	{
		image = QImage((uchar*)input.data, input.cols, input.rows, input.step, QImage::Format_Grayscale8);
	}
	QPixmap cpy = QPixmap::fromImage(image);
	return cpy;
}

template<int size>
inline void ContoursGenerator::setSize()
{
	ui->spinBox_Height->setValue(size);
	ui->spinBox_Width->setValue(size);
}

ColorScaler::ColorScaler(double min, double max, const cv::Scalar& minColor, const cv::Scalar& maxColor) :
	m_min(min)
	, m_max(max)
	, m_minColor(minColor)
	, m_maxColor(maxColor)
{
}

cv::Scalar ColorScaler::getColor(double value) const
{
	if (value < m_min)
	{
		return m_minColor;
	}
	if (value > m_max)
	{
		return m_maxColor;
	}
	double ratio = (value - m_min) / (m_max - m_min);
	cv::Scalar color;
	for (int i = 0; i < 3; ++i)
	{
		color[i] = m_minColor[i] + ratio * (m_maxColor[i] - m_minColor[i]);
	}
	return color;
}
