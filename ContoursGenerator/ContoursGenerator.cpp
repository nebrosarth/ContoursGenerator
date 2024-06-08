#include "ContoursGenerator.h"
#include "DrawOperations.h"
#include <qfile.h>
#include <qfiledialog.h>
#include <opencv2/ximgproc.hpp>
#include "ContoursOperations.h"

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

GenImg ContoursGenerator::generateImage()
{
	GenerationParams params = getUIParams();

	cv::Mat isolines = ContoursOperations::generateIsolines(params);

	cv::Mat mask = cv::Scalar(255) - isolines;

	// apply thinning
	cv::Mat thinned;
	cv::ximgproc::thinning(mask, thinned, cv::ximgproc::THINNING_GUOHALL);

	// crop by 1 pixel
	cv::Rect cropRect(1, 1, thinned.cols - 2, thinned.rows - 2);
	thinned = thinned(cropRect);

	std::vector<Contour> contours;

	// Find contours
	ContoursOperations::findContours(thinned, contours);

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

	// Find depth
	ContoursOperations::findDepth(contours_mat, contours);

	// Depth mat
	cv::Mat depthMat = cv::Mat::zeros(thinned.size(), CV_8UC1);
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
	ContoursOperations::fillContours(contours_mat, contours, drawing);

	// Inpaint contours on drawing

	cv::Mat maskInpaint = cv::Mat::zeros(thinned.size(), CV_8UC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		const Contour& c = contours[i];
		for (size_t j = 0; j < c.points.size(); ++j)
		{
			maskInpaint.at<uchar>(c.points[j]) = 255;
		}
	}

	// Inpaint
	cv::inpaint(drawing, maskInpaint, drawing, 3, cv::INPAINT_TELEA);

	QPixmap pixIso = utils::cvMat2Pixmap(drawing);

	QFont font;
	// Draw contours
	for (const auto& contour : contours)
	{
		DrawOperations::drawContourValues(pixIso, contour, QColor(Qt::black), font);
	}

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

cv::Mat utils::QPixmap2cvMat(const QPixmap& in, bool grayscale)
{
	QImage im = in.toImage();
	if (grayscale)
	{
		im.convertTo(QImage::Format_Grayscale8);
		return cv::Mat(im.height(), im.width(), CV_8UC1, const_cast<uchar*>(im.bits()), im.bytesPerLine()).clone();
	}
	else
	{
		im.convertTo(QImage::Format_BGR888);
		return cv::Mat(im.height(), im.width(), CV_8UC3, const_cast<uchar*>(im.bits()), im.bytesPerLine()).clone();
	}
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
