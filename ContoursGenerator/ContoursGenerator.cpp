#include "ContoursGenerator.h"
#include "PerlinNoise.hpp"
#include <omp.h>
#include "DrawOperations.h"
#include "RandomGenerator.h"
#include <qfile.h>
#include <qfiledialog.h>
#include <qtemporaryfile.h>
#include <quuid.h>

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

	QPixmap pixmap = utils::cvMat2Pixmap(gradInv);

	if (params.generateWells)
	{
		WellParams wellParams = getUIWellParams();
		for (int i = 0; i < params.numOfWells; ++i)
		{
			DrawOperations::drawRandomWell(pixmap, wellParams);
		}
	}

	QPixmap mask = utils::cvMat2Pixmap(grad);

	GenImg result{ pixmap, mask };
	return result;

}

void ContoursGenerator::saveImage(const QString& folderPath, const QPixmap& img, const QPixmap& mask)
{
	QDir().mkpath(folderPath + "/images");
	QDir().mkpath(folderPath + "/masks");

	QString baseName = QUuid::createUuid().toString().mid(1, 8);
	QString imageFileName = folderPath + "/images/" + baseName + "_image.png";
	QString maskFileName = folderPath + "/masks/" + baseName + "_mask.png";

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