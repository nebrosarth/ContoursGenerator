#include "ContoursGenerator.h"
#include "PerlinNoise.hpp"
#include <omp.h>

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

void ContoursGenerator::initConnections()
{
	connect(ui->pushButton_Generate, &QPushButton::pressed, this, &ContoursGenerator::generateImage);
}

void ContoursGenerator::generateImage()
{
	GenerationParams params = getUIParams();

	static std::random_device dev;
	static std::mt19937 rng(dev());
	static std::uniform_int_distribution<std::mt19937::result_type> dist_int(1, 100);

	auto now = std::chrono::system_clock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
	auto value = now_ms.time_since_epoch();
	long long time = value.count();

	const siv::PerlinNoise::seed_type seed = time;
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
	grad = cv::Scalar(255) - grad;

	QPixmap pixmap = utils::cvMat2Pixmap(grad);
	ui->label_Image->setPixmap(pixmap);
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
