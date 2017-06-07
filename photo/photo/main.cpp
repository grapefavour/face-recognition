#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Point text_lb;
	CascadeClassifier cascade;
	cascade.load("lbpcascade_frontalface.xml");
	VideoCapture cap;
	cap.open(0);
	Mat frame;
	int pic_num = 1;
	Mat face;
	Mat gray,face_test;
	Mat warped,filtered;
	while (1)
	{
		cap >> frame;

		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, gray);

		cascade.detectMultiScale(gray, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0) {
				face = gray(faces[i]);
				text_lb = Point(faces[i].x, faces[i].y);
				rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);

				int leftX = cvRound(face.cols*0.16);
				int topX = cvRound(face.rows*0.26);
				int width = cvRound(face.cols*0.30);
				int hight = cvRound(face.rows*0.28);
				int right = cvRound(face.cols*0.54);
				//////Mat topLift = frame(Rect(leftX, topX, width, hight));
			//Mat topRight = frame(Rect(right, topX, width, hight));
				Point2f l(leftX + width*0.5 + faces[i].x, topX + hight*0.5 + faces[i].y);
				Point2f r(right + width*0.5 + faces[i].x, topX + hight*0.5 + faces[i].y);
				//circle(frame, l, 2, Scalar(0, 0, 255));
				//circle(frame, r, 2, Scalar(0, 0, 255));
				//rectangle(face, Rect(leftX, topX, width, hight), Scalar(0, 0, 255),1,8,0);
				//rectangle(face, Rect(right, topX, width, hight), Scalar(0, 0, 255), 1, 8, 0);
				Point2f eyesCenter;
				eyesCenter.x = (l.x + r.x) * 0.5f;
				eyesCenter.y = (l.y + r.y) * 0.5f;
				double dy = (r.y - l.y);
				double dx = (r.x - l.x);
				double len = sqrt(dx*dx + dy*dy);
				//转换弧度到角度  
				double angle = atan2(dy, dx) * 180.0 / CV_PI;
				//手测量表面左眼中心理想地应当在尺度化人脸图像的(0.16,0.14)的比例位置  
				const double DESIRED_LEFT_EYE_X = 0.16;
				const double DESIRED_LEFT_EYE_Y = 0.26;
				const double DESIRED_RIGHT_EYE_X = (1.0f - 0.16);
				// 得到尺度化的量，使用这些量，我们尺度化到我们想要的固定大小  
				const int DESIRED_FACE_WIDTH = 92;
				const int DESIRED_FACE_HEIGHT = 112;
				double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
				double scale = desiredLen * DESIRED_FACE_WIDTH / len;
				Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
				//移动人眼中心到一个想要的中心  
				double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
				double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y;
				rot_mat.at<double>(0, 2) += ex;
				rot_mat.at<double>(1, 2) += ey;
				//转换人脸图像到一个想要的角度，大小和位置。同时用默认的灰度值清除原来的背景图像  
				//warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(0,0,255));
				warpAffine(face, warped, rot_mat, Size(92, 112));

			}
		
		}

		if (faces.size() == 1)
		{
			resize(face,gray, Size(92, 112));
			equalizeHist(gray, face_test);
			filtered = Mat(Size(92, 112), CV_8U);
			bilateralFilter(face_test, filtered, 0, 20.0, 2.0);
			Mat mask = Mat(Size(92, 112), CV_8UC1, Scalar(255));
			double dw = 92;
			double dh = 112;
			Point faceCenter = Point(cvRound(dw * 0.5),
				cvRound(dh * 0.4));
			Size size = Size(cvRound(dw * 0.5), cvRound(dh * 0.8));
			ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);

			filtered.setTo(Scalar(128), mask);
			//Mat faceROI = frame_gray(faces[0]);
			//Mat myFace;
			//resize(faceROI, myFace, Size(92, 112));
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 255), 2, LINE_AA);
			//imshow("人脸", filtered);
			string filename = format("D:\\AA\\pic%d.jpg", pic_num);
			imwrite(filename, filtered);
			imshow("rr", filtered);
			cvMoveWindow("rr",0,0);
			waitKey(1000);
			destroyWindow(filename);
			pic_num++;
			if (pic_num == 11)
			{
				return 0;
			}
		}
		if (faces.size()==2) {
			putText(frame, "duoren", text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));

		}
		imshow("frame", frame);
		//imshow("人脸样本", filtered);
		waitKey(100);
	}
	return 0;
}
