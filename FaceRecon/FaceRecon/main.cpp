#include<opencv2\opencv.hpp>  
#include<opencv2\face.hpp>
#include<iostream>  

using namespace std;
using namespace cv;
using namespace cv::face;

int main()
{
	int tt = 0;
	float y1 = 0;
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat edges;
	Mat gray;

	CascadeClassifier cascade;
	CascadeClassifier eyes_cascade;
	CascadeClassifier smile_cascade;
	CascadeClassifier mouth_cascade;
	CascadeClassifier nose_cascade;
	bool stop = false;
	eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml");
	cascade.load("haarcascade_frontalface_alt.xml");
	smile_cascade.load("haarcascade_smile.xml");
	mouth_cascade.load("haarcascade_mcs_mouth.xml");
	nose_cascade.load("haarcascade_mcs_nose.xml");

	Ptr<FaceRecognizer> model= createEigenFaceRecognizer();
	model->load("MyFacePCAModel.xml");
	Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();
	model1->load("MyFaceFisherModel.xml");
	//model1->save("MyFaceFisherModel.xml");

	Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();
	model2->load("MyFaceLBPHModel.xml");
	while (1)
	{
		cap >> frame;
		
		
		Point2f eyesCenter;
		
		//建立用于存放人脸的向量容器  
		vector<Rect> faces;

		cvtColor(frame, gray, CV_BGR2GRAY);
		//改变图像大小，使用双线性差值  
		//resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);  
		//变换后的图像进行直方图均值化处理  
		equalizeHist(gray, gray);

		cascade.detectMultiScale(gray, faces,
			1.1, 3, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT  
			//|CV_HAAR_DO_ROUGH_SEARCH  
			| CV_HAAR_SCALE_IMAGE,
			Size(40, 40));
		Mat warped;
		Mat face;
		Point text_lb;
		Point text_lb2;
		Mat face_test;
		Mat face_test2;

		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				face = gray(faces[i]);
				tt++;
				text_lb = Point(faces[i].x, faces[i].y);
				text_lb2 = Point(180, 180);

				rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);
				Mat faceROI = gray(faces[i]);
				std::vector<Rect> eyes;

				//-- In each face, detect eyes
				eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));


				int leftX = cvRound(face.cols*0.16);
				int topX = cvRound(face.rows*0.26);
				int width = cvRound(face.cols*0.30);
				int hight = cvRound(face.rows*0.28);
				int right = cvRound(face.cols*0.54);
				Mat topLift = frame(Rect(leftX, topX, width, hight));
				Mat topRight = frame(Rect(right, topX, width, hight));
				Point2d l(leftX + width*0.5 + faces[i].x, topX + hight*0.5 + faces[i].y);
				Point2d r(right + width*0.5 + faces[i].x, topX + hight*0.5 + faces[i].y);
				//circle(frame,l,2, Scalar(0, 0, 255));
				//circle(frame, r, 2, Scalar(0, 0, 255));
				//rectangle(face, Rect(leftX, topX, width, hight), Scalar(0, 0, 255),1,8,0);
				//rectangle(face, Rect(right, topX, width, hight), Scalar(0, 0, 255), 1, 8, 0);

				eyesCenter.x = (l.x + r.x) * 0.5f;
				eyesCenter.y = (l.y + r.y) * 0.5f;
				//circle(frame, eyesCenter, 2, Scalar(0, 0, 255));
				//if (tt == 1) {
				//y1 = eyesCenter.y;
					//cout << tt << endl;
				//}
			//	if (y1 != 0) {
				//	if (eyesCenter.y - y1 > 0) {
						//putText(frame, "node", text_lb2, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
						//cout << "点头" << endl;

					//}
				//}

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
				//rot_mat.at<double>(0, 2) += ex;
				//rot_mat.at<double>(1, 2) += ey;
				//转换人脸图像到一个想要的角度，大小和位置。同时用默认的灰度值清除原来的背景图像  
				 //warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(0,0,255));
				//warpAffine(face, warped, rot_mat, Size(92, 112));


				int z = 0;
				for (size_t j = 0; j < eyes.size(); j++)//检测眼睛
				{
					if (eyes[i].height > 3 && faces[i].width > 3) {
						Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);

						int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
						circle(frame, eye_center, 5, Scalar(255, 0, 0), 0, 8, 0);
						z = 1;
						putText(frame, "open", Point(150, 150), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
					}

				}
				if (z == 0) {
					putText(frame, "close", Point(150, 150), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				}


				//std::vector<Rect> smile;

				//smile_cascade.detectMultiScale(faceROI, smile, 1.1, 55, CASCADE_SCALE_IMAGE);

				//for (size_t j = 0; j < smile.size(); j++)
				//{
					//Rect rect(faces[i].x + smile[j].x, faces[i].y + smile[j].y, smile[j].width, smile[j].height);
					//rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);
					//putText(frame, "smile", Point(rect.x*0.5, rect.y*0.5), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				//}
				//mouth_cascade.detectMultiScale(faceROI, smile, 1.1, 55, CASCADE_SCALE_IMAGE);

				//for (size_t j = 0; j < smile.size(); j++)
				//{
			//	Rect rect(faces[i].x + smile[j].x, faces[i].y + smile[j].y, smile[j].width, smile[j].height);
				//rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);
				//if (smile[i].height > 3 && smile[i].width >3) {
					//putText(frame, "smile", Point(180, 180), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				//}
				//}
			}
				Mat face_test3;
				Mat filtered;
				int predictPCA = -1;
				double confidence = 0.0;

				if (face.rows >= 120)
				{
					resize(face, face_test, Size(92, 112));
					equalizeHist(face_test, face_test3);
					filtered = Mat(Size(92, 112), CV_8U);
					bilateralFilter(face_test3, filtered, 0, 20.0, 2.0);
					Mat mask = Mat(Size(92, 112), CV_8UC1, Scalar(255));
					double dw = 92;
					double dh = 112;
					Point faceCenter = Point(cvRound(dw * 0.5),
						cvRound(dh * 0.4));
					Size size = Size(cvRound(dw * 0.5), cvRound(dh * 0.8));
					ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);

					filtered.setTo(Scalar(128), mask);
				}
				if (!filtered.empty())
				{
					//double  current _threshold = model->getDouble("threshold");
					//这行将阈值设置为0.0：
					//model ->set （“threshold” ， 0.0 ）;
					predictPCA = model2->predict(filtered);
					//model->predict(filtered, predictPCA, confidence);

					//测试图像应该是灰度图  
					namedWindow("face1");
					imshow("face1", filtered);

				}



				//Mat face_test_gray;  
				//cvtColor(face_test, face_test_gray, CV_BGR2GRAY);  


				//CvText text()
				cout << predictPCA << endl;
				cout << confidence << endl;
				if (predictPCA == -1)
				{

					string name = "???";
					putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				}
				if (predictPCA == 2)
				{
					string name = "favour";
					putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				}
				if (predictPCA == 4)
				{
					string name2 = "ZhiGuo Li";
					putText(frame, name2, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				}
				if (predictPCA == 5)
				{
					string name1 = "jack";
					putText(frame, name1, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				}
			

		}
		
	
		namedWindow("face");
		imshow("face", frame);
		
		waitKey(100);
	}

	return 0;
}