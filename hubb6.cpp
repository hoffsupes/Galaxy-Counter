
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cstdarg>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "fstream"
// #include "conio.h"
#define PI 3.14

using namespace std;
using namespace cv;

Mat docleaning(Mat img, int n = 3)
{
    
//     Mat strel = Mat::zeros(3,3,CV_8UC1);
//     
//                              strel.at<uchar>(0,1) = 1;
//     strel.at<uchar>(1,0) = 1;strel.at<uchar>(1,1) = 1;strel.at<uchar>(1,2) = 1;
//                              strel.at<uchar>(2,1) = 1;
    imwrite("NR/bfsmoothing.jpg",img);
    Mat strel = getStructuringElement(MORPH_ELLIPSE,Size(n,n));   // equivalent to 3x3 above if n = 3, made n variable so that filtering can be applied to general images
    
    morphologyEx(img,img,MORPH_OPEN,strel);
    morphologyEx(img,img,MORPH_CLOSE,strel);		// morphological smoothing

    return img;
    
    
}

Mat do_redshift(Mat img)
{
vector<Mat> I;
split(img,I);

Mat q2,q3;
threshold(I[2] - I[0],q2,255,1,THRESH_OTSU);
threshold(I[2] - I[1],q3,255,1,THRESH_OTSU);

bitwise_and(q3,q2,q2);

Mat y;

return q2 + q3;
}

Mat do_edgethold(Mat Igray)
{
    Mat sobx,soby,ui;
    
    Sobel(Igray, sobx, CV_32F, 1,0);
    Sobel(Igray, soby, CV_32F, 0,1);
    
    cv::sqrt( sobx.mul(sobx) + soby.mul(soby) ,sobx); 
   
    sobx.convertTo(sobx,CV_8UC1); 
   imwrite("NR/sobel.jpg",sobx);
    double thold = threshold(sobx,ui,255,1,THRESH_OTSU);
    Mat cpy;
    Igray.copyTo(cpy,ui); 
    
    thold = threshold(cpy,ui,255,1,THRESH_OTSU); 
    Igray.convertTo(Igray,CV_32F);
    normalize(Igray,Igray,0,1,NORM_MINMAX);
    
    Mat yy = Igray > (thold / 255);  // optimum OTSU thresholding
    
//     imshow("canny",do_canny(Igray,55,3,9));
//      y();
//     thold = threshold(sobx,ui,255,1,THRESH_OTSU);  // ANDing in the mask for the edge thresholded image can remove "twin" galaxy formulations but generates a lot of noise
//     bitwise_and(ui,yy,yy);
    return yy;
}

Mat do_watershed(Mat I, Mat Igray, Mat yy, Mat conn_comp, int n = 3)        // watershed done here in a nonstandard way
{    
    Mat sobx,soby,ui;
    
    Sobel(Igray, sobx, CV_32F, 1,0);
    Sobel(Igray, soby, CV_32F, 0,1);
    
    cv::sqrt( sobx.mul(sobx) + soby.mul(soby) ,sobx); 
   
    sobx.convertTo(sobx,CV_8UC1); 
    int thold = threshold(sobx,ui,255,1,THRESH_OTSU);
    
    bitwise_and(yy,ui,yy);
    
    Mat dl1,lab;
    
    distanceTransform(yy,dl1,lab,DIST_L2,3); imwrite("NR/dist_tform.jpeg",dl1);
    double aa,bb; 
    minMaxLoc(dl1,&aa,&bb);

    Mat sfg = dl1 > 0.01*bb;                imwrite("NR/sfg.jpeg",sfg);
    Mat sbg;
    dilate(yy,sbg,Mat::ones(3,3,CV_8UC1),Point(-1,-1),2); // sure BG regions 
    sbg = sbg - sfg; // sureBG - sureFG 
    imwrite("NR/sbg.jpeg",sbg*255);
    int nolab = connectedComponents(sfg,conn_comp,8);
    conn_comp +=1;
    conn_comp.setTo(0,sbg);

    watershed(I,conn_comp);
    
    double a,b;
    minMaxLoc(conn_comp,&a,&b);
    Mat kk = Mat::zeros(I.size(),CV_8UC1);
    for(int i = 0; i < b; i++){kk += (conn_comp == i);}    
    
    erode(kk,kk,getStructuringElement(MORPH_ELLIPSE, Size(3,3)),Point(-1,-1),2);imwrite("NR/wshed.jpeg",kk);
    return kk;
}

void pross_img_alt(Mat I,Mat Iori,Mat& ui,Mat& conn_comp,int &nolabs, int arthold,double distt = 0.05)
{

//     Mat wat = do_watershed(Iori, I, ui,conn_comp);
//     Mat Y = Mat::zeros(ui.size(), ui.type());
//     
//     for(int i = 1; i < nolabs; i++)
//     {
//         Mat uu = (conn_comp == i);  
//         double ct = double( countNonZero(uu));
//         
//         if( !(countNonZero(uu) > arthold && (countNonZero(uu) < 0.9*conn_comp.rows*conn_comp.cols)) )
// 	{
//         continue;
// 	}
//         
//         Mat pnt;
//         findNonZero(uu,pnt);
//         
//         RotatedRect k = fitEllipse(pnt);
//         
//         Point2f p = k.center;
//         Size2f J = k.size;
//         float bb = J.width / 2;
//         float aa = J.height / 2;
//         
//         double ar = PI * aa * aa;
//         
//         if (ct / ar < 0.01)                              // some discrepency within the area of the object
//         {
//             bitwise_and(uu,wat,uu);
//         }
//         
//         Y += uu*255;
//     }
// 
//     Y.copyTo(ui);
//     nolabs = connectedComponents(ui,conn_comp,8);
    
    Mat sobx,soby,uki;
    Sobel(I, sobx, CV_32F, 1,0);
    Sobel(I, soby, CV_32F, 0,1);
    cv::sqrt( sobx.mul(sobx) + soby.mul(soby) ,sobx); 
    sobx.convertTo(sobx,CV_8UC1); 
    int thold = threshold(sobx,uki,255,1,THRESH_OTSU);

    Mat dl1,lab;
    distanceTransform(uki,dl1,lab,DIST_L2,3);
    double aa,bb; 
    minMaxLoc(dl1,&aa,&bb);

    ui = dl1 > distt*bb;
    nolabs = connectedComponents(ui,conn_comp,8);
}

Mat pross_img(Mat I,Mat Iori,Mat red, Mat watamask)
{
    Mat tmp,labs;imwrite("NR/red.jpeg",red*255);
    bitwise_and(red*255,watamask,tmp);
    imwrite("NR/red_border_mask.jpeg",tmp*255);
    erode(tmp*255,tmp,getStructuringElement(MORPH_ELLIPSE, Size(3,3)));
    
    Mat u = docleaning(do_edgethold(I));
    int nolabs = connectedComponents(u,labs,8); 
    Mat Y = Mat::zeros(red.size(),CV_8UC1);

    for(int i = 1; i < nolabs; i++)
    {
        Mat uu = (labs == i);   
        Mat yy;
        bitwise_and(uu,tmp,yy);
            
        if(countNonZero(yy))
        {
        
            Mat labs1;
            int l1 = connectedComponents(uu,labs1,8);
            int l2 = connectedComponents(yy,labs1,8);
            
            if(l1 == l2)
            {
            Y += uu*255;
            }
            else
            {
                imwrite("NR/uu.jpeg",uu*255);imwrite("NR/yy.jpeg",yy*255);
            Y += yy*255;
            }
        }
    }
            imwrite("NR/Y.jpeg",Y*255);
    return Y*255;
}

Mat do_der(Mat hislong)
{
    hislong = hislong.t();
    Mat k = Mat::zeros(hislong.size(),hislong.type());
    k.colRange(1,k.cols - 1) = hislong.colRange(0,k.cols - 2);
    return (hislong - k).t();
}

Mat gala_w(string sdata,Mat conn_comp,Mat I, Mat ori, int & count, int arthold, int labels, int thickness = 1)
{
    sdata.append("_res.txt");
    ofstream of;
    of.open(sdata.data(),ios::out | ios::ate);
        
    count = 0;
    int a,b,c;
    int maxar = 0;
    of << "\n\n DETECTIONS \n\n";
    
    cout <<"\n Processing Started!";
    for(int i = 0; i < labels; i++)
    {
        cout << ".";
	Mat l = (conn_comp == i);
	Mat pnts;
        
	if( !(countNonZero(l) > arthold && (countNonZero(l) < 0.9*conn_comp.rows*conn_comp.cols)) )
	{
        continue;
	}
        
        Mat zimg = Mat::zeros(conn_comp.size(), CV_8UC1);
                               
        findNonZero(l,pnts);
        RotatedRect k = fitEllipse(pnts);
        
        Point2f p = k.center;
        Size2f J = k.size;
        float bb = J.width / 2;
        float aa = J.height / 2;
        
        float ee = round(10*(1 - bb/aa));

//         circle(zimg, p , bb , Scalar(255),1);
        circle(zimg, p, aa, Scalar(255),1);         /// Scalar (255) using larger radius of the ellipse to get border around the 
        vector<Point> pn;
        findNonZero(zimg,pn);
        Mat cont;
        
        for(int I = 0; I< pn.size(); I++){cont.push_back( int(ori.at<uchar>(pn[I].y,pn[I].x)));}
        
        cont = do_der(do_der(cont));        
        int ringc = countNonZero(cont);
        
//         Point2f vt[4];
//         k.points(vt);
//         
//         Mat zim = Mat::zeros(zimg.size(),zimg.type());
// 
//         line(zim,vt[0],vt[2],Scalar(255),1);
/*        vector<Point> ppl;
        findNonZero(zim,ppl); Mat cnn;
        
        for(int I = 0; I< ppl.size(); I++){cnn.push_back( int(ori.at<uchar>(ppl[I].y,ppl[I].x)));}    */  
        
//         cout << "\n\n CNN::::" << cnn <<"\n\n";
        
        
        if (ee == 0 && ringc == 4)          // counting the ZCR's around an object
        {
            continue;                   // stars have a cross shape on them
        }
        
        count++;
        
//         cout << "\n\n DETECTION ::: " << count << "\n\n"; 
        of << "\n"<< count << ". \t"; 
        
        if(ee == 0)                 // eccentricity zero (stars filtered out)
        {
//          cout << "\n Galaxy Type :: E0\n";
         of << "\t Galaxy Type :: E0    Center:: " << p;
         
         putText(I,"elliptical",p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255),1);;
        
        }

        else if(ee != 0 && ringc > 2)           // eccentricity not zero but count of ZCR's around an object greater than 2
        {
            
        Mat yy;
        //circle(zimg, p , bb , Scalar(255),-1);
        ellipse(zimg,k,Scalar(255),-1);
        ori.copyTo(yy,zimg);
        float cent_val = float(ori.at<uchar>(p.y,p.x));
        float avg_v = float(sum(yy)[0] / countNonZero(zimg));
        float page = 100* (avg_v/cent_val);
        
         
        if(page >= 60)                               // discrepency not present in case of an elliptical galaxy
        {
//          cout << "\n Galaxy Type:: E" << ee << "\n";
         of << "\t Galaxy Type:: E" << ee << "    Center:: " << p << "\n";
         
         putText(I,"elliptical",p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255),1);
        
        }
        else if(page < 60 && page >=30 )           // average intensity varies far too much from the center intensity (Spiral)
        {putText(I,"spiral",p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255),1);
        
//          cout << "\n Galaxy Type:: Spiral\n ";    
         of << "\t Galaxy Type:: Spiral     Center:: " << p;;    
            
        }
        else                                    // average intensity varies WAY too much from the central intensity of the object (Irregular)
        {
        putText(I,"Irregular",p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255),1);
        
//          cout << "\n Galaxy Type:: Spiral\n ";    
         of << "\t Galaxy Type:: Irregular     Center:: " << p;;    
            
        }
            
        }
        else if(ee!= 0 && ringc <= 2)
        {
            putText(I,"elliptical",p,FONT_HERSHEY_SIMPLEX,0.5,Scalar(255),1);
        
//          cout << "\n Galaxy Type:: E" << ee << "\n";      
         of << "\t Galaxy Type:: E" << ee << "    Center:: " << p << "\n";  
        }

//	dilate(l,k,Mat());
//	l = k - l;
	a = rand() % 256;
	b = rand() % 256;
	c = rand() % 256;
        

        ellipse(I,fitEllipse(pnts),Scalar(a,b,c),thickness);
//	pnts.release();


    }
    cout <<"\n Processing Done!";

    of.close();

return I;
}

void get_gala(Mat I,string sdata, int wshed = 0,int alt = 0, int arthold = 70,double distt = 0.1)
{
cout <<"\n Processing:: "<< sdata <<"\n";

if(I.empty())
{
    cout  << "\n File Reading error!! Empty File read!! \n";
    exit(1);
}

    int cnt;
    Mat Igray,ui,conn_comp;
    vector<Point> pnts;
    int layer;
    double k,l;
    Point p;
    
    
    if(I.channels() >= 3)
    {
    cvtColor(I,Igray,CV_BGR2GRAY);
    }
    else
    {
        Igray = I;
    }
    imwrite("NR/gray.jpg",Igray);
    
//     blur(Igray,Igray,Size(3,3));
    ui = docleaning(do_edgethold(Igray));
    imwrite("NR/init_mask.jpg",ui);
    int nolab = connectedComponents(ui,conn_comp,8);
   
    if(wshed)       // watershed mode is on when the user attempts / wants to use it for a specific image (WARNING: It may not be applicable for all images)
    {               // not used in a typical manner, instead used for getting borders which would actually cut through falsely connected objects
    Mat red = docleaning(do_redshift(I));
    Mat watmask = do_watershed(I, Igray,ui, conn_comp,3);         // taking these steps will get one results for one specific case
    bitwise_and(ui,red,ui);                                        // but it may not generalize over all results
    ui = pross_img(Igray,I,red,watmask);
//     imshow("asasasas",ui);
//     waitKey();
    nolab = connectedComponents(ui,conn_comp,8);
    }
    I = gala_w(sdata,conn_comp,I, Igray, cnt, arthold,nolab);
    imwrite("NR/final.jpg",I);
    string g = sdata;
    g.append("_res_.jpeg");
    imwrite(g.data(),I);
    
    cout << "\n Galaxy count::: "<< cnt <<"\n";
    
}

int main(int argc, char* argv[])
{
        int wshed = 0;                      // point to note here, watershed is used in a non-standard way to cut through any connected objects
        int alt = 0;
        int wait = 0;                       // change value to 1 to activate debug mode
        
        if (argc < 2)
        {
            cout << "\nRUNNING IN NORMAL MODE\n";
            wshed = 0;
        }
        else if(argc == 2)
        {
            cout << "\nRUNNING IN WATERSHED MODE\n";
            cout << "\nWARNING: MAY NOT BE SUITABLE FOR ALL KINDS OF IMAGES\n";
            wshed = 1;
        }
        else
        {
            cout << "\n PLEASE ENTER ANY INPUT in the command line for WATERSHED MODE and no optional commands for NORMAL mode.\n";
            cout << "\n Usage g++ -o gala_p -g hubb6.cpp `pkg-config --cflags --libs opencv`;./gala_p; # This is a NORMAL mode run \n";
            cout << "\n Usage g++ -o gala_p -g hubb6.cpp `pkg-config --cflags --libs opencv`;./gala_p 1; # This is a WATERSHED mode run  \n";
            cout << "\n Usage g++ -o gala_p -g hubb6.cpp `pkg-config --cflags --libs opencv`;./gala_p 100; # This is also a WATERSHED mode run  \n";
            cout << "\n Usage g++ -o gala_p -g hubb6.cpp `pkg-config --cflags --libs opencv`;./gala_p 9999; # This is also a WATERSHED mode run  \n";
            cout << "\n Usage g++ -o gala_p -g hubb6.cpp `pkg-config --cflags --libs opencv`;./gala_p 9999 100; # Incorrect Syntax and Usage!! Follow Correct Syntax!!   \n";
            cout << "\n If you're on Windows you're on your own, please fix command line parameters as needed! \n";            
            cout << "\n Currently operating in NORMAL mode!! Please use execution options more carefully next time, read the README before proceeding! \n";
            
        }   
        
//         get_gala(imread("DB/9.jpg"),string("NNK/"),1);
        
        cout << "\n Welcome to Gala_Count v1.0! Please read the README!! If you feel that the result is not satisfactory then please run in watershed mode! \n Please SAVE your results before proceeding, WATERSHED mode will overwrite ALL the images!\n WARNING: WATERSHED MODE MAY NOT BE SUITABLE FOR ALL KINDS OF IMAGES!! PROCEED WITH CAUTION!!  \n";
        
	DIR *pdir = NULL;
        string dirr("DB/");
        string ree("RES1/");
	pdir = opendir(dirr.data()); 
        
	if(pdir ==  NULL){cout<<"\nFile Directory Inaccessible!!\n";exit(1);}	
	struct dirent *pent = NULL;
	        
	while( pent = readdir(pdir) )
	{
            
            if(pent == NULL){cout<<"\nCheck Your files / you may not have permission to access this folder. \n"; exit(1);}
            string * filnam = new string( pent->d_name );
            if(filnam->at(0) == '.') { continue; }
            
            string nam = dirr;
            nam.append(pent->d_name);
            string res = ree;
            res.append(pent->d_name);           
            
            get_gala(imread(nam.data()),res,wshed);
            
            if(wait)
            {
            int ch = cin.get();  
            }
	}

    cout <<"\n Exiting Database Directory ....... DONE!!\n\n\n";
    closedir(pdir);
    cout << "\n Thank you for using Gala Count v1.0! Have a nice day! Use nova.astronomy.com to validate results if needed! \n";
    return 1;
}
