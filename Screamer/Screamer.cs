using OpenCvSharp;
using OpenCvSharp.Dnn;

const string configFile = "Models/deploy.prototxt";
const string faceModel = "Models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
using var faceNet = CvDnn.ReadNetFromCaffe(configFile, faceModel);

var image = new Mat();
var detectFaces = true;
var run = true;

using var capture = new VideoCapture(0);
capture.Set(VideoCaptureProperties.Fps, 60);
capture.Set(VideoCaptureProperties.BufferSize, 1);
using var window = new Window("Screamer - Calibration Mode");

bool calibrationMode = true;
List<double> openMouthSamples = new List<double>();
List<double> closedMouthSamples = new List<double>();
double openThreshold = 0;
double closedThreshold = 0;
bool isCurrentlyCalibrating = false;
string calibrationState = "closed"; // "closed" or "open"

bool lastMouthState = false;
DateTime lastStateChange = DateTime.Now;
TimeSpan lastOpenDuration = TimeSpan.Zero;

double AnalyzeMouthRegion(Mat mouthRegion)
{
    if (mouthRegion == null || mouthRegion.Empty() || 
        mouthRegion.Width <= 0 || mouthRegion.Height <= 0)
    {
        return 0;
    }

    // Compare red/pink
    using var hsv = new Mat();
    Cv2.CvtColor(mouthRegion, hsv, ColorConversionCodes.BGR2HSV);
    
    var lowerRed1 = new Scalar(0, 40, 40);
    var upperRed1 = new Scalar(15, 255, 255);
    var lowerRed2 = new Scalar(165, 40, 40);
    var upperRed2 = new Scalar(180, 255, 255);
    
    using var mask1 = new Mat();
    using var mask2 = new Mat();
    using var redMask = new Mat();
    
    Cv2.InRange(hsv, lowerRed1, upperRed1, mask1);
    Cv2.InRange(hsv, lowerRed2, upperRed2, mask2);
    Cv2.BitwiseOr(mask1, mask2, redMask);
    
    var redPixels = Cv2.CountNonZero(redMask);
    var totalPixels = mouthRegion.Width * mouthRegion.Height;
    var redPercentage = (redPixels * 100.0) / totalPixels;

    // Darkness/contrast
    using var gray = new Mat();
    Cv2.CvtColor(mouthRegion, gray, ColorConversionCodes.BGR2GRAY);
    var avgBrightness = Cv2.Mean(gray).Val0;
    
    // Variance in texture
    using var mean = new Mat();
    using var stddev = new Mat();
    Cv2.MeanStdDev(gray, mean, stddev);
    var variance = stddev.At<double>(0) * stddev.At<double>(0);
    
    double combinedScore = redPercentage + (255 - avgBrightness) * 0.1 + variance * 0.05;
    
    return combinedScore;
}

bool DetectMouthOpen(double combinedScore)
{
    if (calibrationMode) return false;
    
    // Use calibrated thresholds with buffer for stability
    double midpoint = (openThreshold + closedThreshold) / 2;
    double buffer = Math.Abs(openThreshold - closedThreshold) * 0.2; // 20% buffer
    
    // Prevent flickering w/ hysteresis 
    if (lastMouthState) 
        return combinedScore > (midpoint - buffer);
    else 
        return combinedScore > (midpoint + buffer);
}

while (run)
{
    capture.Read(image);
    if (image.Empty()) break;

    var newSize = new Size(1920, 1080);
    using var frame = new Mat();
    Cv2.Resize(image, frame, newSize);

    // Calibration instructions
    if (calibrationMode)
    {
        string instructions = "";
        if (calibrationState == "closed")
        {
            instructions = "CALIBRATION: Keep mouth CLOSED and press SPACE (10 samples needed)";
            instructions += $" - Samples: {closedMouthSamples.Count}/10";
        }
        else if (calibrationState == "open")
        {
            instructions = "CALIBRATION: Keep mouth OPEN and press SPACE (10 samples needed)";
            instructions += $" - Samples: {openMouthSamples.Count}/10";
        }
        else
        {
            instructions = "CALIBRATION COMPLETE! Press 'c' to start detection";
        }
        
        var textSize = Cv2.GetTextSize(instructions, HersheyFonts.HersheySimplex, 0.8, 2, out int baseline);
        Cv2.Rectangle(frame, new Point(5, 5), new Point(textSize.Width + 15, textSize.Height + baseline + 15), Scalar.Black, -1);
        
        Cv2.PutText(frame, instructions, new Point(10, 30),
                   HersheyFonts.HersheySimplex, 0.8, Scalar.Yellow, 2);
    }

    TimeSpan currentOpenTime = TimeSpan.Zero;
    bool anyMouthDetected = false;

    if (detectFaces)
    {
        int frameHeight = frame.Rows;
        int frameWidth = frame.Cols;

        using var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(300, 300),
            new Scalar(104, 117, 123), false, false);
        faceNet.SetInput(blob, "data");

        using var detection = faceNet.Forward("detection_out");
        int numDetections = detection.Size(2);

        for (int i = 0; i < numDetections; i++)
        {
            float confidence = detection.At<float>(0, 0, i, 2);

            if (confidence > 0.7)
            {
                int x1 = (int)(detection.At<float>(0, 0, i, 3) * frameWidth);
                int y1 = (int)(detection.At<float>(0, 0, i, 4) * frameHeight);
                int x2 = (int)(detection.At<float>(0, 0, i, 5) * frameWidth);
                int y2 = (int)(detection.At<float>(0, 0, i, 6) * frameHeight);

                x1 = Math.Max(0, Math.Min(x1, frameWidth - 1));
                y1 = Math.Max(0, Math.Min(y1, frameHeight - 1));
                x2 = Math.Max(x1 + 1, Math.Min(x2, frameWidth));
                y2 = Math.Max(y1 + 1, Math.Min(y2, frameHeight));

                if (x2 > x1 && y2 > y1)
                {
                    using var faceImg = new Mat(frame, new Rect(x1, y1, x2 - x1, y2 - y1));
                    
                    int mouthY = (int)(faceImg.Height * 0.6);
                    int mouthHeight = (int)(faceImg.Height * 0.3);
                    int mouthX = (int)(faceImg.Width * 0.25);
                    int mouthWidth = (int)(faceImg.Width * 0.5);
                    
                    var mouthRect = new Rect(mouthX, mouthY, mouthWidth, mouthHeight);
                    using var mouthRegion = new Mat(faceImg, mouthRect);
                    
                    double combinedScore = AnalyzeMouthRegion(mouthRegion);
                    
                    Cv2.PutText(frame, $"Score: {combinedScore:F1}", new Point(x1, y2 + 20),
                               HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);

                    if (calibrationMode)
                    {
                        Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), Scalar.Yellow, 3);
                        var mouthGlobalRect = new Rect(x1 + mouthX, y1 + mouthY, mouthWidth, mouthHeight);
                        Cv2.Rectangle(frame, mouthGlobalRect, Scalar.Blue, 2);
                    }
                    else
                    {
                        bool mouthIsOpen = DetectMouthOpen(combinedScore);
                        anyMouthDetected = mouthIsOpen;

                        if (mouthIsOpen != lastMouthState)
                        {
                            if (lastMouthState)
                            {
                                lastOpenDuration = DateTime.Now - lastStateChange;
                            }
                            lastMouthState = mouthIsOpen;
                            lastStateChange = DateTime.Now;
                        }

                        if (mouthIsOpen)
                        {
                            currentOpenTime = DateTime.Now - lastStateChange;
                        }

                        var rectColor = mouthIsOpen ? Scalar.Green : Scalar.Red;
                        Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), rectColor, 3);

                        string status = mouthIsOpen ? "MOUTH OPEN" : "MOUTH CLOSED";
                        Cv2.PutText(frame, status, new Point(x1, y1 - 10),
                                   HersheyFonts.HersheySimplex, 0.7, rectColor, 2);
                    }
                    
                    if (isCurrentlyCalibrating)
                    {
                        if (calibrationState == "closed")
                        {
                            closedMouthSamples.Add(combinedScore);
                        }
                        else if (calibrationState == "open")
                        {
                            openMouthSamples.Add(combinedScore);
                        }
                        isCurrentlyCalibrating = false;
                    }
                }
            }
        }
    }

    // Status
    if (!calibrationMode)
    {
        string currentStatus = anyMouthDetected ? "MOUTH OPEN" : "MOUTH CLOSED";
        string timeInfo = anyMouthDetected ? 
            $"Current: {currentOpenTime.TotalSeconds:F1}s" :
            $"Last open: {lastOpenDuration.TotalSeconds:F1}s";
        
        var statusSize = Cv2.GetTextSize(currentStatus, HersheyFonts.HersheySimplex, 0.8, 2, out int statusBaseline);
        var timeSize = Cv2.GetTextSize(timeInfo, HersheyFonts.HersheySimplex, 0.6, 2, out int timeBaseline);
        int maxWidth = Math.Max(statusSize.Width, timeSize.Width);
        
        Cv2.Rectangle(frame, new Point(5, 5), new Point(maxWidth + 15, statusSize.Height + timeSize.Height + statusBaseline + timeBaseline + 25), Scalar.Black, -1);

        Cv2.PutText(frame, currentStatus, new Point(10, 30),
                   HersheyFonts.HersheySimplex, 0.8, 
                   anyMouthDetected ? Scalar.Green : Scalar.Red, 2);
        
        Cv2.PutText(frame, timeInfo, new Point(10, 60),
                   HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);
    }

    window.Image = frame;
    var key = Cv2.WaitKey(1);
    switch ((char)key)
    {
        case (char)27: // ESC
            run = false;
            break;
        case 'f':
            detectFaces = !detectFaces;
            break;
        case ' ':
            if (calibrationMode && (calibrationState == "closed" || calibrationState == "open"))
            {
                isCurrentlyCalibrating = true;
                
                // Check if we have enough samples
                if (calibrationState == "closed" && closedMouthSamples.Count >= 9)
                {
                    calibrationState = "open";
                    Console.WriteLine($"Closed mouth samples: Avg = {closedMouthSamples.Average():F2}");
                }
                else if (calibrationState == "open" && openMouthSamples.Count >= 9)
                {
                    // Calculate thresholds
                    closedThreshold = closedMouthSamples.Average();
                    openThreshold = openMouthSamples.Average();
                    
                    Console.WriteLine($"Calibration complete!");
                    Console.WriteLine($"Closed: {closedThreshold:F2}, Open: {openThreshold:F2}");
                    
                    calibrationState = "complete";
                }
            }
            break;
        case 'c':
            if (calibrationState == "complete")
            {
                calibrationMode = false;
                Console.WriteLine("Detection mode started!");
            }
            break;
    }
}