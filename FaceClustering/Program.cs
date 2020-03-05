using System;
using System.Collections.Generic;
using DlibDotNet;

namespace FaceClustering
{
    class Program
    {
        const string inputFilePath = "./images/group.jpg";

        static void Main(string[] args)
        {
            Console.WriteLine("Loading detectors...");
            
            using(var detector = Dlib.GetFrontalFaceDetector())
            using(var predictor = ShapePredictor.Deserialize("shape_predictor_5_face_landmarks.dat"))
            using(var dnn = DlibDotNet.Dnn.LossMetric.Deserialize("dlib_face_recognition_resnet_model_v1.dat"))
            using(var img = Dlib.LoadImage<RgbPixel>(inputFilePath)) {
                var chips = new List<Matrix<RgbPixel>>();
                var faces = new List<Rectangle>();

                Console.WriteLine("Detecting faces...");

                foreach(var face in detector.Operator(img)) {
                    var shape = predictor.Detect(img, face);

                    var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                    var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                    var matrix = new Matrix<RgbPixel>(faceChip);
                    chips.Add(matrix);
                    faces.Add(face);
                }

                Console.WriteLine($"Found {chips.Count} faces in image");
                Console.WriteLine("Recognizing faces...");

                var descriptors = dnn.Operator(chips);
                var edges = new List<SamplePair>();

                for (uint i = 0; i < descriptors.Count; i++) 
                    for (var j = i; j < descriptors.Count; ++j) 
                        if (Dlib.Length(descriptors[i] - descriptors[j]) < 0.6)
                            edges.Add(new SamplePair(i, j));

                Dlib.ChineseWhispers(edges, 100, out var clusters, out var labels);
                Console.WriteLine($"Found {clusters} unique person(s) in the image");

                var palette = new RgbPixel[] {
                    new RgbPixel(0xe6, 0x19, 0x4b),
                    new RgbPixel(0xf5, 0x82, 0x31),
                    new RgbPixel(0xff, 0xe1, 0x19),
                    new RgbPixel(0xbc, 0xf6, 0x0c),
                    new RgbPixel(0x3c, 0xb4, 0x4b),
                    new RgbPixel(0x46, 0xf0, 0xf0),
                    new RgbPixel(0x43, 0x63, 0xd8),
                    new RgbPixel(0x91, 0x1e, 0xb4),
                    new RgbPixel(0xf0, 0x32, 0xe6),
                    new RgbPixel(0x80, 0x80, 0x80)
                };

                for (var i = 0; i < faces.Count; i++) {
                    Dlib.DrawRectangle(img, faces[i], color: palette[labels[i]], thickness: 4);
                }

                Dlib.SaveJpeg(img, "images/output.jpg");
            }
        }
    }
}
