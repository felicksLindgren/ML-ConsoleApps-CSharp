using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using CNTKUtil;

namespace GanDemo
{
    class Program
    {
        const string FileName = "x_channels_first_8_5.bin";

        #region the number of latent dimensions to use in the generator

        static readonly int latentDimensions = 32;

        #endregion

        #region the image dimensions and number of color channels

        static readonly int imageHeight = 32;
        static readonly int imageWidth = 32;
        static readonly int channels = 3;

        #endregion 

        static void Main(string[] args)
        {
            #region Unpack archive

            if(!File.Exists(FileName)) {
                Console.WriteLine("Unpacking archive...");
                ZipFile.ExtractToDirectory("frog_pictures.zip", ".");
            }

            #endregion

            #region Load training and test data

            Console.WriteLine("Loading data files...");
            var trainingData = DataUtil.LoadBinary<float>(FileName, 5000, channels * imageWidth * imageHeight);

            #endregion

            #region Create generator

            var generatorVar = CNTK.Variable.InputVariable(new int[] { latentDimensions }, CNTK.DataType.Float, name: "generator_input");
            var generator = generatorVar
                .Dense(128 * 16 * 16, v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Reshape(new int[] { 16, 16, 128 })
                .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .ConvolutionTranspose(
                    filterShape: new int[] { 4, 4 },
                    numberOfFilters: 256,
                    strides: new int[] { 2, 2 },
                    outputShape: new int[] { 32, 32 },
                    padding: true, 
                    activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1)
                )
                .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Convolution2D(256, new int[] { 5, 5 }, padding: true, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Convolution2D(channels, new int[] { 7, 7 }, padding: true, activation: CNTK.CNTKLib.Tanh)
                .ToNetwork();

            #endregion 
            
            #region Create discriminator

            var discriminatorVar = CNTK.Variable.InputVariable(new int[] { imageWidth, imageHeight, channels }, CNTK.DataType.Float, name: "discriminator_imput");
            var discriminator = discriminatorVar
                .Convolution2D(128, new int[] { 3, 3 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Convolution2D(128, new int[] { 4, 4 }, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Convolution2D(128, new int[] { 4, 4 }, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Convolution2D(128, new int[] { 4, 4 }, strides: new int[] { 2, 2 }, activation: v => CNTK.CNTKLib.LeakyReLU(v, 0.1))
                .Dropout(0.4)
                .Dense(1, CNTK.CNTKLib.Sigmoid)
                .ToNetwork();

            #endregion

            var gan = Gan.CreateGan(generator, discriminator);
            var labelVar = CNTK.Variable.InputVariable(shape: new CNTK.NDShape(0), dataType: CNTK.DataType.Float, name: "label_var");

            var discriminatorLoss = CNTK.CNTKLib.BinaryCrossEntropy(discriminator, labelVar);
            var ganLoss = CNTK.CNTKLib.BinaryCrossEntropy(gan, labelVar);

            var discriminatorLearner = discriminator.GetAdaDeltaLearner(1);
            var ganLearner = gan.GetAdaDeltaLearner(1);

            var discriminatorTrainer = discriminator.GetTrainer(discriminatorLearner, discriminatorLoss, discriminatorLoss);
            var ganTrainer = gan.GetTrainer(ganLearner, ganLoss, ganLoss);

            var outputFolder = "images";
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);
            
            Console.WriteLine("Training Gan...");
            var numEpochs = 100_000;
            var batchSize = 8;
            var start = 0;

            for (var epoch = 0; epoch < numEpochs; epoch++)
            {
                var generatedImages = Gan.GenerateImages(generator, batchSize, latentDimensions);

                start = Math.Min(start, trainingData.Length - batchSize);
                var batch = Gan.GetTrainingBatch(discriminatorVar, generatedImages, trainingData, batchSize, start);
                start += batchSize;

                if (start >= trainingData.Length)
                    start = 0;
                
                var discriminatorResult = discriminatorTrainer.TrainBatch(
                    new[] {
                        (discriminator.Arguments[0], batch.featureBatch),
                        (labelVar, batch.labelBatch)
                    }, true
                );

                var misleadingBatch = Gan.GetMisleadingBatch(gan, batchSize, latentDimensions);
                
                var ganResult = ganTrainer.TrainBatch(
                    new[] {
                        (gan.Arguments[0], misleadingBatch.featureBatch),
                        (labelVar, misleadingBatch.labelBatch)
                    }, true
                );

                if (epoch % 100 == 0) {
                    Console.WriteLine($"Epoch: {epoch}, Discriminator loss: {discriminatorResult.Loss}, Gan loss: {ganResult.Loss}");
                    var path = Path.Combine(outputFolder, $"generated_frog_{epoch}.png");
                    Gan.SaveImage(generatedImages[0].ToArray(), imageWidth, imageHeight, path);
                }
            }
        }
    }
}
