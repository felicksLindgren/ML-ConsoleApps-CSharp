using System;
using System.IO;
using System.Linq;
using CNTK;
using CNTKUtil;
using Microsoft.ML;
using SpamDetection.Models;
using XPlot.Plotly;

namespace SpamDetection
{
    class Program
    {
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "data/spam.tsv"); 
        static void Main(string[] args)
        {
            var context = new MLContext();

            Console.WriteLine("Loading data...");

            var data = context.Data.LoadFromTextFile<SpamClass>(
                path: DataPath,
                hasHeader: true, 
                separatorChar: ','
            );

            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.3);

            Console.WriteLine("Featurizing text...");
            var pipeline = context.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SpamClass.Message)
            );

            var model = pipeline.Fit(partitions.TrainSet);

            var trainingData = model.Transform(partitions.TrainSet);
            var testingData = model.Transform(partitions.TestSet);

            var training = context.Data.CreateEnumerable<ProcessedData>(trainingData, reuseRowObject: false);
            var testing = context.Data.CreateEnumerable<ProcessedData>(testingData, reuseRowObject: false);

            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();
            var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
            var testing_labels = testing.Select(v => v.GetLabel()).ToArray();

            var nodeCount = training_data.First().Length;
            Console.WriteLine($"Embedded text data in {nodeCount} dimensions");

            var features = NetUtil.Var(new int[] { nodeCount }, DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            var network = features
                .Dense(16, CNTKLib.ReLU)
                .Dense(16, CNTKLib.ReLU)
                .Dense(1, CNTKLib.Sigmoid)
                .ToNetwork();

            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            var lossFunc = CNTKLib.BinaryCrossEntropy(network.Output, labels);
            var errorFunc = NetUtil.BinaryClassificationError(network.Output, labels);

            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: true
            );

            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
            var evaluator = network.GetEvaluator(errorFunc);

            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");
            
            var maxEpochs = 10;
            var batchSize = 64;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var testingError = new double[maxEpochs];
            var batchCount = 0;

            for(int epoch = 0; epoch < maxEpochs; epoch++) {
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                training_data.Index().Shuffle().Batch(batchSize, (indices, begin, end) => {
                    var featureBatch = features.GetBatch(training_data, indices, begin, end);
                    var labelBatch = (labels.GetBatch(training_labels, indices, begin, end));

                    var result = trainer.TrainBatch(
                        new[] {
                            (features, featureBatch),
                            (labels, labelBatch)
                        }, false
                    );

                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    batchCount++;
                });

                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                testingError[epoch] = 0.0;
                batchCount = 0;
                testing_data.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch for testing
                    var featureBatch = features.GetBatch(testing_data, begin, end);
                    var labelBatch = labels.GetBatch(testing_labels, begin, end);

                    // test the network on the batch
                    testingError[epoch] += evaluator.TestBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                        }
                    );
                    batchCount++;
                });
                testingError[epoch] /= batchCount;
                Console.WriteLine($"{testingError[epoch]:F3}");
            }

            var chart = Chart.Plot(
                new [] 
                {
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = trainingError,
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = testingError,
                        name = "testing",
                        mode = "lines+markers"
                    }
                }
            );
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Classification error");
            chart.WithTitle("Spam Detection");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());

            var finalError = testingError[maxEpochs-1];
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");
        }
    }
}
