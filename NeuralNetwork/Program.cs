using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;

using NeuralNetwork.Network;
using NeuralNetwork.Learning;
using NeuralNetwork.Activation;
using NeuralNetwork.Utils.Extensions;

using DatasetUtility;
using MNIST_Reader;
using System.Diagnostics;

namespace NeuralNetwork
{
    static class Program
    {

        #region Monk test
       
        static double[] Encode(int value, byte encode)
        {
            double[] encoded = new double[encode];
            encoded[value - 1] = 1;
            return encoded;
        }

        static double[] EncodingInput(string[] inputs)
        {
            List<double> input = new List<double>();
            byte[] encoding = { 3, 3, 2, 3, 4, 2 };

            for (int i = 0; i < inputs.Length; i++)
            {
                input.InsertRange(input.Count, Encode(Int32.Parse(inputs[i]), encoding[i]));
            }
            return input.ToArray();
        }

        static Dataset ReadMonkDataset(StringReader stream)
        {
            try
            {
                Dataset dataset = new Dataset();
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ' ' };
                int c = 0;


                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    double[] output = { Double.Parse(strings[0]) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 1).ToArray();
                    double[] input = EncodingInput(strings);

                    dataset.Add(new Sample(input, output));

                    trainingExample = stream.ReadLine();
                }

                return dataset;


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static double RunMonkTest(NeuralNet net, Dataset testSet)
        {
            double successRatio = 0.0;
            int numOfExamples = testSet.Size;

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                double netOutput = net.Output.At(0);
                double roundOutput = Math.Round(netOutput);
                double expected = sample.Output.At(0);

                successRatio += (expected == roundOutput) ? 1 : 0;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        private static void TestMonk()
        {
            Dataset trainSet;
            Dataset testSet;

            #region Testing Monk Dataset 1
            
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0, 0, 0.0001);


            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_1_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_1_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);
                
                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 1");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.Write("Train the network...");
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.5, 0, 0.01, 0);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 2
           
            layerSize = new[] { 4, 1 };
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagationTrainer(net, 0, 0, 0.0001);


            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_2_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_2_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 2");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.Write("Train the network...");
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.5, 0, 0.01, 0);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 3
            layerSize = new[] { 10, 1 };
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagationTrainer(net, 0, 0, 0.0001);


            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_3_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_3_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 3");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.Write("Train the network...");
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.5, 0, 0.01, 0);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion
        }
       
        #endregion

        #region MNIST test

        private static Dataset ReadMNIST(MemoryStream imageStream, MemoryStream labelStream)
        {
            Dataset mnist = new Dataset();

            MNIST_Image_Reader imageReader = new MNIST_Image_Reader(imageStream);
            MNIST_Label_Reader labelReader = new MNIST_Label_Reader(labelStream);

            double[] inputImage = imageReader.ReadNext();
            double[] outputLabel = labelReader.ReadNext();

            Console.WriteLine("Start read images");
            while (inputImage != null && outputLabel != null)
            {
                mnist.Add(new Sample(inputImage, outputLabel));

                inputImage = imageReader.ReadNext();
                outputLabel = labelReader.ReadNext();
            }
            Console.WriteLine("End read images");

            return mnist;
        }

        private static double RunMNISTTest(NeuralNet net, Dataset testSet)
        {
            int numOfExamples = testSet.Size;
            double successRatio = 0;

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                double[] netOutput = net.Output.ToArray();
                double[] expected = sample.Output.ToArray();

                int expectedLabel = Array.IndexOf(expected, (byte) 1);
                int outputLabel = MaxIndex(netOutput);

                if (expectedLabel == outputLabel)
                    successRatio++;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        private static int MaxIndex(double[] array)
        {
            double max = Double.MinValue;
            int maxIndex = -1;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] > max)
                {
                    max = array[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private static void TestMNIST()
        {
            Dataset mnistTrainSet;
            Dataset mnistTestSet;

            int[] layerSize = { 100, 10 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(784, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.01, 0.06, 0.001);

            using (MemoryStream imageTrainStream = new MemoryStream(Properties.Resources.train_images))
            using (MemoryStream labelTrainStream = new MemoryStream(Properties.Resources.train_labels))
            {
                mnistTrainSet = ReadMNIST(imageTrainStream, labelTrainStream);

                backProp.MaxEpoch = 1000;
                backProp.BatchSize = 1;

                //backProp.Learn(mnistTrainSet);
                net = backProp.CrossValidationLearn(mnistTrainSet, 2);

                mnistTrainSet = null;

                using (MemoryStream imageTestStream = new MemoryStream(Properties.Resources.train_images))
                using (MemoryStream labelTestStream = new MemoryStream(Properties.Resources.train_labels))
                {
                    mnistTestSet = ReadMNIST(imageTestStream, labelTestStream);

                    double accuracy = RunMNISTTest(net, mnistTestSet);
                    Console.WriteLine("Success ratio: {0}", accuracy);
                }
            }

        }

        #endregion

        #region Unipi test

        static Dataset ReadExamDataset(StringReader stream)
        {
            try
            {
                Dataset dataset = new Dataset();
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ',' };
                int c = 0;


                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    System.Globalization.CultureInfo us = new System.Globalization.CultureInfo("en-us");
                    double[] output = { Double.Parse(strings[strings.Length - 2], us), 
                                        Double.Parse(strings[strings.Length - 1], us) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 3).ToArray();
                    double[] input = new double[strings.Length];

                    for (int index = 0; index < input.Length; index++)
                        input[index] = Double.Parse(strings[index], us);

                    dataset.Add(new Sample(input, output));

                    trainingExample = stream.ReadLine();
                }

                return dataset;
            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        static Matrix<double> NormalizeMatrix(Matrix<double> mat)
        {
            Tuple<double, double>[] mean_stdDev = new Tuple<double, double>[mat.ColumnCount];

            for (int c = 0; c < mat.ColumnCount; c++)
                mean_stdDev[c] = Statistics.MeanStandardDeviation(mat.Column(c));


            for (int r = 0; r < mat.RowCount; r++)
                for (int c = 0; c < mat.ColumnCount; c++)
                {
                    mat.At(r, c, (mat.At(r, c) - mean_stdDev[c].Item1) / mean_stdDev[c].Item2);
                }

            return mat;
        }

        static Dataset ReadExamDatasetAndNormalize(StringReader stream)
        {
            try
            {
                Dataset dataset = new Dataset();
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ',' };
                int r = 0;
                Matrix<double> tmp = null;
                double[] values = null;

                while (trainingExample != null)
                {
                    string[] strings = trainingExample.Split(separator);
                    if (values == null)
                        values = new double[strings.Length - 1];
                    else
                        Array.Clear(values, 0, values.Length);

                    for (int index = 1; index < strings.Length; index++)
                        values[index - 1] = Double.Parse(strings[index], System.Globalization.CultureInfo.InvariantCulture);

                    if (tmp == null)
                        tmp = Matrix<double>.Build.DenseOfRowVectors(Vector<double>.Build.DenseOfArray(values));
                    else
                        tmp = tmp.InsertRow(r, Vector<double>.Build.DenseOfArray(values));

                    r++;
                    trainingExample = stream.ReadLine();
                }

                tmp = NormalizeMatrix(tmp);

                for (int actualRow = 0; actualRow < r; actualRow++)
                {
                    Vector<double> input = tmp.Row(actualRow, 0, 10);
                    Vector<double> output = tmp.Row(actualRow, 10, 2);

                    dataset.Add(new Sample(input, output));

                    input = output = null;
                }

                return dataset;
            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static double RunExamTest(NeuralNet net, Dataset testSet, bool flag = false)
        {
            int numOfExamples = testSet.Size;
            double error = 0.0;

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                Vector<double> netOutput = net.Output;
                Vector<double> expected = sample.Output;
                if (flag)
                {
                    Console.WriteLine("{0}[0]) {1} / {2}", i, netOutput.At(0), expected.At(0));
                    Console.WriteLine("{0}[1]) {1} / {2}", i, netOutput.At(1), expected.At(1));
                    //Console.WriteLine("{0} {1}", netOutput.At(0).ToString(System.Globalization.CultureInfo.InvariantCulture), 
                        //netOutput.At(1).ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                Vector<double> errorVector = expected - netOutput;
                error += errorVector.DotProduct(errorVector);
            }

            error /= 2;
            error /= numOfExamples;

            return error;
        }

        private static double RunExamTestAccuracy(NeuralNet net, Dataset testSet)
        {
            double successRatio = 0.0;
            int numOfExamples = testSet.Size;
            double errorLimit = 0.1;

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                Vector<double> netOutput = net.Output;
                Vector<double> expected = sample.Output;
                Vector<double> errorVector = expected - netOutput;
                double error = errorVector.DotProduct(errorVector);
                error /= 2;

                successRatio += (error <= errorLimit) ? 1 : 0;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        private static void TestAA1Exam()
        {
            Dataset trainSet;
            Dataset testSet;

            #region Testing AA1 Dataset
            int[] layerSize = { 15, 2 };
            IActivationFunction[] functions = { new SigmoidFunction(), new LinearFunction() };
            NeuralNet net = new NeuralNet(10, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.02, 0, 0.0001);

            using (StringReader trainSetStream = new StringReader(Properties.Resources.AA1_trainset))
            using (StringReader testSetStream = new StringReader(Properties.Resources.AA1_testset))
            {
                trainSet = ReadExamDatasetAndNormalize(trainSetStream);
                testSet = ReadExamDatasetAndNormalize(testSetStream);
                
                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("AA1 Exam Test");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the error is {0}", RunExamTest(net, testSet));
                Console.WriteLine("Before training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("Train the network...");
                //net = backProp.CrossValidationLearn(trainSet, 10);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.3, 0, 0.01, 0.3, 10);

                Console.WriteLine("done!");

                Console.WriteLine("After training the error is {0}", RunExamTest(net, testSet, false));
                Console.WriteLine("After training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion
        }

        #endregion

        /*
        

        #region Regression test

        static Tuple<double[][], double[][]> ReadMPGDataset(StringReader stream, string outputFile = null)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ',' };

                List<double> inputs = new List<double>();
                List<double[]> inputsArray = new List<double[]>();
                List<double[]> outputs = new List<double[]>();
                int c = 0;
                StreamWriter writer = null;


                if (outputFile != null)
                    writer = new StreamWriter(new FileStream(outputFile, FileMode.CreateNew));

                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    System.Globalization.CultureInfo us = new System.Globalization.CultureInfo("en-us");
                    double[] output = { Double.Parse(strings[0], us) };
                    double input = Double.Parse(strings[3], us);

                    inputs.Add(input);
                    outputs.Add(output);

                    trainingExample = stream.ReadLine();
                }

                //double mean = Statistics.Mean(inputs);
                //double std = Statistics.StandardDeviation(inputs);

                //foreach (double input in inputs)
                //    inputsArray.Add(new double[] { (input - mean) / std });

                double max = inputs.Max();
                double min = inputs.Min();
                double diff = max - min;

                inputs = inputs.Select((x) => (x - min) / diff).ToList();

                double[][] inputList = inputsArray.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static void TestRegression()
        {
            Tuple<double[][], double[][]> dataset;

            double[][] trainingExamples = { new double[] {0.0000000e+00},
                                            new double[] {4.9811206e-02},
                                            new double[] {9.9622411e-02},
                                            new double[] {1.5496820e-01},
                                            new double[] {2.1031398e-01},
                                            new double[] {2.6565976e-01},
                                            new double[] {3.2100555e-01},
                                            new double[] {3.8250086e-01},
                                            new double[] {4.4399618e-01},
                                            new double[] {5.1232431e-01},
                                            new double[] {5.8065244e-01},
                                            new double[] {6.5657258e-01},
                                            new double[] {7.4092829e-01},
                                            new double[] {8.3465687e-01},
                                            new double[] {9.3879972e-01},
                                            new double[] {1.0673712e+00},
                                            new double[] {1.2102283e+00},
                                            new double[] {1.3689585e+00},
                                            new double[] {1.5453253e+00},
                                            new double[] {1.7040555e+00},
                                            new double[] {1.8469126e+00},
                                            new double[] {1.9897697e+00},
                                            new double[] {2.1326269e+00},
                                            new double[] {2.2754840e+00},
                                            new double[] {2.4183412e+00},
                                            new double[] {2.5611983e+00},
                                            new double[] {2.7040555e+00},
                                            new double[] {2.8469126e+00},
                                            new double[] {2.9897697e+00},
                                            new double[] {3.1326269e+00},
                                            new double[] {3.2754840e+00},
                                            new double[] {3.4342142e+00},
                                            new double[] {3.5929443e+00},
                                            new double[] {3.7693112e+00},
                                            new double[] {3.9456780e+00},
                                            new double[] {4.1220449e+00},
                                            new double[] {4.2984117e+00},
                                            new double[] {4.4747786e+00},
                                            new double[] {4.6511454e+00},
                                            new double[] {4.8275122e+00},
                                            new double[] {4.9862424e+00},
                                            new double[] {5.1449726e+00},
                                            new double[] {5.3037027e+00},
                                            new double[] {5.4465599e+00},
                                            new double[] {5.5894170e+00},
                                            new double[] {5.7322741e+00},
                                            new double[] {5.8910043e+00},
                                            new double[] {6.0673712e+00},
                                            new double[] {6.2437380e+00},
                                            new double[] {6.3865951e+00},
                                            new double[] {6.5294523e+00},
                                            new double[] {6.6451666e+00},
                                            new double[] {6.7388951e+00},
                                            new double[] {6.8232509e+00},
                                            new double[] {6.8991710e+00},
                                            new double[] {6.9674991e+00},
                                            new double[] {7.0289944e+00},
                                            new double[] {7.0904898e+00},
                                            new double[] {7.1458355e+00},
                                            new double[] {7.2011813e+00},
                                            new double[] {7.2565271e+00},
                                            new double[] {7.3118729e+00},
                                            new double[] {7.3616841e+00},
                                            new double[] {7.4114953e+00},
                                            new double[] {7.4613065e+00},
                                            new double[] {7.5166523e+00},
                                            new double[] {7.5719981e+00},
                                            new double[] {7.6273439e+00},
                                            new double[] {7.6826896e+00},
                                            new double[] {7.7441850e+00},
                                            new double[] {7.8056803e+00},
                                            new double[] {7.8740084e+00},
                                            new double[] {7.9499285e+00},
                                            new double[] {8.0342843e+00},
                                            new double[] {8.1384271e+00},
                                            new double[] {8.2812843e+00},
                                            new double[] {8.4576511e+00},
                                            new double[] {8.6005082e+00},
                                            new double[] {8.7162225e+00},
                                            new double[] {8.8099511e+00},
                                            new double[] {8.8943068e+00},
                                            new double[] {8.9702270e+00},
                                            new double[] {9.0461471e+00},
                                            new double[] {9.1144752e+00},
                                            new double[] {9.1828034e+00},
                                            new double[] {9.2511315e+00},
                                            new double[] {9.3194596e+00},
                                            new double[] {9.3877877e+00},
                                            new double[] {9.4637079e+00},
                                            new double[] {9.5396280e+00},
                                            new double[] {9.6239837e+00},
                                            new double[] {9.7177123e+00},
                                            new double[] {9.8334266e+00},
                                            new double[] {9.9762837e+00} };

            double[][] expectedOutputs = {  new double[] {5.0472287e+00},
                                            new double[] {5.3578233e+00},
                                            new double[] {5.6631773e+00},
                                            new double[] {5.9954650e+00},
                                            new double[] {6.3194714e+00},
                                            new double[] {6.6342607e+00},
                                            new double[] {6.9389189e+00},
                                            new double[] {7.2644727e+00},
                                            new double[] {7.5752742e+00},
                                            new double[] {7.9019937e+00},
                                            new double[] {8.2078077e+00},
                                            new double[] {8.5216219e+00},
                                            new double[] {8.8365981e+00},
                                            new double[] {9.1432246e+00},
                                            new double[] {9.4288621e+00},
                                            new double[] {9.7006661e+00},
                                            new double[] {9.8995281e+00},
                                            new double[] {1.0000000e+01},
                                            new double[] {9.9785747e+00},
                                            new double[] {9.8589252e+00},
                                            new double[] {9.6875629e+00},
                                            new double[] {9.4722185e+00},
                                            new double[] {9.2283179e+00},
                                            new double[] {8.9700894e+00},
                                            new double[] {8.7099098e+00},
                                            new double[] {8.4578833e+00},
                                            new double[] {8.2216585e+00},
                                            new double[] {8.0064628e+00},
                                            new double[] {7.8153166e+00},
                                            new double[] {7.6493772e+00},
                                            new double[] {7.5083595e+00},
                                            new double[] {7.3793283e+00},
                                            new double[] {7.2769567e+00},
                                            new double[] {7.1912086e+00},
                                            new double[] {7.1319242e+00},
                                            new double[] {7.0971852e+00},
                                            new double[] {7.0866290e+00},
                                            new double[] {7.1014417e+00},
                                            new double[] {7.1439651e+00},
                                            new double[] {7.2168785e+00},
                                            new double[] {7.3099832e+00},
                                            new double[] {7.4287298e+00},
                                            new double[] {7.5698775e+00},
                                            new double[] {7.7102248e+00},
                                            new double[] {7.8543608e+00},
                                            new double[] {7.9901367e+00},
                                            new double[] {8.1120287e+00},
                                            new double[] {8.1811075e+00},
                                            new double[] {8.1424295e+00},
                                            new double[] {8.0056086e+00},
                                            new double[] {7.7555710e+00},
                                            new double[] {7.4617659e+00},
                                            new double[] {7.1617022e+00},
                                            new double[] {6.8444896e+00},
                                            new double[] {6.5222058e+00},
                                            new double[] {6.2041410e+00},
                                            new double[] {5.8970102e+00},
                                            new double[] {5.5721082e+00},
                                            new double[] {5.2663811e+00},
                                            new double[] {4.9499871e+00},
                                            new double[] {4.6249993e+00},
                                            new double[] {4.2936579e+00},
                                            new double[] {3.9919847e+00},
                                            new double[] {3.6889091e+00},
                                            new double[] {3.3863034e+00},
                                            new double[] {3.0529487e+00},
                                            new double[] {2.7251731e+00},
                                            new double[] {2.4056096e+00},
                                            new double[] {2.0968480e+00},
                                            new double[] {1.7694999e+00},
                                            new double[] {1.4618688e+00},
                                            new double[] {1.1468958e+00},
                                            new double[] {8.3451213e-01},
                                            new double[] {5.3908804e-01},
                                            new double[] {2.5639799e-01},
                                            new double[] {2.6310092e-02},
                                            new double[] {0.0000000e+00},
                                            new double[] {1.7872281e-01},
                                            new double[] {4.4125718e-01},
                                            new double[] {7.2073056e-01},
                                            new double[] {1.0153623e+00},
                                            new double[] {1.3092107e+00},
                                            new double[] {1.6243823e+00},
                                            new double[] {1.9214258e+00},
                                            new double[] {2.2266224e+00},
                                            new double[] {2.5355511e+00},
                                            new double[] {2.8437744e+00},
                                            new double[] {3.1468653e+00},
                                            new double[] {3.4722659e+00},
                                            new double[] {3.7799306e+00},
                                            new double[] {4.0938085e+00},
                                            new double[] {4.3986380e+00},
                                            new double[] {4.6956321e+00},
                                            new double[] {4.9132400e+00} };
            


            int[] layerSize = { 10, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new LinearFunction() };
            NeuralNet net = new NeuralNet(1, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.01, 0, 0);


            backProp.MaxEpoch = 10000;
            backProp.BatchSize = 1;

                
            backProp.Learn(trainingExamples, expectedOutputs);

            ShowRegressionResults(net, trainingExamples);
            
        }

        private static void ShowRegressionResults(NeuralNet net, double[][] examples)
        {
            System.Globalization.CultureInfo us = new System.Globalization.CultureInfo("en-us");
            StreamWriter writer = new StreamWriter(new FileStream(@"C:\Users\Gabriele\neural-net\matlab\output_mynet.txt", FileMode.Create));

            for (int i = 0; i < examples.Length; i++)
            {
                net.ComputeOutput(examples[i]);
                writer.WriteLine(net.Output.At(0).ToString(us));
            }

            writer.Close();
        }

        #endregion

        private static Tuple<double[][], double[][]> ReadSinData(StreamReader stream)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample = trainingExample.Trim();
                char[] separator = { ',' };

                List<double> inputs = new List<double>();
                List<double[]> inputsArray = new List<double[]>();
                List<double[]> outputs = new List<double[]>();
                int c = 0;


                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    System.Globalization.CultureInfo us = new System.Globalization.CultureInfo("en-us");
                    double[] output = { Double.Parse(strings[1], System.Globalization.CultureInfo.InvariantCulture) };
                    double input = Double.Parse(strings[0], us);

                    inputs.Add(input);
                    outputs.Add(output);

                    trainingExample = stream.ReadLine();
                }

                double mean = Statistics.Mean(inputs);
                double std = Statistics.StandardDeviation(inputs);

                int newMin = -1;
                int newMax = 1;
                double max = inputs.Max();
                double min = inputs.Min();
                double diff = max - min;

                //inputs = inputs.Select((x) => ((newMax - newMin) * (x - min)) / (max - min)).ToList();
                
                foreach (double input in inputs)
                    inputsArray.Add(new double[] { input });

                double[][] inputList = inputsArray.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static void SinRegressionResults(NeuralNet net, double[][] examples, string outfileName, string ext)
        {
            System.Globalization.CultureInfo us = new System.Globalization.CultureInfo("en-us");
            StreamWriter writer = new StreamWriter(new FileStream(@"C:\Users\Gabriele\neural-net\matlab\"+outfileName+"."+ext, FileMode.Create));

            for (int i = 0; i < examples.Length; i++)
            {
                net.ComputeOutput(examples[i]);
                writer.WriteLine(net.Output.At(0).ToString(us));
            }

            writer.Close();
        }

        private static void SinRegression()
        {
            Tuple<double[][], double[][]> dataset;

            double[][] input;
            double[][] target;

            using (StreamReader sinDataset = File.OpenText(@"C:\Users\Gabriele\neural-net\matlab\sinDataset.txt"))
            {
                dataset = ReadSinData(sinDataset);
                input = dataset.Item1;
                target = dataset.Item2;
            }

            int[] layerSize = { 10, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new LinearFunction() };
            NeuralNet net = new NeuralNet(1, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.01, 0, 0);


            backProp.MaxEpoch = 1000;
            backProp.BatchSize = 1;

            for (double eta = 0.1; eta <= 1; eta += 0.1)
            {
                Console.WriteLine("Train with eta = " + eta);
                backProp.LearningRate = eta;
                backProp.Learn(input, target);
                SinRegressionResults(net, target, "sinOutput-" + eta,"txt");
                net.RandomizeWeights();
            }

                

            //Console.WriteLine("Hidden weights\n{0}", net[net.NumberOfLayers - 2].Weights.Print());
            //Console.WriteLine("Output weights\n{0}", net.OutputLayer.Weights.Print());

            //Console.WriteLine("Hidden bias\n{0}", net[net.NumberOfLayers - 2].Bias.ToColumnMatrix().Print());
            //Console.WriteLine("Output bias\n{0}", net.OutputLayer.Bias.At(0));
            

            
        }
        */
        static void Main()
        {
            //TestMonk();
            //TestAA1Exam();
            //TestRegression();
            //SinRegression();
            TestMNIST();
        }

    }
}
