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
                    double[] output = { Double.Parse(strings[0]) * 2 - 1 };
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

        private static double RunMonkTest(NeuralNet net, Dataset testSet, string path = null)
        {
            double successRatio = 0.0;
            int numOfExamples = testSet.Size;

            StreamWriter writer = null;

            if (path != null)
                writer = new StreamWriter(path, false);

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                double netOutput = net.Output.At(0);
                double roundOutput = Math.Round(netOutput);
                double expected = sample.Output.At(0);

                if (writer != null)
                    writer.WriteLine("{0},{1}", roundOutput.ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                expected.ToString(System.Globalization.CultureInfo.InvariantCulture));

                successRatio += (expected == roundOutput) ? 1 : 0;
            }

            if (writer != null)
                writer.Close();

            return successRatio / (double)numOfExamples;
        }

        private static double TestMonk1(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 3, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.3, 0, 0, 0.0001);

            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_1_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_1_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = trainSet.Size;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 1");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("Train the network...");


                //net = backProp.Learn(trainSet, @"C:\Users\Gabriele\neural-net\matlab\monk-1-learning_curves.log", testSet);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk-1-learning_curves.log", testSet);

                Console.WriteLine("done!");
                double acc = 0;
                Console.WriteLine("After training the success ratio is {0}", acc = RunMonkTest(net, testSet, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk1-out.csv"));

                Console.WriteLine("*******************");

                return acc;
            }
        }

        private static void TestMonk2(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 3, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.3, 0, 0, 0.0001);

            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_2_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_2_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = trainSet.Size;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 2");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("Train the network...");


                //net = backProp.Learn(trainSet, @"C:\Users\Gabriele\neural-net\matlab\monk-1-learning_curves.log", testSet);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk-1-learning_curves.log", testSet);

                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk1-out.csv"));

                Console.WriteLine("*******************");
            }
        }

        private static void TestMonk3(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 3, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.3, 0, 0, 0.0001);

            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_3_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_3_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = trainSet.Size;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 3");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("Train the network...");


                //net = backProp.Learn(trainSet, @"C:\Users\Gabriele\neural-net\matlab\monk-1-learning_curves.log", testSet);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk-1-learning_curves.log", testSet);

                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\old-release\monk1\monk1-out.csv"));

                Console.WriteLine("*******************");
            }
        }

        private static void TestMonk()
        {
            

            double[] etaValues = { 0.01, 0.1, 0.3, 0.5 };
            double[] alphaValues = { 0, 0.01, 0.05, 0.1 };
            double[] lambdaValues = { 0, 0.01, 0.001, 0.0001 };

            IActivationFunction hyperbolic = new HyperbolicTangentFunction();
            IActivationFunction sigmoid = new SigmoidFunction();
            IActivationFunction linear = new LinearFunction();

            IActivationFunction[] functions = { hyperbolic, hyperbolic };
            
            double acc = 0;
            for (int i = 0; i < 10; i++ )
                acc+=TestMonk1(functions, etaValues, alphaValues, lambdaValues);
            Console.WriteLine("accuracy: " + acc / 10.0);
            //TestMonk2(functions, etaValues, alphaValues, lambdaValues);

            //TestMonk3(functions, etaValues, alphaValues, lambdaValues);
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

        static Matrix<double> NormalizeMatrixMinMax(Matrix<double> mat)
        {
            Tuple<double, double>[] min_max = new Tuple<double, double>[mat.ColumnCount];

            for (int c = 0; c < mat.ColumnCount; c++)
            {
                min_max[c] = new Tuple<double, double>(Statistics.Minimum(mat.Column(c)),
                                                       Statistics.Maximum(mat.Column(c)));                
            }
            

            for (int r = 0; r < mat.RowCount; r++)
                for (int c = 0; c < mat.ColumnCount; c++)
                {
                    double value = ((mat.At(r, c) - min_max[c].Item1) / (min_max[c].Item2 - min_max[c].Item1));
                    if (c < 10)
                        value = value * 2 - 1;
                    mat.At(r, c, value);
                    
                }

            return mat;
        }

        static Matrix<double> NormalizeMatrixMeanStd(Matrix<double> mat)
        {
            Tuple<double, double>[] mean_std = new Tuple<double, double>[mat.ColumnCount];

            for (int c = 0; c < mat.ColumnCount; c++)
                mean_std[c] = Statistics.MeanStandardDeviation(mat.Column(c));


            for (int r = 0; r < mat.RowCount; r++)
                for (int c = 0; c < mat.ColumnCount; c++)
                {
                    mat.At(r, c, ((mat.At(r, c) - mean_std[c].Item1) / mean_std[c].Item2));
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

                tmp = NormalizeMatrixMinMax(tmp);

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

            StreamWriter outputFile = new StreamWriter(@"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\aa1-test-res.txt", false);
            StreamWriter expectedFile = new StreamWriter(@"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\aa1-test-exp.txt", false);
            

            for (int i = 0; i < numOfExamples; i++)
            {
                Sample sample = testSet[i];
                net.ComputeOutput(sample.Input);
                Vector<double> netOutput = net.Output;
                Vector<double> expected = sample.Output;

                outputFile.WriteLine("{0} {1}", netOutput.At(0).ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                netOutput.At(1).ToString(System.Globalization.CultureInfo.InvariantCulture));

                expectedFile.WriteLine("{0} {1}", expected.At(0).ToString(System.Globalization.CultureInfo.InvariantCulture),
                                                expected.At(1).ToString(System.Globalization.CultureInfo.InvariantCulture));

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

            outputFile.Close();
            expectedFile.Close();

            return error;
        }

        private static double RunExamTestAccuracy(NeuralNet net, Dataset testSet)
        {
            double successRatio = 0.0;
            int numOfExamples = testSet.Size;
            double errorLimit = 0.01;

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
            int[] layerSize = { 5, 2 };

            IActivationFunction hyperbolic = new HyperbolicTangentFunction();
            IActivationFunction sigmoid = new SigmoidFunction();
            IActivationFunction linear = new LinearFunction();

            IActivationFunction[] functions = { hyperbolic, sigmoid };
            NeuralNet net = new NeuralNet(10, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.1, 0.01, 0.0001, 0.0001);

            double[] etaValues = { 0.2, 0.3, 0.4 };
            double[] alphaValues = { 0, 0.04, 0.05, 0.06 };
            double[] lambdaValues = { 0, 0.0001, 0.0005 };

            using (StringReader trainSetStream = new StringReader(Properties.Resources.AA1_trainset))
            using (StringReader testSetStream = new StringReader(Properties.Resources.AA1_testset))
            {
                trainSet = ReadExamDatasetAndNormalize(trainSetStream);
                testSet = ReadExamDatasetAndNormalize(testSetStream);
                
                backProp.MaxEpoch = 10000;
                backProp.BatchSize = trainSet.Size;

                Console.WriteLine("*******************");
                Console.WriteLine("AA1 Exam Test");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the error is {0}", RunExamTest(net, testSet));
                Console.WriteLine("Before training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("Train the network...");
                //net = backProp.CrossValidationLearn(trainSet, 10);
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.3, 0, 0.01, 0.3, 10);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 10, 10, @"D:\FTP\home\Documents\Informatics\unipi\aa1\project\test-output\aa1-learning_curves.log", testSet);
                //net = backProp.Learn(trainSet, @"C:\Users\Gabriele\neural-net\matlab\aa1-learning_curves.log", testSet);
                Console.WriteLine("done!");

                Console.WriteLine("After training the error is {0}", RunExamTest(net, testSet, false));
                //Console.WriteLine("After training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion
        }
        #endregion


        public static void Main()
        {
			TestMonk();
            //TestAA1Exam();
        }

    }
}
