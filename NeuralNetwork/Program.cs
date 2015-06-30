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
using NeuralNetwork.ErrorFunctions;
using NeuralNetwork.Utils.Extensions;

using DatasetUtility;
using MNIST_Reader;
using System.Diagnostics;

namespace NeuralNetwork
{    
    static class Program
    {
        private const string basePath = @"D:\home\Documents\Informatics\unipi\aa1\project\output";

        #region Monk test
       
        private static double[] Encode(int value, byte encode)
        {
            double[] encoded = new double[encode];
            encoded[value - 1] = 1;
            return encoded;
        }

        private static double[] EncodingInput(string[] inputs)
        {
            List<double> input = new List<double>();
            byte[] encoding = { 3, 3, 2, 3, 4, 2 };

            for (int i = 0; i < inputs.Length; i++)
            {
                input.InsertRange(input.Count, Encode(Int32.Parse(inputs[i]), encoding[i]));
            }
            return input.ToArray();
        }

        private static Dataset ReadMonkDataset(StringReader stream)
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

        private static void TestMonk1(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 3, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, ErrorFunctionContainer.SQUARED_ERROR, 0.1, 0.5, 0, 0.0001);

            using (StringReader trainSetStream = new StringReader(Properties.Resources.monks_1_train))
            using (StringReader testSetStream = new StringReader(Properties.Resources.monks_1_test))
            {
                trainSet = ReadMonkDataset(trainSetStream);
                testSet = ReadMonkDataset(testSetStream);

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 1");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testSet));

                Console.WriteLine("Train the network...");

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = trainSet.Size;

                Console.WriteLine("done!");

                net = backProp.Learn(trainSet, basePath + @"\monk1\monk-1-learning_curves.log", testSet);
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, basePath+@"\monk1\monk-1-learning_curves.log", testSet);
                
                Console.WriteLine("After the training the accuracy is: {0}", RunMonkTest(net, testSet, basePath+@"\monk1\monk1-out.csv"));

                Console.WriteLine("*******************");
            }
        }

        private static void TestMonk2(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 2, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, ErrorFunctionContainer.SQUARED_ERROR, 0.1, 0.5, 0, 0.0001);

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


                net = backProp.Learn(trainSet, basePath + @"\monk2\monk-2-learning_curves.log", testSet);
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, basePath+@"\monk2\monk-2-learning_curves.log", testSet);

                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet, basePath+@"\monk2\monk2-out.csv"));

                Console.WriteLine("*******************");
            }
        }

        private static void TestMonk3(IActivationFunction[] functions, double[] etaValues, double[] alphaValues, double[] lambdaValues)
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 4, 1 };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, ErrorFunctionContainer.SQUARED_ERROR, 0.1, 0.5, 0, 0.0001);

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


                net = backProp.Learn(trainSet, basePath + @"\monk3\monk-3-learning_curves.log", testSet);
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 3, 10, basePath+@"\monk3\monk-3-learning_curves.log", testSet);

                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testSet, basePath+@"\monk3\monk3-out.csv"));

                Console.WriteLine("*******************");
            }
        }

        private static void TestMonk()
        {
            double[] etaValues = { 0.01, 0.1 };
            double[] alphaValues = { 0 };
            double[] lambdaValues = { 0 };

            IActivationFunction hyperbolic = new HyperbolicTangentFunction();
            IActivationFunction sigmoid = new SigmoidFunction();
            IActivationFunction linear = new LinearFunction();

            IActivationFunction[] functions = { hyperbolic, sigmoid };

            TestMonk1(functions, etaValues, alphaValues, lambdaValues);

            TestMonk2(functions, etaValues, alphaValues, lambdaValues);

            TestMonk3(functions, etaValues, alphaValues, lambdaValues);
        }
       
        #endregion

        #region Unipi test

        private static Dataset ReadExamDataset(StringReader stream)
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
        
        private static Matrix<double> Unipi_NormalizeMatrixMinMax(Matrix<double> mat)
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
                    //if (c < 10)
                        value = value * 2 - 1;
                    mat.At(r, c, value);
                    
                }

            return mat;
        }
        
        private static Matrix<double> NormalizeMatrixMeanStd(Matrix<double> mat)
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
        
        private static Dataset ReadExamDatasetAndNormalize(StringReader stream)
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

                tmp = Unipi_NormalizeMatrixMinMax(tmp);

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

        private static double RunExamTest(NeuralNet net, Dataset testSet, ErrorFunction errorFunction, bool flag = false)
        {
            int numOfExamples = testSet.Size;
            double error = 0.0;

            StreamWriter outputFile = new StreamWriter(basePath+@"\test-aa1\aa1-test-res.txt", false);
            StreamWriter expectedFile = new StreamWriter(basePath + @"\test-aa1\aa1-test-exp.txt", false);
            

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
                error += errorFunction(expected, netOutput);
            }

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
            int[] layerSize = { 10, 2 };

            IActivationFunction hyperbolic = new HyperbolicTangentFunction();
            IActivationFunction sigmoid = new SigmoidFunction();
            IActivationFunction linear = new LinearFunction();

            IActivationFunction[] functions = { hyperbolic, linear };
            NeuralNet net = new NeuralNet(10, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, ErrorFunctionContainer.EUCLIDEAN_ERROR, 0.03, 0, 0, 0.0001);
            
            double[] etaValues = { 0.05 };
            double[] alphaValues = { 0.02 };
            double[] lambdaValues = { 0.001 };
            

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
                Console.WriteLine("Before training the error is {0}", RunExamTest(net, testSet, ErrorFunctionContainer.EUCLIDEAN_ERROR));
                Console.WriteLine("Before training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("Train the network...");
                //net = backProp.CrossValidationLearn(trainSet, 10);
                //net = backProp.CrossValidationLearnWithModelSelection(trainSet, 0.01, 0.01, 0.3, 0, 0.01, 0.3, 10);
                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 10, 10, basePath + @"\test-aa1\aa1-learning_curves.log", testSet);
                //net = backProp.Learn(trainSet, basePath + @"\test-aa1\aa1-learning_curves.log", testSet);
                Console.WriteLine("done!");

                Console.WriteLine("After training the error is {0}", RunExamTest(net, testSet, ErrorFunctionContainer.EUCLIDEAN_ERROR));
                //Console.WriteLine("After training the accuracy is {0}", RunExamTestAccuracy(net, testSet));

                Console.WriteLine("*******************");
            }
            #endregion
        }
        
        #endregion

        #region Multiclassifier gaussian

        private static Matrix<double> MCG_NormalizeMatrixMinMax(Matrix<double> mat)
        {
            Tuple<double, double>[] min_max = new Tuple<double, double>[mat.ColumnCount];

            for (int c = 0; c < 2; c++)
            {
                min_max[c] = new Tuple<double, double>(Statistics.Minimum(mat.Column(c)),
                                                       Statistics.Maximum(mat.Column(c)));
            }


            for (int r = 0; r < mat.RowCount; r++)
                for (int c = 0; c < 2; c++)
                {
                    double value = ((mat.At(r, c) - min_max[c].Item1) / (min_max[c].Item2 - min_max[c].Item1));
                    value = value * 2 - 1;
                    mat.At(r, c, value);
                }

            return mat;
        }

        private static Dataset ReadMCGExamDatasetAndNormalize(StringReader stream)
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
                int index = 0;

                while (trainingExample != null)
                {
                    string[] strings = trainingExample.Split(separator);
                    if (values == null)
                        values = new double[5];
                    else
                        Array.Clear(values, 0, values.Length);

                    for (index = 0; index < strings.Length - 1; index++)
                        values[index] = Double.Parse(strings[index], System.Globalization.CultureInfo.InvariantCulture);
                    
                    Array.Copy(Encode(Int32.Parse(strings[index], System.Globalization.CultureInfo.InvariantCulture), 3), 0,
                               values, index, 3);

                    if (tmp == null)
                        tmp = Matrix<double>.Build.DenseOfRowVectors(Vector<double>.Build.DenseOfArray(values));
                    else
                        tmp = tmp.InsertRow(r, Vector<double>.Build.DenseOfArray(values));

                    r++;
                    trainingExample = stream.ReadLine();
                }

                tmp = MCG_NormalizeMatrixMinMax(tmp);

                for (int actualRow = 0; actualRow < r; actualRow++)
                {
                    Vector<double> input = tmp.Row(actualRow, 0, 2);
                    Vector<double> output = tmp.Row(actualRow, 2, 3);

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

        private static double TestAccuracy(NeuralNet net, Dataset testSet, string filepath = null)
        {
            double accuracy = 0.0;

            foreach (Sample s in testSet.Samples)
            {
                int expectedClass = s.Output.MaximumIndex();
                net.ComputeOutput(s.Input);
                int obtainedClass = net.Output.MaximumIndex();

                if (obtainedClass == expectedClass)
                    ++accuracy;
            }

            return accuracy / testSet.Size;
        }

        private static void TestMCG()
        {
            Dataset trainSet;
            Dataset testSet;

            int[] layerSize = { 10, 3 };

            IActivationFunction hyperbolic = new HyperbolicTangentFunction();
            IActivationFunction sigmoid = new SigmoidFunction();
            IActivationFunction linear = new LinearFunction();
            IActivationFunction softmax = new SoftmaxFunction();

            IActivationFunction[] functions = { hyperbolic, softmax };
            NeuralNet net = new NeuralNet(2, layerSize, functions);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, ErrorFunctionContainer.SQUARED_ERROR, 0.1, 0.01, 0.0001, 0.0001);

            double[] etaValues = { 0.1, 0.3, 0.5 };
            double[] alphaValues = { 0, 0.01, 0.05 };
            double[] lambdaValues = { 0, 0.0001, 0.001 };

            using (StringReader trainSetStream = new StringReader(Properties.Resources.train_gaussian))
            using (StringReader testSetStream = new StringReader(Properties.Resources.test_gaussian))
            {
                trainSet = ReadMCGExamDatasetAndNormalize(trainSetStream);
                testSet = ReadMCGExamDatasetAndNormalize(testSetStream);

                //backProp.MaxEpoch = 1000;
                backProp.BatchSize = trainSet.Size;

                net = backProp.CrossValidationLearnWithModelSelection(trainSet, etaValues, alphaValues, lambdaValues, 10, 10, basePath + @"\multigaussian\multigaussian-learning_curves.log", testSet);

                double accuracy = TestAccuracy(net, testSet);

                Console.WriteLine("Accuracy for multi gaussian problem: {0}", accuracy);
            }
        }

        #endregion

        public static void Main()
        {
			TestMonk();
            //TestAA1Exam();
            //TestMCG();
        }
    }
}
