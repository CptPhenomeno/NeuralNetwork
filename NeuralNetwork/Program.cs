using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Windows.Forms;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using NeuralNetwork.Network;
using NeuralNetwork.Learning;
using NeuralNetwork.Activation;

namespace NeuralNetwork
{
    static class Program
    {

        private static void TestXOR()
        {
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new HyperbolicTangentFunction() };

            NeuralNet net = new NeuralNet(2, layerSize, functions);

            #region Input
            Vector<double> input1 = Vector<double>.Build.Dense(new[] { 0.0, 0.0 });
            Vector<double> input2 = Vector<double>.Build.Dense(new[] { 0.0, 0.1 });
            Vector<double> input3 = Vector<double>.Build.Dense(new[] { 1.0, 0.0 });
            Vector<double> input4 = Vector<double>.Build.Dense(new[] { 1.0, 0.1 });

            Vector<double>[] inputs = { input1, input2, input3, input4 };
            #endregion

            #region Output
            Vector<double> output1 = Vector<double>.Build.Dense(new[] { 0.0 });
            Vector<double> output2 = Vector<double>.Build.Dense(new[] { 1.0 });
            Vector<double> output3 = Vector<double>.Build.Dense(new[] { 1.0 });
            Vector<double> output4 = Vector<double>.Build.Dense(new[] { 0.0 });

            Vector<double>[] outputs = { output1, output2, output3, output4 };
            #endregion

            Console.WriteLine(  "****************\n"+
                                "*    Before    *\n"+
                                "****************\n");
            net.ComputeOutput(input1);
            Console.WriteLine("output1 {0} - atteso {1}", net.Output, 0);

            net.ComputeOutput(input2);
            Console.WriteLine("output2 {0} - atteso {1}", net.Output, 1);

            net.ComputeOutput(input3);
            Console.WriteLine("output3 {0} - atteso {1}", net.Output, 1);

            net.ComputeOutput(input4);
            Console.WriteLine("output4 {0} - atteso {1}", net.Output, 0);

            BackPropagationTrainer bprop = new BackPropagationTrainer(net);
            bprop.LearningRate = 0.7;
            bprop.Learn(inputs, outputs);

            Console.WriteLine(  "\n****************\n"+
                                "*     After    *\n"+
                                "****************\n");
            net.ComputeOutput(input1);
            Console.WriteLine("output1 {0} - atteso {1}", net.Output, 0);
                                                                    
            net.ComputeOutput(input2);
            Console.WriteLine("output2 {0} - atteso {1}", net.Output, 1);
                                                                    
            net.ComputeOutput(input3);
            Console.WriteLine("output3 {0} - atteso {1}", net.Output, 1);
                                                                    
            net.ComputeOutput(input4);
            Console.WriteLine("output4 {0} - atteso {1}", net.Output, 0);
        }        

        private static Matrix<double>[] ToMatrix(double[][] array)
        {
            Matrix<double>[] m = new Matrix<double>[array.Length];
            for (int i = 0; i < array.Length; i++)
                m[i] = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(array[i]));
            return m;
        }
      
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

        static Tuple<double[][], double[][]> ReadMonkDataset(StringReader stream, string outputFile = null)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ' ' };

                List<double[]> inputs = new List<double[]>();
                List<double[]> outputs = new List<double[]>();
                int c = 0;
                StreamWriter writer = null;


                if (outputFile != null)
                    writer = new StreamWriter(new FileStream(outputFile, FileMode.CreateNew));

                while (trainingExample != null)
                {
                    ++c;
                    string[] strings = trainingExample.Split(separator);
                    double[] output = { Double.Parse(strings[0]) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 1).ToArray();
                    double[] input = EncodingInput(strings);

                    inputs.Add(input);
                    outputs.Add(output);

                    if (writer != null)
                    {
                        foreach (double d in input)
                            writer.Write(d.ToString() + " ");
                        foreach (double d in output)
                            writer.Write(d.ToString() + " ");
                        writer.WriteLine();
                    }

                    trainingExample = stream.ReadLine();
                }

                if(writer != null)
                    writer.Close();

                double[][] inputList = inputs.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static double RunMonkTest(NeuralNet net, double[][] input, double[][] output)
        {
            double successRatio = 0.0;
            int numOfExamples = input.Length;

            for (int i = 0; i < input.Length; i++)
            {
                net.ComputeOutput(input[i]);
                double netOutput = net.Output.At(0);
                double roundOutput = Math.Round(netOutput);
                double expected = output[i][0];

                successRatio += (expected == roundOutput) ? 1 : 0;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        private static void TestMonk()
        {
            Tuple<double[][], double[][]> dataset;
            Tuple<double[][], double[][]> testset;

            double[][] trainingExamples;
            double[][] expectedOutputs;

            double[][] testInput;
            double[][] testOutput;

            #region Testing Monk Dataset 1
            int[] layerSize = { 3, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(17, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.3);


            using (StringReader trainSet = new StringReader(Properties.Resources.monks_1_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_1_test))
            {
                dataset = ReadMonkDataset(trainSet);
                testset = ReadMonkDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 1");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.Learn(trainingExamples, expectedOutputs);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 2
            layerSize = new[] { 3, 1 };
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagationTrainer(net, 0.7, 0.3);


            using (StringReader trainSet = new StringReader(Properties.Resources.monks_2_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_2_test))
            {
                dataset = ReadMonkDataset(trainSet);
                testset = ReadMonkDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 2");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.Learn(trainingExamples, expectedOutputs);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion

            #region Testing Monk Dataset 3
            layerSize = new[] { 3, 1 };
            net = new NeuralNet(17, layerSize, functions);
            backProp = new BackPropagationTrainer(net, 0.7, 0.3);


            using (StringReader trainSet = new StringReader(Properties.Resources.monks_3_train))
            using (StringReader testSet = new StringReader(Properties.Resources.monks_3_test))
            {
                dataset = ReadMonkDataset(trainSet);
                testset = ReadMonkDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                backProp.MaxEpoch = 10000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Monk Dataset 3");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.Learn(trainingExamples, expectedOutputs);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunMonkTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion
        }
       
        #endregion

        #region Unipi test
        
        static Tuple<double[][], double[][]> ReadExamDataset(StringReader stream, string outputFile = null)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ',' };

                List<double[]> inputs = new List<double[]>();
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
                    double[] output = { Double.Parse(strings[strings.Length - 2], us), 
                                        Double.Parse(strings[strings.Length - 1], us) };
                    List<string> stringList = new List<string>(strings);
                    strings = stringList.GetRange(1, strings.Length - 3).ToArray();
                    double[] input = new double[strings.Length];

                    for (int index = 0; index < input.Length; index++)
                        input[index] = Double.Parse(strings[index], us);

                    inputs.Add(input);
                    outputs.Add(output);

                    if (writer != null)
                    {
                        foreach (double d in input)
                            writer.Write(d.ToString() + " ");
                        foreach (double d in output)
                            writer.Write(d.ToString() + " ");
                        writer.WriteLine();
                    }

                    trainingExample = stream.ReadLine();
                }

                if (writer != null)
                    writer.Close();

                double[][] inputList = inputs.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static double RunExamTest(NeuralNet net, double[][] input, double[][] output)
        {
            double successRatio = 0.0;
            int numOfExamples = input.Length;
            double errorLimit = 0.01;

            for (int i = 0; i < input.Length; i++)
            {
                net.ComputeOutput(input[i]);
                Vector<double> netOutput = net.Output;
                Vector<double> expected = Vector.Build.DenseOfArray(output[i]);
                Vector<double> errorVector = expected - netOutput;
                double error = errorVector.DotProduct(errorVector);
                error /= 2;

                successRatio += (error <= errorLimit) ? 1 : 0;
            }

            return successRatio / (double)numOfExamples * 100.0;
        }

        private static void TestAA1Exam()
        {
            Tuple<double[][], double[][]> dataset;
            Tuple<double[][], double[][]> testset;

            double[][] trainingExamples;
            double[][] expectedOutputs;

            double[][] testInput;
            double[][] testOutput;

            #region Testing Monk Dataset 1
            int[] layerSize = { 10, 2 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(10, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.5, 0.01);


            using (StringReader trainSet = new StringReader(Properties.Resources.AA1_trainset))
            using (StringReader testSet = new StringReader(Properties.Resources.AA1_testset))
            {
                dataset = ReadExamDataset(trainSet);
                testset = ReadExamDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                backProp.MaxEpoch = 1000000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("AA1 Exam Test");
                Console.WriteLine("*******************");
                Console.WriteLine("Before training the success ratio is {0}", RunExamTest(net, testInput, testOutput));

                Console.Write("Train the network...");
                backProp.Learn(trainingExamples, expectedOutputs);
                Console.WriteLine("done!");

                Console.WriteLine("After training the success ratio is {0}", RunExamTest(net, testInput, testOutput));

                Console.WriteLine("*******************");
            }
            #endregion
        }

        #endregion

        #region Sin regression test

        static Tuple<double[][], double[][]> ReadSinDataset(StringReader stream, string outputFile = null)
        {
            try
            {
                string trainingExample = stream.ReadLine();
                trainingExample.TrimEnd();
                char[] separator = { ',' };

                List<double[]> inputs = new List<double[]>();
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
                    double[] output = { Double.Parse(strings[1], us) };
                    double[] input = { Double.Parse(strings[0], us) };

                    inputs.Add(input);
                    outputs.Add(output);

                    if (writer != null)
                    {
                        foreach (double d in input)
                            writer.Write(d.ToString() + " ");
                        foreach (double d in output)
                            writer.Write(d.ToString() + " ");
                        writer.WriteLine();
                    }

                    trainingExample = stream.ReadLine();
                }

                if (writer != null)
                    writer.Close();

                double[][] inputList = inputs.ToArray();
                double[][] outputList = outputs.ToArray();

                return new Tuple<double[][], double[][]>(inputList, outputList);


            }
            catch (UnauthorizedAccessException e)
            {
                Console.WriteLine(e);
                return null;
            }
        }

        private static void TestSinRegression()
        {
            Tuple<double[][], double[][]> dataset;
            Tuple<double[][], double[][]> testset;

            double[][] trainingExamples;
            double[][] expectedOutputs;

            double[][] testInput;
            double[][] testOutput;

            #region Testing sin regression
            int[] layerSize = { 5, 1 };
            IActivationFunction[] functions = { new SigmoidFunction(), new SigmoidFunction() };
            NeuralNet net = new NeuralNet(1, layerSize, functions);
            BlockingCollection<string> data = new BlockingCollection<string>(100);
            BackPropagationTrainer backProp = new BackPropagationTrainer(net, 0.5, 0.3);


            using (StringReader trainSet = new StringReader(Properties.Resources.sinData))
            using (StringReader testSet = new StringReader(Properties.Resources.sinTest))
            {
                dataset = ReadSinDataset(trainSet);
                testset = ReadSinDataset(testSet);

                trainingExamples = dataset.Item1;
                expectedOutputs = dataset.Item2;

                testInput = testset.Item1;
                testOutput = testset.Item2;

                backProp.MaxEpoch = 1000;
                backProp.BatchSize = 1;

                Console.WriteLine("*******************");
                Console.WriteLine("Sin Regression Test");
                Console.WriteLine("*******************");
                Console.Write("Train the network...");
                backProp.Learn(trainingExamples, expectedOutputs);
                Console.WriteLine("done!");
                
                ShowSinRegressionResults(net, testInput, testOutput);

                Console.WriteLine("*******************");
            }
            #endregion
        }

        private static void ShowSinRegressionResults(NeuralNet net, double[][] input, double[][] output)
        {
            int numOfExamples = input.Length;

            for (int i = 0; i < input.Length; i++)
            {
                net.ComputeOutput(input[i]);
                double netOutput = net.Output.At(0);
                Console.WriteLine(netOutput);
            }
        }

        #endregion

        static void Main()
        {
            //TestMonk();
            //TestAA1Exam();
            TestSinRegression();
        }

    }
}
